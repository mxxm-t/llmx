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
#include <future>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
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
#include "server/service_record.hpp"
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

// An image's encode as the scheduler runs it (docs/VISION.md, section 10): `piece` records the next of its `pieces` on the first stage's thread, never under a hold.
// `rows` never waits: null until the last piece's work has retired, then the image's rows.
struct ImageSteps {
    std::function<void()> piece;
    std::function<std::shared_ptr<const std::vector<float>>()> rows;
    size_t pieces = 0;
};

// An image's rows from the last whole block it reaches into, which a history forked at that block reads, so a fork inside the image encodes nothing (docs/VISION.md, section 5).
// The request and each entry holding its history keep them, counted in the entry's bytes and freed with it (Cache::keep_tail).
struct ImageTail {
    std::array<uint8_t, 32> digest{};
    size_t rows = 0, from = 0;
    std::shared_ptr<const std::vector<float>> encoded;
    size_t bytes() const { return encoded ? encoded->size() * sizeof(float) : 0; }
};

// One request from submission to completion: the connection thread reads its channel, and everything below the channel belongs to the scheduler thread.
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
    MessageSpans spans_;           // where its messages end and their roles, from the chat route (Api::message_spans), for a turn that reuses little
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
    std::vector<ImageTail> tails_;   // its images' last rows, its own encodes' and those of the history it forked
    uint32_t last_id_ = 0;
    // Drafting (docs/SPECULATIVE.md, section 3): its record, the draft its own history gave ahead of the proposer, and the verify its pass in flight carries, its last pick then the drafts, empty for a decode entry.
    infer::spec::Record record_;
    std::vector<uint32_t> own_draft_;
    std::vector<uint32_t> verify_;
    // The drafts its chain on the head stage's thread gave for its history at length chained_at_, which its next pass verifies (Scheduler::propose).
    std::vector<uint32_t> chained_;
    size_t chained_at_ = SIZE_MAX;
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
    Clock::time_point first_at_{}, pushed_{};   // when its first token and its last went to the channel, the scheduler thread's copies for the service record
    StepTokens steps_;                          // its tokens a step as a reply, which the service rule's gap follows
    std::atomic<uint64_t> steps_taken_{0}, verifies_{0}, drafted_{0}, kept_{0};   // its steps, those that verified drafts and the drafts fed and kept, for the live view
    Clock::time_point first_before_{};   // first_ as that promotion found it: unset where the pass gives the first token, so the pass is in prompt_ms
    size_t boundary_at_ = 0;       // on a model that keeps a state, where the slice that reaches it keeps the state as a message boundary; 0 once kept or skipped
    size_t boundary_kept_ = 0;     // where its pass in flight keeps that state, the end of that pass's slice, which goes to host memory once that pass retires; 0 for none
    size_t keep_at_ = 0;           // on a model that keeps a state, where the slice that reaches it keeps the state as a checkpoint; 0 once kept or skipped
    Clock::time_point rows_at_{};  // when a pass last took or gave back its prompt rows, from which the share rule counts its wait for rows
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
    // The model's context is reserved here for `passes` passes in flight (one more than the stages on a pipelined layer split and one elsewhere when none is given; passes that do not fit run fewer, and stderr says so).
    Scheduler(infer::Model& model, const bpe::Tokenizer& tok, size_t max_seqs, size_t max_queue, size_t passes = 0, bool timed = false,
              size_t host_bytes = 0, infer::spec::Proposer* proposer = nullptr, size_t draft_max = 0, bool priced = true, const DiskOptions& disk = {},
              size_t ctx_size = 0)
        : model_(model), tok_(tok), max_seqs_(max_seqs), ubatch_(model.prefill_batch()), max_queue_(max_queue), limit_(request_limit(model, ctx_size)), timed_(timed), host_cap_(host_bytes),
          wait_bound_(std::chrono::duration_cast<Clock::duration>(disk.wait ? *disk.wait : std::chrono::milliseconds(kWaitBound))), read_wait_(disk.read_wait), recompute_row_ms_(disk.recompute_row_ms),
          proposer_(draft_max ? proposer : nullptr), draft_max_(proposer ? draft_max : 0), draft_rows_(draft_rows(max_seqs, draft_max_)), priced_(priced), pools_(pools_of(model)),
          samplers_(std::min<size_t>(kSamplers, (size_t)core::automatic_threads() - 1)) {
        if (model_.keeps_state() && max_seqs_ > model_.state_slots())
            throw std::logic_error("server: " + std::to_string(max_seqs_) + " requests at once need as many recurrent state slots, and the model holds " +
                                   std::to_string(model_.state_slots()));
        // One pass more than the stages, so a prompt's next chunk continues while a pass of replies and its chunk before are in flight (docs/SERVER.md, The service rule).
        size_t p = passes ? passes : model_.pipelined() && model_.stage_count() > 1 ? model_.stage_count() + 1 : 1;
        if (p > 1 && !model_.pipelined())
            throw std::runtime_error("server: " + std::to_string(p) + " passes in flight need a layer split over several devices, each running its layers whole; this placement runs one at a time");
        // A pass carries at most draft_rows draft rows beside a row a decoding request and a ubatch of prompt rows, each draft row wanting its logits.
        const size_t drafts = draft_rows_;
        for (;; --p) {
            logit_rows_ = logit_rows(p, logits_a_pass(max_seqs_, draft_max_));
            try {
                model_.reserve_passes(ctx_, p, ubatch_ + max_seqs_ + drafts, logit_rows_.size);
                break;
            } catch (const std::logic_error&) {
                throw;
            } catch (const std::exception& e) {
                if (p == 1) throw;
                std::fprintf(stderr, "server: %zu passes in flight do not fit (%s); running %zu\n", p, e.what(), p - 1);
            }
        }
        slots_.resize(p);
        // A chain on the head stage's thread and a mark's rerun on the state stages' threads take the model's context, and the chain the drafter's buffers, as taken now (Model::reserve_drafting).
        model_.reserve_drafting();
        // A stage whose recording waits on its devices gets a thread of its own, so the round goes on to the other stages while it records (docs/SERVER.md, the round).
        recorders_.resize(model_.stage_count());
        plain_.resize(recorders_.size());
        for (size_t s = 0; s < recorders_.size(); ++s) {
            plain_[s] = (p > 1 && model_.stage_waits(s)) || (!model_.stage_on_host(s) && model_.stage_reads(s));
            if (!plain_[s]) continue;
            locks_.push_back(std::make_unique<std::mutex>());
            recorders_[s] = std::make_unique<Recorder>(*locks_.back(), lanes_.add(s, *locks_.back()));
            // The lane is given its record, not the place it is kept: a scheduler destroyed before the thread first runs is emptying that place.
            Recorder* rec = recorders_[s].get();
            infer::StageLanes::Policy policy;
            policy.ready = [rec] { return rec->pass.posted || !rec->items.empty(); };
            // Items go first: each was queued before any pass formed after it, and touches no pass formed before it.
            policy.step = [this, rec, s](std::unique_lock<std::mutex>& lk) {
                if (!rec->items.empty()) run_item(*rec, lk);
                else record(*rec, s, lk);
            };
            lanes_.start(s, std::move(policy));
        }
        // A history settling waits for its parts on the stages' threads as a hold waits for them (wake).
        model_.on_stage_wait([this](bool waiting) {
            if (waiting) {
                ++quieting_;
                ++stage_waits_;
            } else {
                --quieting_;
            }
        });
        // A job reads the next turn's ids at one extent for every extent that turn's prompt can have, so only where rows are one class from there up to the limit.
        const size_t limit = token_limit();
        steady_from_ = limit;
        while (steady_from_ > 2 && model_.row_class(steady_from_ - 1) == model_.row_class(limit)) --steady_from_;
        // The cache's copies go to the stages' threads as items of no request, each reporting to its copy (Model::copy_out).
        cache_.on_defer([this](infer::Model::StageItem it) { defer_item(nullptr, std::move(it)); });
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
                                    Request::Clock::time_point received = Request::Clock::now(), std::vector<uint32_t> cut = {}, Images images = {}, MessageSpans spans = {}) {
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
        r->spans_ = std::move(spans);
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
        // Its wait for admission (so far, while it waits), its time from reading to the first token (0 before it) and its mean gap between tokens since.
        uint64_t queued_ms = 0, first_token_ms = 0;
        double inter_token_ms = 0;
        // Its steps as a reply, those that verified drafts, and the drafts they fed and kept.
        uint64_t steps = 0, verifies = 0, drafted = 0, kept = 0;
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
        size_t reuse_misses = 0;       // first admissions of a long prompt that reused under half of it (Cache, kMissTokens), the last one's numbers beside
        ReuseMiss last_miss;
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
        DiskTier::Start disk_digest{"off", {}, "off"};   // the tier's digest and adoption (DiskTier::Start), `off` without a tier
        uint64_t disk_bytes = 0, disk_bytes_written = 0;   // their files' bytes, and every finished write's
        bool disk_writing = false;                // a disk tier that has not stopped writing
        std::vector<size_t> drafted{}, kept{};    // by draft position, the drafts verifies fed and those they kept
        std::vector<size_t> own_drafted{}, own_kept{};   // those of them requests' own histories gave ahead of the proposer (spec::lookup_first)
        std::vector<infer::spec::Acceptance> by_source{};   // the proposer's acceptance, then the history drafts' by match bucket, as the server prices drafts with them
        size_t draft_failures = 0;                // drafts that threw, each costing its pass the drafts and no request
        size_t holds = 0;                         // times the scheduler's thread stopped every stage's recorder to work on their devices itself (quiet, Held)
        size_t stage_waits = 0;                   // times the scheduler's thread waited for work it had queued on a stage's thread (Model::on_stage_wait)
        size_t yields = 0, kv_tokens = 0, kv_donor_tokens = 0;   // pauses that gave the pool's edge to a paused request (Growth::yields), and the pool's reserved blocks as tokens, each the largest over the pools: the running requests' part, and the donors'
        size_t turns_without_memory = 0;          // turns at the pool's edge kept because the host had no memory for the copy the history would leave (host_room)
        uint64_t finished = 0, prompt_tokens = 0, generated_tokens = 0;   // since start, the clients' requests that were admitted and ended, and their prompt and generated tokens
        uint64_t prefilled_tokens = 0, published_tokens = 0;   // client prompt rows retired and output tokens published, including requests still running
        uint64_t host_limit = 0, disk_limit = 0;  // the host tier's and the disk tier's byte caps, 0 for a tier that is off
        uint64_t prompt_passes = 0;               // since start, the passes that carried a request's prompt rows
        uint64_t continued = 0;                   // since start, those formed while the same prompt's pass before them was in flight (Scheduler::continues)
        // Since start, the passes formed while replies and prompt work both waited (docs/SERVER.md, The service rule): of replies alone, of one prompt's chunk alone, and of both kinds or two prompts, which on a split the rule never forms.
        uint64_t rule_replies = 0, rule_chunks = 0, rule_mixed = 0;
        LiveRequests live{};
        ServiceRecord::Snapshot service{};        // the service record (docs/SERVER.md, The price table)
    };
    Stats stats() const {
        // The service record is copied under its own lock alone, before the scheduler's, so a health poll never holds the round.
        ServiceRecord::Snapshot service = record_.snapshot();
        std::lock_guard<std::mutex> lk(m_);
        Stats s{active_count_.load(), queue_.size(), 0, 0, 0, (size_t)pauses_, paused_count_.load(), stalls_, waits_, recomputed_, 0, checkpoints_.load(), slots_.size(), in_flight_.load(), samplers_.threads(), {}, {},
                timed_, timing_};
        cache_.fill(s);
        for (size_t p = 0; p < s.reserved.size(); ++p) {
            const size_t held = p < s.donor_blocks.size() ? s.donor_blocks[p] : 0;
            s.kv_tokens = std::max(s.kv_tokens, (s.reserved[p] - held) * pools_.block_tokens[p]);
            s.kv_donor_tokens = std::max(s.kv_donor_tokens, held * pools_.block_tokens[p]);
        }
        s.finished = finished_.load();
        s.prompt_passes = prompt_passes_.load();
        s.continued = continued_.load();
        s.rule_replies = rule_replies_.load();
        s.rule_chunks = rule_chunks_.load();
        s.rule_mixed = rule_mixed_.load();
        s.prompt_tokens = prompt_total_.load();
        s.generated_tokens = generated_total_.load();
        s.prefilled_tokens = prefilled_total_.load();
        s.published_tokens = published_total_.load();
        s.draft_failures = draft_failures_.load();
        s.holds = holds_.load();
        s.stage_waits = stage_waits_.load();
        s.yields = yields_.load();
        s.turns_without_memory = turns_without_memory_.load();
        s.host_limit = host_cap_;
        // Every source's by position, and the requests' own histories' apart.
        tally_.tally(infer::spec::Source::lookup, s.own_drafted, s.own_kept);
        tally_.tally(infer::spec::Source::proposer, s.drafted, s.kept);
        s.by_source.assign({tally_.proposer, tally_.lookup[0], tally_.lookup[1], tally_.lookup[2]});
        if (s.drafted.size() < s.own_drafted.size()) s.drafted.resize(s.own_drafted.size()), s.kept.resize(s.own_drafted.size());
        for (size_t i = 0; i < s.own_drafted.size(); ++i) s.drafted[i] += s.own_drafted[i], s.kept[i] += s.own_kept[i];
        s.live = live_;
        live_requested_ = true;
        s.service = std::move(service);
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
                           std::any_of(active.begin(), active.end(), [&](const std::shared_ptr<Request>& r) { return working(*r) && !cache_.waits_encode(*r) && (!r->seq_.in_flight() || continues(*r)); });
                };
                // Under --disk-cache-keep a server with nothing to do for kDiskIdle writes ahead (disk_round), and the loop wakes for it.
                cache_.idle_moment(!active.empty() || flying() || !queue_.empty() || !paused_.empty());
                const auto bound = soonest(cache_.next_disk_bound(lines()), cache_.encode_bound());
                // With nothing to do the host tier's slabs are taken ahead, a step a round with no sleep between, so a copy made later allocates none (grow_host).
                const bool ahead = !stopping_ && active.empty() && !flying() && queue_.empty() && paused_.empty() && cache_.grow_ahead();
                if (!ahead && bound) cv_.wait_until(lk, *bound, due);
                else if (!ahead) cv_.wait(lk, due);
                if (stopping_) break;
                seen_ = arrivals_;
                cache_.host_pressure();
                cache_.write_turn(write_turn(active));
                cache_.disk_round(active, lines());
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
                if (!active[i]->finish_pending_.empty() && !active[i]->seq_.in_flight()) finish(active, i, active[i]->finish_pending_, active[i]->error_pending_);
                else ++i;
            }
            cache_.complete_jobs(active, lines());
            // Cancelled requests leave before the next pass, and a request in flight once its pass has retired.
            for (size_t i = 0; i < active.size();) {
                if (active[i]->cancel_.load() && !active[i]->seq_.in_flight() && !active[i]->seq_.landing()) finish(active, i, "cancel");
                else ++i;
            }
            // Room is made and passes formed only in a free slot; a request in flight is never paused, parked or given room.
            if (free_slot() < slots_.size()) {
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
                    form_round(active, host);
                    // A request in flight outside a pass, its chain on the head stage's thread, comes back when the chain reports.
                    const auto left = std::find_if(active.begin(), active.end(), [](const std::shared_ptr<Request>& r) { return !r->job_ && !r->seq_.in_flight() && Cache::image_stop(*r) > r->seq_.length(); });
                    if (!flying() && chains_.empty() && !encoded_ && left != active.end()) throw std::logic_error("server: a round with active requests formed an empty pass: " + left_out(active, **left, requests(active)));
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
                const auto woken = [&] { return ended() || stopping_ || cache_.woken() || arrivals_ != seen_; };
                const auto bound = soonest(cache_.next_disk_bound(lines()), cache_.encode_bound());
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

    ~Scheduler() { model_.on_stage_wait({}); }

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
        bool continues = false;      // formed while its request's pass before it was in flight (Scheduler::continues)
        uint64_t job = 0;            // its hand-over to a recorder, which the recorder's result names
        Clock::time_point begun;
        PassWork work;                  // its work as the price table keys it
        std::vector<double> predicted;  // per stage, the stage time the prices predicted as it was formed, 0 where they did not price it
        size_t pieces = 0;              // an image's pieces queued on the first stage's thread ahead of it, which a replies' pass waits behind
        double pieces_host_ms = 0;      // this thread's time queuing them, the pieces' own time where the first stage records here
    };

    bool flying() const {
        return std::any_of(slots_.begin(), slots_.end(), [](const Slot& k) { return k.live; });
    }
    size_t flights_live() const {
        return (size_t)std::count_if(slots_.begin(), slots_.end(), [](const Slot& k) { return k.live; });
    }
    // A free slot for a pass; slots_.size() when none.
    size_t free_slot() const { return (size_t)(std::find_if(slots_.begin(), slots_.end(), [](const Slot& s) { return !s.live; }) - slots_.begin()); }
    // What a pass is formed of: every ready request's rows, the generating requests' alone, or one prompt's chunk alone.
    enum class Rows { all, generated, prompt, quick };   // quick: the replies and the plan's quick prompt in one pass
    std::vector<Flight> flights() const { return std::vector<Flight>(slots_.begin(), slots_.end()); }

    // The service rule's plan for a round (docs/SERVER.md, The service rule): whether replies and prompt work both wait, the replies' passes, a chunk's stage time and the prompts' order.
    // Prompt work is the clients' prompts, and with none waiting the jobs', which so take only what no prompt wants.
    struct Plan {
        bool overlap = false, priced = false, met = true, capped = false;
        std::shared_ptr<Request> quick;   // while the objective is unmet, the shortest prompt that rides the replies' pass (quick_prompt), whole once its cost is measured
        size_t quick_rows = 0;            // its rows: all it lacks, or the learning step while its price is not yet known
        double replies_ms = 0;
        size_t packed = 0, share = 0, limit = 0;
        double budget_ms = std::numeric_limits<double>::infinity(), gap_ms = 0;
        std::vector<std::shared_ptr<Request>> order;
    };
    Plan plan(const std::vector<std::shared_ptr<Request>>& active) {
        Plan p;
        size_t decoders = 0, deepest = 0;
        double fewest = std::numeric_limits<double>::infinity();
        std::vector<std::shared_ptr<Request>> prompts, jobs;
        for (const auto& r : active) {
            if (r->cancel_.load()) continue;
            if (r->job_) {
                if (working(*r)) jobs.push_back(r);
            } else if (!r->gen_.empty() && r->seq_.length() + 1 >= history_tokens(*r)) {
                // A reply, waiting for its decode entry or with it in flight.
                ++decoders;
                deepest = std::max(deepest, r->seq_.length());
                fewest = std::min(fewest, r->steps_.mean());
            } else if (r->seq_.length() < history_tokens(*r)) {
                prompts.push_back(r);
            }
        }
        if (prompts.empty()) prompts = jobs;
        p.overlap = decoders && !prompts.empty();
        const size_t S = model_.stage_count();
        p.limit = passes_in_flight(slots_.size(), S, p.overlap);
        p.packed = packed_passes(decoders, model_.decode_columns(), slots_.size());
        p.share = packed_share(decoders, p.packed);
        if (p.overlap) {
            // The replies' pass packed, the slowest stage's; before it is measured no budget binds.
            const size_t band = depth_band(deepest);
            double replies = 0;
            for (size_t s = 0; s < S; ++s) replies = std::max(replies, record_.decode_ms(s, band, p.share));
            if (replies > 0) {
                const ChunkPlan c = chunk_plan(fewest, replies, S);
                p.priced = true;
                p.replies_ms = replies;
                p.met = c.met;
                p.capped = c.capped;
                p.budget_ms = c.budget_ms;
                p.gap_ms = c.gap_ms;
            }
        }
        // Least remaining priced work first, any prompt without rows for kPromptWaitMs first of all, the oldest of equals (docs/SERVER.md, Prompt shares).
        const Clock::time_point now = Clock::now();
        std::vector<ReadyPrompt> views;
        // Each prompt's remaining rows at the slowest stage's price for the quick rule, 0 where a stage cannot price it yet, and the step that would learn it.
        std::vector<double> cost;
        std::vector<size_t> steps;
        for (const auto& r : prompts) {
            const size_t at = r->seq_.length(), left = history_tokens(*r) - std::min(history_tokens(*r), at);
            const size_t cls = class_id(class_at(r->classes_, at).extent), band = depth_band(at);
            const double row = record_.row_ms(0, cls, band, ubatch_);
            double slowest = 0;
            bool measured = true;
            size_t step = SIZE_MAX;
            for (size_t s = 0; s < S; ++s) {
                size_t learn = kPieceStep;
                const double ms = record_.slice_ms(s, cls, band, std::max<size_t>(left, 1), learn);
                measured = measured && ms > 0;
                slowest = std::max(slowest, ms);
                if (ms <= 0) step = std::min(step, learn);
            }
            cost.push_back(measured ? slowest : 0.0);
            steps.push_back(measured ? left : std::min(left, step == SIZE_MAX ? kPieceStep : step));
            const Clock::time_point since = std::max(r->rows_at_, r->admitted_);
            views.push_back({(double)left * (row > 0 ? row : 1.0), r->seq_.in_flight() ? 0.0 : std::chrono::duration<double, std::milli>(now - since).count(), 0.0, left});
        }
        Share share{Share::Rule::least_remaining};
        share.wait_ms = kPromptWaitMs;
        while (!views.empty()) {
            const size_t i = next_prompt(share, views, {});
            // With the objective unmet a prompt takes a chunk only once the wait bound makes it due, but the shortest quick prompt rides the replies' pass: whole, or the learning step while its price is not yet known.
            // A prompt longer than a pass cannot run whole, so it is never quick.
            const bool quick = !p.met && !prompts[i]->seq_.in_flight() && views[i].remaining_rows <= ubatch_ && (cost[i] == 0 || quick_prompt(cost[i], p.replies_ms)) &&
                               (!p.quick || views[i].remaining_rows < history_tokens(*p.quick) - std::min(history_tokens(*p.quick), p.quick->seq_.length()));
            if (quick) {
                p.quick = prompts[i];
                p.quick_rows = steps[i];
            }
            if (p.met || views[i].idle_ms >= kPromptWaitMs) {
                p.order.push_back(prompts[i]);
            }
            prompts.erase(prompts.begin() + (std::ptrdiff_t)i);
            views.erase(views.begin() + (std::ptrdiff_t)i);
            cost.erase(cost.begin() + (std::ptrdiff_t)i);
            steps.erase(steps.begin() + (std::ptrdiff_t)i);
        }
        if (p.quick) p.order.erase(std::remove(p.order.begin(), p.order.end(), p.quick), p.order.end());
        return p;
    }
    // A row class's index for the price table, numbered as first seen.
    size_t class_id(size_t extent) {
        const size_t id = class_ids_.emplace(model_.row_class(extent), class_ids_.size()).first->second;
        record_.class_extent(id, extent);
        return id;
    }
    // The rows of r's next chunk within `budget_ms` of every stage's time, by each stage's prices (PriceTable::chunk_rows); a ubatch where no budget binds.
    size_t chunk_rows(const Request& r, double budget_ms) {
        if (!(budget_ms < std::numeric_limits<double>::infinity())) return ubatch_;
        const size_t at = r.seq_.length(), cls = class_id(class_at(r.classes_, at).extent), band = depth_band(at);
        size_t n = ubatch_;
        for (size_t s = 0; s < model_.stage_count(); ++s) n = std::min(n, record_.chunk_rows(s, cls, band, budget_ms, ubatch_));
        return n;
    }
    // The round's new passes by the service rule, one in each free slot while a request is ready.
    // While both wait, on a split the replies go in the plan's passes and each other slot takes one prompt's chunk; on one pass in flight the chunk rides the replies' pass.
    void form_round(std::vector<std::shared_ptr<Request>>& active, std::vector<std::pair<size_t, size_t>>& host) {
        const Plan p = plan(active);
        size_t replying = 0;
        for (const Slot& s : slots_) replying += s.live && s.decoders;
        for (size_t k = free_slot(); k < slots_.size() && flights_live() < p.limit && !first_stage_held(); k = free_slot()) {
            Rows rows = Rows::all;
            if (p.overlap && slots_.size() > 1) rows = replying < p.packed && ready_decoders(active) ? Rows::generated : Rows::prompt;
            // Cold, on one pass in flight the replies go alone first, so their pass is measured before a chunk rides it.
            if (p.overlap && !p.priced && slots_.size() == 1 && ready_decoders(active)) rows = Rows::generated;
            // A replies' pass takes the quick prompt every other time.
            const auto riding = [&](Rows r) { return (r == Rows::generated || (r == Rows::all && p.overlap)) && p.quick && quick_turn_ && ready_decoders(active) ? Rows::quick : r; };
            rows = riding(rows);
            bool formed = form(k, active, host, rows, p);
            // A slot no prompt can use goes to the replies.
            if (!formed && rows == Rows::prompt) formed = form(k, active, host, rows = riding(Rows::generated), p);
            if (!formed) return;
            if (slots_[k].decoders && p.quick) quick_turn_ = rows != Rows::quick;
            replying += slots_[k].decoders > 0;
            if (!p.overlap) continue;
            if (p.priced) record_.objective(p.gap_ms, p.met, p.capped);
            size_t prompts = 0;
            for (size_t e = slots_[k].decoders; e < slots_[k].members.size(); ++e) ++prompts;
            rule_replies_ += slots_[k].decoders && !prompts;
            rule_chunks_ += prompts && !slots_[k].decoders;
            rule_mixed_ += (slots_[k].decoders && prompts) || prompts > 1;
        }
    }
    // Whether a decoding request is ready for a pass.
    static bool ready_decoders(const std::vector<std::shared_ptr<Request>>& active) {
        return std::any_of(active.begin(), active.end(), [](const std::shared_ptr<Request>& r) { return !r->job_ && decoding(*r) && !r->seq_.in_flight() && !r->stalled_; });
    }

    // A new pass in slot k: decode entries first, then prompt rows in the plan's order, up to ubatch tokens; while replies and prompts both wait, the replies packed by the decode columns and one prompt's chunk by the gap's budget.
    // Then its logits rows, begin_pass and its first stage, or on the host that stage's place in `host`; false when no request has rows to add, which with nothing in flight throws rather than leaving the loop to spin.
    bool form(size_t k, std::vector<std::shared_ptr<Request>>& active, std::vector<std::pair<size_t, size_t>>& host, Rows rows, const Plan& plan) {
        Slot& f = slots_[k];
        entries_.clear();
        f.members.clear();
        f.wanting.clear();
        f.from.clear();
        f.rows.clear();
        f.first_row.clear();
        f.decoders = 0;
        f.want = 0;
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
            if (rows != Rows::prompt && !r->job_ && ready(*r) && decoding(*r)) waiting.push_back(r.get());
        decoders += waiting.size();
        const size_t share = std::min(plan.overlap ? plan.share : decode_share(decoders, plan.limit, model_.stage_count()), waiting.size());
        std::stable_sort(waiting.begin(), waiting.end(), [](const Request* a, const Request* b) { return a->landed_ < b->landed_; });
        waiting.resize(share);
        // A decoding request with drafts takes a verify entry, its last pick then the drafts, every row's logits, extent 1, in the decode columns the pass's decoders leave.
        std::vector<std::shared_ptr<Request>> decoding_now;
        for (auto& r : active)
            if (std::find(waiting.begin(), waiting.end(), r.get()) != waiting.end()) decoding_now.push_back(r);
        propose(decoding_now, draft_columns(share, active.size()));
        for (auto& r : decoding_now) {
            // A request whose chain went to the head stage's thread sits this pass out and verifies its drafts once they land.
            if (r->seq_.in_flight()) continue;
            if (r->verify_.size() > 1) {
                infer::BatchEntry e{&r->seq_, r->verify_.data(), r->verify_.size(), true, true};
                e.extent = 1;
                add_entry(r, e);
            } else {
                add_entry(r, infer::BatchEntry{&r->seq_, &r->last_id_, 1, true});
            }
            ++f.decoders;
        }
        size_t budget = rows == Rows::generated ? 0 : rows == Rows::quick ? plan.quick_rows : ubatch_, keeps = 0;
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
                r->boundary_kept_ = e.keep ? at + n : 0;
            }
            // The slice that ends a proof's path keeps its state there, which the proof compares.
            if (r->prove_state_ && r->prove_at_ == at + n && !e.keep) {
                e.keep = cache_.checkpoint_room(keeps, r->source_);
                keeps += e.keep;
            }
            add_entry(r, e);
        };
        // Prompt rows in the plan's order: while replies and prompts both wait one prompt's chunk, of the rows the gap's budget gives, and otherwise a slice of each in turn up to the ubatch.
        // A prompt in flight whose slice has committed the first stage continues in a pass of its own (continues); one at an image without rows spends its turn on a piece of that image's encode (docs/VISION.md, section 10).
        bool company;
        {
            std::lock_guard<std::mutex> lk(m_);
            company = !queue_.empty() || !paused_.empty();
        }
        const bool alone = active.size() == 1 && !active[0]->job_ && !company;
        const std::vector<std::shared_ptr<Request>> only_quick = rows == Rows::quick ? std::vector<std::shared_ptr<Request>>{plan.quick} : std::vector<std::shared_ptr<Request>>{};
        for (const auto& r : rows == Rows::quick ? only_quick : plan.order) {
            if (!budget) break;
            const bool continued = r->seq_.in_flight();
            if (continued && !(entries_.empty() && continues(*r))) continue;
            if (!continued && r->stalled_) continue;
            if (Cache::image_stop(*r) <= r->seq_.length()) {
                // Beside replies an encode takes the pieces a replies' pass meets as a prompt takes a chunk, by the chunk budget at a piece's measured time (encode_pieces).
                const size_t allowed = plan.overlap ? encode_pieces(plan.budget_ms, piece_ms_) : 0;
                if (encoded_ || continued || (plan.overlap && pieces_since_replies_ >= allowed)) continue;
                // Every piece while nothing else could use the stages; a health poll ends the run of pieces too, so the round boundary it waits for comes.
                size_t queued = 1;
                const auto more = [&] {
                    if (plan.overlap) return pieces_since_replies_ + queued < allowed && ++queued;
                    if (!alone || r->job_ || flying() || !entries_.empty() || r->cancel_.load()) return false;
                    std::lock_guard<std::mutex> lk(m_);
                    return arrivals_ == seen_ && !stopping_ && !live_requested_;
                };
                const Clock::time_point piece_start = Clock::now();
                if (cache_.encode_piece(*r, more)) {
                    pieces_host_ms_ += ms_since(piece_start);
                    pieces_since_replies_ += queued;
                    acted_ = encoded_ = true;
                    budget = 0;
                    break;
                }
                continue;
            }
            const size_t before = entries_.size();
            // Until the replies' pass is priced a chunk is one small step.
            // A lone prompt on a split is cut so every stage reads; otherwise a ubatch, and beside replies the service rule's chunk.
            const size_t most = alone ? lone_slice(history_tokens(*r) - r->seq_.length(), model_.stage_count(), ubatch_) : ubatch_;
            // The quick prompt rides whole, or as the learning step while its price is not yet known.
            const size_t chunk = rows == Rows::quick ? plan.quick_rows : plan.priced ? chunk_rows(*r, plan.budget_ms) : kPieceStep;
            slice(r, std::min(budget, plan.overlap ? chunk : most));
            if (entries_.size() == before) continue;
            r->rows_at_ = Clock::now();
            if (continued || plan.overlap) break;
        }
        if (entries_.empty()) return false;
        const bool continuing = f.members.front()->seq_.in_flight();
        f.base = take_rows(logit_rows_, f.want);
        if (f.base == logit_rows_.size) throw std::logic_error("server: no logits rows for a new pass");
        try {
            model_.begin_pass(ctx_, k, entries_.data(), entries_.size(), f.base);
        } catch (...) {
            give_rows(logit_rows_, f.base, f.want);
            throw;
        }
        price(f);
        prompt_passes_ += f.members.size() > f.decoders;
        f.pieces = 0;
        f.pieces_host_ms = 0;
        if (f.decoders) {
            f.pieces = pieces_since_replies_;
            f.pieces_host_ms = pieces_host_ms_;
            pieces_since_replies_ = 0;
            pieces_host_ms_ = 0;
        }
        continued_ += continuing;
        f.continues = continuing;
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

    // A pass's work as the price table keys it, each stage's predicted time and its rows by kind, for the service record (docs/SERVER.md, The price table); it changes nothing a pass holds.
    void price(Slot& f) {
        f.work = PassWork{};
        RowKinds kinds;
        for (size_t e = 0; e < entries_.size(); ++e) {
            const infer::BatchEntry& x = entries_[e];
            const Request& m = *f.members[e];
            const size_t extent = x.extent ? x.extent : x.n;
            if (extent == 1) {
                f.work.decode(x.n, f.from[e]);
            } else {
                const size_t id = class_ids_.emplace(model_.row_class(extent), class_ids_.size()).first->second;
                record_.class_extent(id, extent);
                f.work.chunk(id, x.n, f.from[e]);
                const size_t b = depth_band(f.from[e]);
                if (kinds.by_band.size() <= b) kinds.by_band.resize(b + 1, 0);
                kinds.by_band[b] += x.n;
            }
            if (m.job_) {
                kinds.reread += x.n;
            } else if (extent != 1) {
                kinds.prompt += x.n;
            } else if (f.from[e] < m.reached_) {
                kinds.recomputed += x.n;
            } else {
                kinds.decode += 1;
                kinds.drafts += x.n - 1;
            }
        }
        f.predicted.assign(model_.stage_count(), 0.0);
        for (size_t s = 0; s < f.predicted.size(); ++s) f.predicted[s] = record_.predict(s, f.work);
        record_.formed(kinds);
    }

    // Stage s of the pass in slot k; a failure has abandoned the pass in the model, and it fails that pass alone.
    void advance(std::vector<std::shared_ptr<Request>>& active, size_t k, size_t s) {
        // A stage with a thread records there what would overlap another pass: a stage that reads its rows always, a tensor group's always, a single device's while another pass is in flight.
        // A stage whose thread still holds items records there too, after them.
        if (Recorder* r = plain_[s] && (model_.stage_reads(s) || recorded_apart(model_.stage_grouped(s), flights_live()) || busy_items(s)) ? recorders_[s].get() : nullptr) {
            // A stage records one pass at a time: a pass that finds its recorder taken waits for a later round, which gives the stage its oldest waiting pass.
            if (recording(s)) return;
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
                Recorder::Post& post = r->pass;
                post.slot = k;
                post.job = slots_[k].job = ++jobs_posted_;
                post.pass = slots_[k].formed;
                post.work = slots_[k].work;
                post.predicted_ms = slots_[k].predicted[s];
                post.posted = true;
            }
            r->cv.notify_all();
            return;
        }
        acted_ = true;
        const Clock::time_point start = timed_ ? Clock::now() : Clock::time_point{};
        const Clock::time_point begun = Clock::now();
        const backend::Backend::HostTimes before = timed_ ? host_times() : backend::Backend::HostTimes{};
        try {
            // Undoing the pass drains every device, which a stage being recorded on its thread must not see, so the recorders go quiet first.
            model_.run_pass_stage(ctx_, k, s, [this] { quiet(); });
        } catch (const std::exception& e) {
            fail_pass(active, k, e.what());
            return;
        }
        const auto sent = model_.stage_sent(ctx_, k);
        const auto waits = model_.stage_host_waits(ctx_, k);
        record_.post({slots_[k].formed, s, slots_[k].work, slots_[k].predicted[s], begun, Clock::now(), waits.first, waits.second, sent.first, sent.second});
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
    // A reading is a call on the stage's devices, so it is taken on the stage's thread between its recordings (run_on).
    void end_round(Clock::time_point start) {
        static constexpr uint64_t kReadRounds = 32;
        round_.round_ms += ms_since(start);
        ++round_.rounds;
        const size_t S = model_.stage_count();
        if (round_.stage_ms.size() != S) {
            round_.stage_ms.assign(S, 0.0);
            std::fill(host_stage_ms_.begin(), host_stage_ms_.end(), 0.0);
            for (size_t s = 0; s < S; ++s) run_on(s, [&] { model_.stage_backend(s).device_ms(); });
            span_start_ = Clock::now();
            span_rows_ = 0;
        } else if (round_.rounds % kReadRounds == 0) {
            round_.span_ms += ms_since(span_start_);
            for (size_t s = 0; s < S; ++s) {
                double ms = 0;
                run_on(s, [&] { ms = model_.stage_backend(s).device_ms(); });
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
        // A pass's cost is the time the server gave it since the last pass retired, less the chains drafted meanwhile, which the price counts apart.
        Retirement& last = retirement_;
        const double since_retired = last.any ? ms_since(last.at) - last.chain_ms : std::numeric_limits<double>::infinity();
        const double took = std::min(ms_since(f.begun), since_retired);
        // A replies' pass that waited behind an image's pieces took their time beside its own, which the prices predicted; pieces run on this thread, before the pass began, took the time they held it.
        if (f.pieces) {
            double own = 0;
            for (double ms : f.predicted) own = std::max(own, ms);
            const double ms = std::max(own > 0 && took > own ? took - own : 0.0, f.pieces_host_ms);
            if (ms > 0) piece_ms_ = piece_ms_ > 0 ? 0.75 * piece_ms_ + 0.25 * ms / (double)f.pieces : ms / (double)f.pieces;
        }
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
            if (e >= f.decoders) r.rows_at_ = Clock::now();
            // Only the pass whose rows reach the boundary holds its state: an earlier slice of a continued prompt retires first.
            if (r.boundary_kept_ && f.from[e] + f.rows[e] == r.boundary_kept_) {
                // A state computed over rows restored from unproved files is kept by no one: prove keeps the one at its path's end once the proof has ended equal.
                if (!r.unproved_ || to != r.prove_at_) r.boundary_kept_ = 0;
                std::lock_guard<std::mutex> lk(m_);
                if (!r.cancel_.load() && !r.unproved_) cache_.asked_boundary(r);
            }
        }
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
        give_rows(logit_rows_, f.base, f.want);
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
    // Each drafts within the `columns`, its reservation and tokens left (spec::draft_length) and the price's gain (spec::draft_depths): from its own history where that convinces (spec::lookup_first), else from the proposer.
    void propose(const std::vector<std::shared_ptr<Request>>& rs, size_t columns) {
        using infer::spec::Source;
        for (const auto& r : rs) r->verify_.clear();
        if (!proposer_) return;
        keeps_.clear();
        askers_.clear();
        chained_.clear();
        // A request whose chain landed verifies those drafts, within the columns and a free mark, and asks for none; drafts of another history are dropped.
        landed_.clear();
        for (const auto& r : rs) {
            if (r->chained_at_ == SIZE_MAX) continue;
            landed_.push_back(r.get());
            std::vector<uint32_t> d = std::move(r->chained_);
            const bool here = r->chained_at_ == r->seq_.length();
            r->chained_.clear();
            r->chained_at_ = SIZE_MAX;
            const auto past = std::find_if(d.begin(), d.end(), [&](uint32_t id) { return id >= model_.n_vocab(); });
            const size_t k = here ? std::min({columns, (size_t)(past - d.begin()), d.size()}) : 0;
            if (!k || !model_.mark(r->seq_)) continue;
            columns -= k;
            r->verify_.assign(1, r->last_id_);
            r->verify_.insert(r->verify_.end(), d.begin(), d.begin() + (std::ptrdiff_t)k);
        }
        const bool first = proposer_->after_lookup();
        for (const auto& r : rs) {
            // A request resumed by a fork of its whole history decodes before any pass has fed its sequence, and drafts from the pass after (Model::can_draft).
            if (r->job_ || !model_.can_draft(r->seq_)) continue;
            if (std::find(landed_.begin(), landed_.end(), r.get()) != landed_.end()) continue;
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
            keeps_.emplace_back(k);
            // A draft is priced by its source's acceptance, a history draft's by its match bucket's, each weighed with the server's of the same.
            for (size_t j = 0; j < k; ++j) keeps_.back()[j] = record.drafting().keeps(j, tally_.at(record.drafted, record.bucket));
            chained_.push_back(record.drafted == Source::proposer);
            askers_.push_back(r.get());
        }
        // A chain runs on the head's devices before the passes posted there after it, so every pass in flight waits on it as the one being formed does.
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
        // The columns and marks go to the drafts the price kept, those most likely kept first, so a request admitted earlier with a poor draft does not take them from a better one.
        order_.resize(askers_.size());
        std::iota(order_.begin(), order_.end(), size_t(0));
        const auto kept_share = [&](size_t i) {
            double kept = 0;
            for (size_t j = 0; j < takes_[i]; ++j) kept += keeps_[i][j];
            return takes_[i] ? kept / (double)takes_[i] : 0.0;
        };
        std::stable_sort(order_.begin(), order_.end(), [&](size_t a, size_t b) { return kept_share(a) > kept_share(b); });
        size_t free_marks = marks;
        for (const size_t i : order_) {
            if (!takes_[i]) continue;
            takes_[i] = free_marks ? std::min(takes_[i], columns) : 0;
            columns -= takes_[i];
            free_marks -= takes_[i] ? 1 : 0;
        }
        // The requests that draft: `asked_[i]` is request i's place among the proposer's asks, or none for a draft of its own history.
        size_t n = 0, chains = 0, steps = 0;
        asked_.assign(askers_.size(), SIZE_MAX);
        for (size_t i = 0; i < askers_.size(); ++i) {
            if (!takes_[i]) continue;
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
        // A chain on the head's devices goes to the head stage's thread where it has one, its requests sitting the pass out; elsewhere it runs here, holding every stage for a proposer that runs on others' devices too.
        bool failed = false, queued = false;
        if (chains) {
            const Clock::time_point start = Clock::now();
            try {
                if (proposer_->on_head() && recorders_.back()) {
                    queued = queue_chain(chains, steps);
                } else if (proposer_->on_device() && !proposer_->on_head()) {
                    const Held still(*this);
                    proposer_->draft_all(asks_.data(), chains);
                } else {
                    proposer_->draft_all(asks_.data(), chains);
                }
            } catch (const std::exception& e) {
                std::fprintf(stderr, "server: a draft failed and its pass runs without the proposer's drafts (%s)\n", e.what());
                ++draft_failures_;
                failed = true;
            }
            const double chain = ms_since(start);
            if (!failed && !queued) times_.chain(steps, chains, chain);
            round_.drafting_ms += chain;
            formed_chains_ms_ += chain;
            retirement_.chain_ms += chain;
        }
        for (size_t i = 0; i < askers_.size(); ++i) {
            if (!takes_[i] || (chained_[i] && (failed || queued))) continue;
            Request& r = *askers_[i];
            const std::vector<uint32_t>& d = chained_[i] ? asks_[asked_[i]].out : r.own_draft_;
            // A draft past the vocabulary, and those after it, are not fed: a verify needs nothing a decode does not.
            const auto past = std::find_if(d.begin(), d.end(), [&](uint32_t id) { return id >= model_.n_vocab(); });
            const size_t k = std::min(takes_[i], (size_t)(past - d.begin()));
            // A mark is the model's bookkeeping and waits on no device but the history's own tickets, so it needs no hold of the recorders.
            if (!k || !model_.mark(r.seq_)) continue;
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
                // The rerun goes to the state stages' threads behind the verify; the lengths change here, and the mark goes back once the rerun has run (release_settled).
                model_.retract(r.seq_, keep, [this, &r](infer::Model::StageItem it) { defer_item(&r, std::move(it)); });
                model_.release_settled(r.seq_);
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

    // A stage's thread: it records the pass the round hands it (advance) and decides nothing, and the round takes what it did (collect); a failure is the scheduler's thread's to undo.
    struct Recorder {
        Recorder(std::mutex& mutex, infer::StageLanes::Lane& lane) : m(mutex), cv(lane.cv) {}
        struct Post {
            bool posted = false;
            size_t slot = 0;
            uint64_t job = 0;        // the hand-over, which the recording's end names
            uint64_t pass = 0;       // the pass's formation order, work and predicted stage time, for the service record
            PassWork work;
            double predicted_ms = 0;
        };
        struct End {
            size_t slot = 0;
            uint64_t job = 0;
            bool failed = false;
            std::string error;
        };
        // Device work of a request outside a pass (infer::Model::StageItem) or a drafter's chain, queued on the scheduler's thread in formation order.
        // An item with `then` reports to the round whether or not it threw, and the round calls `then` with its error; one without reports only a failure, which ends its request.
        struct Item {
            const Request* r = nullptr;
            std::function<void()> run;
            std::function<void(const std::string&)> then;
        };
        struct ItemEnd {
            const Request* r = nullptr;
            std::string error;
            std::function<void(const std::string&)> then;
        };
        std::mutex& m;                 // its lane's
        std::condition_variable& cv;   // its lane's wake (infer::StageLanes)
        Post pass;                     // under m
        std::vector<End> ends;         // under m, the recordings ended since the round last collected
        std::deque<Item> items;        // under m, waiting
        bool item_running = false;     // under m
        std::vector<ItemEnd> reports;  // under m, items that threw or report since the round last collected
    };

    // One item on its stage's thread; a failure is its request's alone, which the round ends, and a chain's end is the round's.
    void run_item(Recorder& rec, std::unique_lock<std::mutex>& lk) {
        Recorder::Item it = std::move(rec.items.front());
        rec.items.pop_front();
        rec.item_running = true;
        lk.unlock();
        std::string error;
        try {
            it.run();
        } catch (const std::exception& e) {
            error = e.what();
        }
        lk.lock();
        rec.item_running = false;
        const bool report = !error.empty() || it.then;
        if (report) rec.reports.push_back({it.r, error, std::move(it.then)});
        rec.cv.notify_all();
        lk.unlock();
        if (report) wake();
        lk.lock();
    }

    // A drafter's chain on the head stage's thread (infer::Model::Chain): begun here, recorded there behind the work queued before it, and ended by the round once it has run.
    struct Chaining {
        std::vector<infer::spec::Proposer::Ask> asks;
        std::vector<Request*> rs;             // asks[i]'s request
        std::unique_ptr<infer::Model::Chain> chain;
        size_t steps = 0;
        double record_ms = 0;                 // the head stage's thread's, read once it has reported
        bool ended = false;
    };

    // The first `chains` of asks_ as one chain on the head stage's thread, each request in flight until it ends; false where none drafts.
    bool queue_chain(size_t chains, size_t steps) {
        auto c = std::make_shared<Chaining>();
        c->rs.assign(chains, nullptr);
        for (size_t i = 0; i < askers_.size(); ++i)
            if (asked_[i] != SIZE_MAX) c->rs[asked_[i]] = askers_[i];
        c->asks.resize(chains);
        for (size_t i = 0; i < chains; ++i) std::swap(c->asks[i], asks_[i]);
        c->steps = steps;
        c->chain = proposer_->begin_all(c->asks.data(), chains);
        if (!c->chain) return false;
        chains_.push_back(c);
        Recorder& rec = *recorders_.back();
        infer::Model::Chain* chain = c->chain.get();
        double* ms = &c->record_ms;
        {
            std::lock_guard<std::mutex> lk(rec.m);
            rec.items.push_back({nullptr, [this, chain, ms] {
                                     const Clock::time_point start = Clock::now();
                                     model_.draft_record(*chain);
                                     *ms = ms_since(start);
                                 },
                                 [this, c](const std::string& error) { end_chain(*c, error); }});
        }
        rec.cv.notify_all();
        return true;
    }

    // A chain's end: its requests out of flight, each with the drafts its next pass verifies, or none where the chain failed.
    void end_chain(Chaining& c, const std::string& error) {
        if (c.ended) return;
        c.ended = true;
        const auto at = std::find_if(chains_.begin(), chains_.end(), [&](const std::shared_ptr<Chaining>& x) { return x.get() == &c; });
        const std::shared_ptr<Chaining> held = *at;
        chains_.erase(at);
        const Clock::time_point start = Clock::now();
        model_.draft_end(*c.chain);
        const bool drafted = c.chain->recorded;
        if (drafted) times_.chain(c.steps, c.rs.size(), c.record_ms);
        else {
            std::fprintf(stderr, "server: a draft failed and its requests decode without the proposer's drafts (%s)\n", error.empty() ? "abandoned" : error.c_str());
            ++draft_failures_;
        }
        for (size_t i = 0; i < c.rs.size(); ++i) {
            Request& r = *c.rs[i];
            r.chained_at_ = r.seq_.length();
            r.chained_ = drafted ? std::move(c.asks[i].out) : std::vector<uint32_t>{};
        }
        const double ms = ms_since(start);
        round_.drafting_ms += ms;
        retirement_.chain_ms += ms;
    }

    // Request r's work outside a pass on a stage: queued to the stage's thread, or run here where the stage has none.
    void defer_item(const Request* r, infer::Model::StageItem it) {
        Recorder* rec = it.stage < recorders_.size() ? recorders_[it.stage].get() : nullptr;
        if (!rec) {
            it.run();
            return;
        }
        // A restore's part that fails is the history gone, which the round meets in every holder of it (history_failed); every part of a copy reports, so the round wakes for what waited on it (Cache::promote).
        std::function<void(const std::string&)> then;
        if (it.restore) then = [this, tag = it.restore](const std::string& error) {
            if (!error.empty()) failed_restores_.push_back({tag, error});
        };
        else if (!r) then = [](const std::string&) {};
        {
            std::lock_guard<std::mutex> lk(rec->m);
            rec->items.push_back({r, std::move(it.run), std::move(then)});
        }
        rec->cv.notify_all();
    }

    // Work on stage s's devices that this thread waits for: an item on the stage's thread, behind what it was given before, or run here where the stage has none.
    // The other stages' threads go on recording; this thread waits as it would on the stage's ticket, its own state untouched meanwhile.
    void run_on(size_t s, const std::function<void()>& work) {
        Recorder* rec = s < recorders_.size() ? recorders_[s].get() : nullptr;
        if (!rec) {
            work();
            return;
        }
        auto done = std::make_shared<std::promise<void>>();
        std::future<void> ran = done->get_future();
        {
            std::lock_guard<std::mutex> lk(rec->m);
            rec->items.push_back({nullptr, [work, done] {
                try {
                    work();
                    done->set_value();
                } catch (...) {
                    done->set_exception(std::current_exception());
                }
            }});
        }
        rec->cv.notify_all();
        // The stage's thread may end a recording meanwhile, which tells the round without this thread's lock only while it waits so (wake).
        ++quieting_;
        ran.wait();
        --quieting_;
        ran.get();
    }

    // The earlier of two times the round wakes at, either possibly none.
    static std::optional<Clock::time_point> soonest(std::optional<Clock::time_point> a, std::optional<Clock::time_point> b) {
        if (!a) return b;
        if (!b) return a;
        return std::min(*a, *b);
    }

    // Whether stage s's thread holds an item, waiting or running.
    bool busy_items(size_t s) {
        Recorder* rec = s < recorders_.size() ? recorders_[s].get() : nullptr;
        if (!rec) return false;
        std::lock_guard<std::mutex> lk(rec->m);
        return !rec->items.empty() || rec->item_running;
    }

    // One recording of the pass posted to stage s, on its lane's thread with the stage's lock held (infer::StageLanes).
    void record(Recorder& r, size_t s, std::unique_lock<std::mutex>& lk) {
        const Recorder::Post taken = r.pass;
        lk.unlock();
        Recorder::End end{taken.slot, taken.job, false, {}};
        const Clock::time_point begun = Clock::now();
        try {
            model_.record_pass_stage(ctx_, taken.slot, s);
            const auto sent = model_.stage_sent(ctx_, taken.slot);
            const auto waits = model_.stage_host_waits(ctx_, taken.slot);
            record_.post({taken.pass, s, taken.work, taken.predicted_ms, begun, Clock::now(), waits.first, waits.second, sent.first, sent.second});
        } catch (const std::exception& e) {
            end.failed = true;
            end.error = e.what();
        }
        lk.lock();
        r.pass.posted = false;
        r.ends.push_back(std::move(end));
        r.cv.notify_all();
        lk.unlock();
        wake();
        lk.lock();
    }

    // Whether a request of a client is generating, its prompt read.
    static bool replying(const std::vector<std::shared_ptr<Request>>& active) {
        return std::any_of(active.begin(), active.end(), [](const std::shared_ptr<Request>& r) { return !r->job_ && !reads_prompt(r->gen_.size(), r->seq_.length(), history_tokens(*r)); });
    }
    // How a turn-end write beyond the host tier may go this round (Cache::flush_one): whole while nothing runs or waits, a piece at a time beside prompts alone, and beside replies none but an awaited write.
    // A write starved so is lost when its donor goes, which the health counts (docs/SERVER.md, The service rule).
    Cache::WriteTurn write_turn(const std::vector<std::shared_ptr<Request>>& active) const {
        if (!requests(active) && queue_.empty() && paused_.empty()) return Cache::WriteTurn::whole;
        return replying(active) ? Cache::WriteTurn::none : Cache::WriteTurn::piece;
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
            if (!rec->ends.empty() || !rec->reports.empty()) return true;
        }
        return false;
    }

    // Whether a pass is being recorded at stage s.
    bool recording(size_t s) const {
        return std::any_of(slots_.begin(), slots_.end(), [&](const Slot& k) { return k.live && k.recording && k.ran == s; });
    }
    // Whether a pass waits for the first stage's recorder or is being recorded there, so that no other is formed behind it.
    bool first_stage_held() const {
        return recorders_[0] && std::any_of(slots_.begin(), slots_.end(), [&](const Slot& k) { return k.live && k.ran == 0; });
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
            std::vector<Recorder::ItemEnd> reports;
            {
                std::lock_guard<std::mutex> lk(rec->m);
                ends.swap(rec->ends);
                reports.swap(rec->reports);
            }
            for (const Recorder::ItemEnd& f : reports) {
                acted_ = true;
                if (f.then) f.then(f.error);
                else item_failed(active, f.r, f.error);
            }
            for (const auto& [tag, error] : std::vector<std::pair<std::shared_ptr<const void>, std::string>>(std::move(failed_restores_))) history_failed(active, tag.get(), error);
            failed_restores_.clear();
            for (const Recorder::End& e : ends) {
                // The end of a recording whose pass was abandoned meanwhile (abort_all) is nobody's.
                if (!slots_[e.slot].live || !slots_[e.slot].recording || slots_[e.slot].job != e.job) continue;
                acted_ = true;
                slots_[e.slot].recording = false;
                if (!e.failed) {
                    model_.commit_pass_stage(ctx_, e.slot);
                    ++slots_[e.slot].ran;
                    continue;
                }
                const std::vector<size_t> taken = abandon(e.slot);
                fail_pass(active, e.slot, e.error, taken);
            }
        }
        // A retract's mark goes back once its rerun has run on the stages' threads, so another reply's verify can take it this round.
        for (const auto& r : active) model_.release_settled(r->seq_);
    }

    // Every recorder idle: before passes are abandoned, which drains every device and returns histories.
    void quiet() {
        hold();
        release();
    }
    // Every recorder idle until release(): the scheduler's thread then works on the stages' devices between two of their recordings.
    void hold() {
        ++holds_;
        ++quieting_;
        for (auto& rec : recorders_) {
            if (!rec) continue;
            std::unique_lock<std::mutex> lk(rec->m);
            rec->cv.wait(lk, [&] { return !rec->pass.posted && rec->items.empty() && !rec->item_running; });
        }
    }

    // A request's item failed on its stage's thread: it ends with the error alone.
    // Its passes in flight, all formed after the item, are abandoned in formation order, which puts every member's history back; the other members leave flight unchanged and are formed again, a verify's mark returned first.
    void item_failed(std::vector<std::shared_ptr<Request>>& active, const Request* r, const std::string& what) {
        const auto it = std::find_if(active.begin(), active.end(), [&](const std::shared_ptr<Request>& a) { return a.get() == r; });
        if (it == active.end()) return;
        leave_flight(r);
        // A request whose chain is still on the head stage's thread ends once the chain has.
        if (r->seq_.in_flight()) {
            (*it)->finish_pending_ = "error";
            (*it)->error_pending_ = what;
            return;
        }
        finish(active, (size_t)(it - active.begin()), "error", what);
    }

    // A restore whose copy failed on a stage's thread: the history is gone, so each request holding it leaves flight and reads it again from the start, and a donor holding it is dropped.
    void history_failed(std::vector<std::shared_ptr<Request>>& active, const void* tag, const std::string& what) {
        for (const auto& r : active) {
            if (!r->seq_.restoring(tag)) continue;
            leave_flight(r.get());
            std::fprintf(stderr, "server: a restore of %zu tokens failed (%s); its request reads them again\n", r->seq_.length(), what.c_str());
            r->verify_.clear();
            model_.reset(r->seq_);
        }
        cache_.drop_restoring(tag);
    }

    // Every pass in flight holding r abandoned in formation order, with the passes after each, which puts every member's history back; the other members leave flight unchanged and are formed again, a verify's mark returned first.
    void leave_flight(const Request* r) {
        std::vector<size_t> held;
        for (size_t k = 0; k < slots_.size(); ++k)
            if (slots_[k].live && std::any_of(slots_[k].members.begin(), slots_[k].members.end(), [&](const std::shared_ptr<Request>& m) { return m.get() == r; }))
                held.push_back(k);
        std::sort(held.begin(), held.end(), [&](size_t a, size_t b) { return slots_[a].formed < slots_[b].formed; });
        std::vector<size_t> gone;
        for (size_t k : held) {
            if (std::find(gone.begin(), gone.end(), k) != gone.end()) continue;
            const std::vector<size_t> later = abandon(k);
            gone.push_back(k);
            gone.insert(gone.end(), later.begin(), later.end());
        }
        for (size_t k : gone) {
            for (const auto& m : slots_[k].members)
                if (m.get() != r && !m->verify_.empty()) {
                    m->verify_.clear();
                    model_.retract(m->seq_, m->seq_.length());
                }
            vacate(k);
        }
    }
    void release() { --quieting_; }
    // The model for work on a stage's devices that is not a pass's stage (a history copied to or from host memory, a mark, a retract, a kept state, a drafter's chain): held for as long as the value lives, the expression it is called in.
    struct Held {
        Scheduler& s;
        explicit Held(Scheduler& sched) : s(sched) { s.hold(); }
        Held(const Held&) = delete;
        Held& operator=(const Held&) = delete;
        ~Held() { s.release(); }
    };

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
        // Every chain has run, so each ends now, its report then naming an ended chain.
        while (!chains_.empty()) end_chain(*chains_.front(), "");
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
    // Not while a request could not grow, nor a cancelled request, nor a job while a request waits for its seat or room; the slot beyond the stages takes the replies meanwhile (docs/SERVER.md, The service rule).
    bool continues(const Request& r) const {
        if (!r.seq_.in_flight() || stalled_now_ || r.cancel_.load() || decoding(r) || (r.job_ && jobs_yield_)) return false;
        // A boundary's pass retires before the next slice is formed, since its copy to host memory needs the sequence at rest (Cache::keep_boundary).
        if (r.boundary_kept_) return false;
        for (size_t k = 0; k < slots_.size(); ++k) {
            const Slot& s = slots_[k];
            if (!s.live) continue;
            const auto is = [&](const std::shared_ptr<Request>& m) { return m.get() == &r; };
            if (std::none_of(s.members.begin(), s.members.end(), is)) continue;
            if (s.members.size() != 1 || !s.ran || !s.wanting.empty() || s.formed > formed_before_) return false;
        }
        return true;
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
        // The service record's request times: the first token's wait for admission and time from reading, and later a step's gap spread over the tokens it published.
        if (!d.picks.empty()) {
            const Clock::time_point now = Clock::now();
            const auto ms = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            size_t later = d.picks.size();
            if (r.first_at_ == Clock::time_point{}) {
                record_.first_token(ms(r.submitted_, r.admitted_ == Clock::time_point{} ? now : r.admitted_), ms(r.received_, now));
                r.first_at_ = r.pushed_ = now;
                --later;
            }
            for (size_t i = 0; i < later; ++i) record_.inter_token(ms(r.pushed_, now) / (double)later);
            if (later == d.picks.size()) r.steps_.add(later);
            r.pushed_ = now;
        }
        for (Request::Token& t : d.picks) {
            r.last_id_ = t.id;
            // A row a token goes with comes back, once the reader has finished with another, for a later pass to fill.
            const bool row = !t.row.empty();
            r.push(std::move(t), row ? &r.logits_ : nullptr);
            ++published_total_;
        }
        ++r.steps_taken_;
        if (!proposer_) return;
        if (r.verify_.empty()) {
            r.record_.stepped();
        } else {
            r.record_.verified(d.sampled - 1, r.verify_.size() - 1);
            ++r.verifies_;
            r.drafted_ += r.verify_.size() - 1;
            r.kept_ += d.sampled - 1;
        }
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
            RequestView& v = live_.items[live_.count - 1];
            const auto ms = [](Clock::time_point a, Clock::time_point b) { return (uint64_t)std::max<int64_t>(0, std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count()); };
            v.queued_ms = ms(r.submitted_, r.admitted_ == Clock::time_point{} ? now : r.admitted_);
            if (r.first_at_ != Clock::time_point{}) v.first_token_ms = ms(r.received_, r.first_at_);
            v.steps = r.steps_taken_.load();
            v.verifies = r.verifies_.load();
            v.drafted = r.drafted_.load();
            v.kept = r.kept_.load();
            if (r.gen_.size() > 1 && r.pushed_ > r.first_at_) v.inter_token_ms = std::chrono::duration<double, std::milli>(r.pushed_ - r.first_at_).count() / (double)(r.gen_.size() - 1);
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
    const Clock::duration wait_bound_;    // the bound of a wait for a restore or a save while nothing better is measured: kWaitBound, or DiskOptions::wait in tests
    const std::optional<std::chrono::milliseconds> read_wait_;   // DiskOptions::read_wait
    const std::optional<double> recompute_row_ms_;   // DiskOptions::recompute_row_ms
    bool jobs_yield_ = false;             // this round a request waits for a job's seat or room, or could not grow, so no job continues (continues)
    bool stalled_now_ = false;            // this round a request could not grow, so nothing continues and the request its plan waits on leaves flight
    uint64_t formed_before_ = 0;          // the passes formed before this round's steps: a deferred host stage is for one of them, and a continuation follows one of them
    const Clock::time_point epoch_ = Clock::now();   // the stages' clocks count from here
    std::vector<char> plain_;             // per stage, whether it has a thread of its own (Recorder)
    infer::spec::Proposer* const proposer_;   // none drafts nothing
    const size_t draft_max_;
    const size_t draft_rows_;             // the draft rows a pass reserves (draft_rows)
    const bool priced_;
    std::vector<infer::spec::Proposer::Ask> asks_;   // the drafts being proposed, the first of them in use
    std::vector<const Request*> landed_;  // the requests of the pass being formed whose chains landed
    std::vector<std::shared_ptr<Chaining>> chains_;   // the chains on the head stage's thread not yet ended
    std::vector<std::pair<std::shared_ptr<const void>, std::string>> failed_restores_;   // restores whose part failed, reported by the stages' threads and met at the round's collect
    std::vector<Request*> askers_;        // the requests that may draft in the pass being formed
    std::vector<uint8_t> chained_;        // per such request, whether the proposer drafts for it, not its own history
    std::vector<size_t> asked_;           // per such request, its place among the proposer's asks
    std::vector<std::vector<double>> keeps_;   // per request that may draft, the chance it keeps each draft
    std::vector<size_t> takes_;           // per request that may draft, the drafts the price gives it
    std::vector<size_t> order_;           // those requests, the likeliest kept drafts first, as the columns and marks go to them
    infer::spec::PassTimes times_;        // the passes' measured cost, the scheduler thread's
    struct Retirement {
        Clock::time_point at;             // when the last pass of a kind retired, if one has
        bool any = false;
        double chain_ms = 0;              // the chains drafted since then
    };
    Retirement retirement_;               // of the passes
    std::atomic<size_t> draft_failures_{0};   // Stats::draft_failures
    std::atomic<size_t> holds_{0};            // Stats::holds
    std::atomic<size_t> stage_waits_{0};      // Stats::stage_waits
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
    std::atomic<uint64_t> prompt_passes_{0};                                     // Stats::prompt_passes
    std::atomic<uint64_t> continued_{0};                                         // Stats::continued
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
    double prompt_ms_row_ = 0;         // the scheduler thread's, a pass's milliseconds a prompt row, measured; 0 until a pass has
    static constexpr size_t kMeasuredRows = 64;   // the prompt rows a pass needs for its time to give a row's
    Cache cache_{model_, m_, pools_, max_seqs_, host_cap_, limit_, wait_bound_, read_wait_, recompute_row_ms_, epoch_};
    // The waiting lines, as the cache's calls that search them take them.
    Cache::Lines lines() { return {queue_, paused_}; }
public:
    // The cache's handler of each loss (Cache::on_move), which tests set before run.
    std::function<void(const Moved&)>& on_move = cache_.on_move;
private:
    std::atomic<int> quieting_{0};        // the scheduler's thread waits for the recorders or holds them, possibly under the lock
    bool acted_ = false;               // the scheduler thread's, whether this round advanced, collected, retired or formed anything
    bool encoded_ = false;             // and whether it ran a piece of an image's encode, which a round forming no pass may have done
    size_t pieces_since_replies_ = 0;  // an image's pieces queued since the last pass that carried replies
    double pieces_host_ms_ = 0;        // and this thread's time queuing them
    double piece_ms_ = 0;              // a piece's measured time, from the replies' passes that waited behind pieces or this thread's time running them, 0 before any did
    uint64_t jobs_posted_ = 0;           // the scheduler thread's, the hand-overs to recorders so far
    uint64_t arrivals_ = 0, seen_ = 0;   // under the lock, the requests and follows submitted, and how many the round had seen as it began
    std::map<std::vector<size_t>, size_t> class_ids_;   // the scheduler thread's: each row class Model::row_class gave, numbered as first seen, for the price table
    ServiceRecord record_{model_.stage_count()};     // before the recorders' lanes, whose threads post to it
    std::vector<std::unique_ptr<std::mutex>> locks_;   // the recorders' lanes' locks; before the recorders, which refer to them
    bool quick_turn_ = true;   // the scheduler thread's: whether the next replies' pass may take the quick prompt, every other one
    std::atomic<uint64_t> rule_replies_{0}, rule_chunks_{0}, rule_mixed_{0};   // written by the scheduler thread, read by stats(): Stats::rule_*
    std::vector<std::unique_ptr<Recorder>> recorders_;   // per stage, one where Model::stage_waits; before the lanes, whose threads use them
    infer::StageLanes lanes_;   // the recorders' threads (Recorder::cv, quit); last, so they end before anything they record on
};

} // namespace server


