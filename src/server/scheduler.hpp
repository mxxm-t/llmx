#pragma once
// The scheduler of docs/SERVER.md: one thread drives the model, one Model::forward per iteration carrying every decoding request's next token and a slice of a prefilling prompt, sampling each request's logits into its channel.
// Admission is by the KV pool's budget, in queue order. A capped request reserves its whole reach up front and is never paused; an uncapped one reserves its prompt and grows as it generates, and when the pool runs out the latest admitted uncapped request is paused and queued again with its history, resuming where it stopped. Finished requests stay as prefix donors, whose full blocks a repeating prompt forks.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "inference/sampler.hpp"
#include "model/arch_qwen.hpp"
#include "tokenizer/tokenizer.hpp"

namespace server {

struct SampleParams {
    int max_tokens = 64;
    float temp = 0.8f;
    int top_k = 40;
    float top_p = 0.95f;
    float penalty = 1.0f;
    uint64_t seed = 0;
    std::vector<std::string> stop;
    // No cap from the client: max_tokens is what the request may hold, and its blocks are reserved as it grows.
    bool until_limit = false;
};

// One request from submission to completion.
// The connection thread reads the channel: `next` blocks until a token or the end.
// Everything below the channel belongs to the scheduler thread.
class Request {
public:
    Request(std::vector<uint32_t> prompt, SampleParams params)
        : prompt_(std::move(prompt)), params_(std::move(params)), prompt_tokens_(prompt_.size()),
          submitted_(Clock::now()) {}

    // Blocks for the next sampled id; false when the request has ended, with `finish` set: "eos", "stop", "length", "cancel" or "error".
    bool next(uint32_t& id) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return !out_.empty() || done_; });
        if (out_.empty()) return false;
        id = out_.front();
        out_.pop_front();
        return true;
    }
    std::string finish() const {
        std::lock_guard<std::mutex> lk(m_);
        return finish_;
    }
    std::string error() const {
        std::lock_guard<std::mutex> lk(m_);
        return error_;
    }
    // Set by the connection thread when the client goes away; the scheduler drops the request at its next iteration.
    void cancel() { cancel_.store(true); }
    // The client's prompt tokens; a paused request's queued prompt also holds what it generated.
    size_t prompt_tokens() const { return prompt_tokens_; }
    // Prompt tokens taken from a donor's cache rather than prefilled.
    size_t reused() const { return reused_.load(); }
    // Milliseconds from admission to the first token, and from the first token to the end; read once the request has ended.
    struct Timings { double queued_ms = 0, prompt_ms = 0, predicted_ms = 0; };
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
    using Clock = std::chrono::steady_clock;
    void push(uint32_t id) {
        std::lock_guard<std::mutex> lk(m_);
        if (first_ == Clock::time_point{}) first_ = Clock::now();
        out_.push_back(id);
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

    std::vector<uint32_t> prompt_;
    SampleParams params_;
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::deque<uint32_t> out_;
    bool done_ = false;
    std::string finish_, error_;
    std::atomic<bool> cancel_{false};
    std::atomic<size_t> reused_{0};
    const size_t prompt_tokens_;
    Clock::time_point submitted_, admitted_, first_, ended_;   // admitted_ is set once, under m_

    // Scheduler state.
    infer::Sequence seq_;
    std::string finish_pending_;   // set by a sampled end, acted on after the pass
    std::vector<size_t> need_;     // blocks reserved for it, per cache pool
    size_t prompt_done_ = 0;
    size_t fresh_ = 0;             // the prompt tokens this admission prefills, past any reused prefix
    uint32_t last_id_ = 0;
    std::vector<uint32_t> gen_;
    std::string decoded_;
    infer::RNG rng_;
    std::vector<float> logits_;
    uint64_t admission_ = 0;       // order of admission, for choosing whom to pause
    bool resumed_ = false;         // paused once and queued again with its history
    size_t resumed_gen_ = 0;       // generated tokens already folded into prompt_
};

// The queue is full: the request is refused now rather than waiting.
struct QueueFull : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// A request whose prompt and max_tokens pass what one request may hold (Scheduler::token_limit).
struct TooLong : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class Scheduler {
public:
    Scheduler(infer::Model& model, const bpe::Tokenizer& tok, size_t max_seqs, size_t max_queue)
        : model_(model), tok_(tok), max_seqs_(max_seqs ? max_seqs : 1), ubatch_(model.prefill_batch()),
          max_queue_(max_queue ? max_queue : 1), reserved_(model.kv_pools(), 0) {}

    // Tokens one request may hold, prompt and reply together: the model context or the KV pool, whichever is smaller.
    size_t token_limit() const {
        return std::min((size_t)model_.config().context_length, model_.kv_tokens_total());
    }

    // Queue a request; the handle's channel delivers its tokens.
    // A prompt the limit cannot hold is refused here, before it waits, and so is a request arriving at a full queue.
    std::shared_ptr<Request> submit(std::vector<uint32_t> prompt, SampleParams params) {
        if (prompt.empty()) throw std::runtime_error("server: empty prompt");
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
    };
    Stats stats() const {
        std::lock_guard<std::mutex> lk(m_);
        return Stats{active_count_.load(), queue_.size(), donors_.size(), prefix_hits_, prefix_tokens_, (size_t)pauses_};
    }

    // The loop, in the caller's thread, until stop().
    // The only caller of Model::forward for this model.
    void run() {
        std::vector<std::shared_ptr<Request>> active;
        std::vector<infer::BatchEntry> entries;
        std::vector<std::shared_ptr<Request>> wanting;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return stopping_ || !queue_.empty() || !active.empty(); });
                if (stopping_) break;
                // Admission, in queue order, by the pool's budget: a capped request reserves the blocks its prompt and max_tokens can reach, an uncapped one its prompt and a growth step.
                // Donors give theirs up, oldest first, when a request needs them.
                while (!queue_.empty() && active.size() < max_seqs_) {
                    const auto& r = queue_.front();
                    if (r->cancel_.load()) { r->end("cancel"); queue_.pop_front(); continue; }
                    const size_t tokens = r->prompt_.size() + (r->params_.until_limit ? kGrowTokens : (size_t)r->params_.max_tokens - r->gen_.size());
                    std::vector<size_t> need = blocks_for(tokens);
                    while (!room_for(need) && !donors_.empty()) drop_donor();
                    if (!room_for(need)) break;
                    add(reserved_, need);
                    r->need_ = std::move(need);
                    r->admission_ = ++admissions_;
                    admit(*r);
                    active.push_back(r);
                    queue_.pop_front();
                }
                active_count_.store(active.size());
            }
            // Cancelled requests leave before the pass.
            for (size_t i = 0; i < active.size();) {
                if (active[i]->cancel_.load()) finish(active, i, "cancel");
                else ++i;
            }
            if (active.empty()) continue;
            grow(active);
            if (active.empty()) continue;

            // Decode entries first, then prompt slices up to ubatch tokens.
            entries.clear();
            wanting.clear();
            size_t budget = ubatch_;
            for (auto& r : active) {
                if (r->prompt_done_ < r->prompt_.size()) continue;
                entries.push_back(infer::BatchEntry{&r->seq_, &r->last_id_, 1, true});
                wanting.push_back(r);
            }
            for (auto& r : active) {
                if (r->prompt_done_ >= r->prompt_.size() || !budget) continue;
                const size_t n = std::min(budget, r->prompt_.size() - r->prompt_done_);
                const bool last = r->prompt_done_ + n == r->prompt_.size();
                entries.push_back(infer::BatchEntry{&r->seq_, r->prompt_.data() + r->prompt_done_, n, last});
                // The whole prompt's extent, reused prefix included, so its slices take the kernels one pass over it would; and its new tokens, which a streamed layer follows.
                entries.back().extent = r->prompt_.size();
                entries.back().fresh = r->fresh_;
                if (last) wanting.push_back(r);
                budget -= n;
            }
            try {
                model_.forward(ctx_, entries.data(), entries.size());
                // Commit the prompt progress now that the pass is submitted.
                for (auto& r : active)
                    if (r->prompt_done_ < r->prompt_.size())
                        r->prompt_done_ = std::min(r->prompt_.size(), r->prompt_done_ + ubatch_slice(r, entries));
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
        for (auto& r : queue_) r->end("cancel");
        queue_.clear();
        while (!donors_.empty()) drop_donor();
        active_count_.store(0);
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

    // Before a pass, every uncapped decoding request whose next token would pass its reservation takes another step.
    // When the pool is short, donors go first, then the latest admitted uncapped request is paused: its history becomes a donor and it is queued again at the front, so it resumes from its own blocks unless another request needs them.
    void grow(std::vector<std::shared_ptr<Request>>& active) {
        for (size_t i = 0; i < active.size();) {
            Request& r = *active[i];
            if (!r.params_.until_limit || r.prompt_done_ < r.prompt_.size() || !beyond(blocks_for(r.seq_.length() + 1), r.need_)) { ++i; continue; }
            std::vector<size_t> step = blocks_for(r.seq_.length() + 1 + kGrowTokens);
            for (size_t s = 0; s < step.size(); ++s) step[s] = step[s] > r.need_[s] ? step[s] - r.need_[s] : 0;
            {
                std::lock_guard<std::mutex> lk(m_);
                while (!room_for(step) && !donors_.empty()) drop_donor();
                if (room_for(step)) {
                    add(reserved_, step);
                    add(r.need_, step);
                    ++i;
                    continue;
                }
            }
            // The latest admitted uncapped request makes the room, which may be this one.
            size_t victim = i;
            for (size_t j = 0; j < active.size(); ++j)
                if (active[j]->params_.until_limit && active[j]->admission_ > active[victim]->admission_) victim = j;
            pause(active, victim);
            if (victim < i) --i;
        }
    }

    // A request's prompt, then what it generated since it was last queued: its cache holds the first seq_.length() of these, and a paused request resumes from all of them.
    // A decoding request has read its whole prompt and a prefilling one has generated nothing since, so a prompt paused part-way keeps the part it has not read.
    static std::vector<uint32_t> history(const Request& r) {
        std::vector<uint32_t> h = r.prompt_;
        h.insert(h.end(), r.gen_.begin() + (std::ptrdiff_t)r.resumed_gen_, r.gen_.end());
        return h;
    }

    // A paused request's history goes to the donors and the request to the front of the queue, its prompt now that history.
    void pause(std::vector<std::shared_ptr<Request>>& active, size_t i) {
        auto r = active[i];
        std::vector<uint32_t> h = history(*r);
        park(active, i, h);
        r->prompt_ = std::move(h);
        r->prompt_done_ = 0;
        r->resumed_gen_ = r->gen_.size();
        r->resumed_ = true;
        std::lock_guard<std::mutex> lk(m_);
        ++pauses_;
        queue_.push_front(r);
    }

    // A finished request kept for its cache: the tokens its history holds, the sequence holding them and the blocks it has reserved.
    // Shared full blocks are immutable, so a fork of it is safe while it lives.
    struct Donor {
        std::vector<uint32_t> tokens;
        infer::Sequence seq;
        std::vector<size_t> blocks;   // per cache pool
    };

    // The donor sharing the longest run of full blocks with the prompt, and the token count of that run; zero when no donor shares a block.
    // Tokens are compared, not hashed.
    // Only whole blocks are shared since a fork appends only into fresh blocks, and the last prompt token is always prefilled so the request has logits to sample from.
    size_t best_donor(const std::vector<uint32_t>& prompt, size_t& tokens) const {
        const size_t bt = model_.kv_block_tokens();
        size_t best = donors_.size();
        tokens = 0;
        for (size_t d = 0; d < donors_.size(); ++d) {
            const auto& t = donors_[d].tokens;
            size_t n = 0;
            const size_t limit = std::min(t.size(), prompt.size() - 1);
            while (n < limit && t[n] == prompt[n]) ++n;
            n = n / bt * bt;
            if (n > tokens) { tokens = n; best = d; }
        }
        return best;
    }

    // A history for an admitted request: a fork of the best donor rolled back to the shared blocks, or a fresh sequence.
    // Under the lock.
    void admit(Request& r) {
        {
            std::lock_guard<std::mutex> lk(r.m_);
            if (r.admitted_ == Request::Clock::time_point{}) r.admitted_ = Request::Clock::now();
        }
        size_t shared = 0;
        const size_t d = best_donor(r.prompt_, shared);
        if (shared) {
            r.seq_ = model_.fork(donors_[d].seq);
            model_.truncate(r.seq_, shared);
            r.prompt_done_ = shared;
            if (!r.resumed_) {
                r.reused_.store(shared);
                ++prefix_hits_;
                prefix_tokens_ += shared;
            }
        } else {
            r.seq_ = model_.make_sequence();
        }
        r.fresh_ = r.prompt_.size() - r.prompt_done_;
        if (!r.resumed_) r.rng_.seed(r.params_.seed);
    }

    // The oldest donor's blocks back to the pool. Under the lock.
    void drop_donor() {
        Donor& d = donors_.front();
        try { model_.reset(d.seq); } catch (const std::exception&) {}
        sub(reserved_, d.blocks);
        donors_.pop_front();
    }

    // How many prompt tokens of r were in this pass.
    static size_t ubatch_slice(const std::shared_ptr<Request>& r, const std::vector<infer::BatchEntry>& entries) {
        for (const auto& e : entries)
            if (e.seq == &r->seq_ && e.ids != &r->last_id_) return e.n;
        return 0;
    }

    // One sampled token for a request whose logits are in: pushed to its channel unless it ends the request; a request that ends is finished after the pass, once every entry's logits have been read.
    void step(Request& r) {
        // A resumed request's prompt ends with the last token it sent, so the logits after it are those its next token is sampled from, as they would have been.
        const uint32_t id = infer::sample(r.logits_, r.params_.temp, r.params_.top_k, r.params_.top_p,
                                          r.params_.penalty, r.gen_, r.rng_);
        if (tok_.is_eos(id)) { r.finish_pending_ = "eos"; return; }
        r.gen_.push_back(id);
        r.last_id_ = id;
        r.push(id);
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
        std::fprintf(stderr, "request: %zu prompt tokens (%zu reused), %zu generated, %.0f ms queued, %.0f ms to first token, %.1f tok/s, %s%s\n",
                     r->prompt_tokens(), r->reused(), r->gen_.size(), t.queued_ms, t.prompt_ms,
                     t.predicted_ms > 0 ? 1000.0 * (double)(r->gen_.size() > 1 ? r->gen_.size() - 1 : 0) / t.predicted_ms : 0.0,
                     why.c_str(), r->resumed_ ? ", resumed after a pause" : "");
    }

    // A request leaves the active set, its history kept as a donor when it holds a full block and its blocks returned otherwise.
    // A donor keeps only the blocks it holds reserved, and there are at most max_seqs donors, the oldest going when a newcomer needs the room.
    void park(std::vector<std::shared_ptr<Request>>& active, size_t i, const std::vector<uint32_t>& h) {
        auto r = active[i];
        active.erase(active.begin() + (std::ptrdiff_t)i);
        const size_t bt = model_.kv_block_tokens();
        const size_t held = r->seq_.length();
        if (held >= bt) {
            std::lock_guard<std::mutex> lk(m_);
            while (donors_.size() >= max_seqs_) drop_donor();
            Donor d;
            d.tokens.assign(h.begin(), h.begin() + (std::ptrdiff_t)std::min(h.size(), held));
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
        for (size_t s = 0; s < b.size(); ++s) {
            const size_t bt = model_.kv_pool_block_tokens(s);
            b[s] = std::min((tokens + bt - 1) / bt, model_.kv_pool_blocks(s));
        }
        return b;
    }
    // Whether every pool can take `more` beside what is reserved.
    bool room_for(const std::vector<size_t>& more) const {
        for (size_t s = 0; s < more.size(); ++s)
            if (reserved_[s] + more[s] > model_.kv_pool_blocks(s)) return false;
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
    std::deque<Donor> donors_;
    std::atomic<size_t> active_count_{0};
    size_t prefix_hits_ = 0, prefix_tokens_ = 0;   // under the lock
    uint64_t admissions_ = 0;   // the scheduler thread's
    uint64_t pauses_ = 0;       // under the lock
    std::vector<size_t> reserved_;   // per cache pool, blocks promised to admitted requests and held by donors
    bool stopping_ = false;
};

} // namespace server
