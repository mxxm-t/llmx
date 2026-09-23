#pragma once
// The scheduler of docs/SERVER.md: one thread drives the model for every
// request, one Model::forward per iteration carrying every decoding
// request's next token and a slice of a prefilling request's prompt, the
// logits sampled on the host per request into that request's channel.
// Connection threads submit requests and drain channels; nothing else
// touches the model. Admission is by the KV pool's budget: a request is
// taken when the pool can hold its prompt and its max_tokens, in queue
// order, and nothing admitted is ever evicted. Finished requests stay a
// while as prefix donors: a new prompt that repeats a donor's tokens
// forks the donor's full blocks and prefills only what follows.
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
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
};

// One request from submission to completion. The connection thread reads
// the channel: `next` blocks until a token or the end. Everything below
// the channel belongs to the scheduler thread.
class Request {
public:
    Request(std::vector<uint32_t> prompt, SampleParams params)
        : prompt_(std::move(prompt)), params_(std::move(params)) {}

    // Blocks for the next sampled id; false when the request has ended,
    // with `finish` set: "eos", "stop", "length", "cancel" or "error".
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
    // Set by the connection thread when the client goes away; the
    // scheduler drops the request at its next iteration.
    void cancel() { cancel_.store(true); }
    const std::vector<uint32_t>& prompt() const { return prompt_; }
    const SampleParams& params() const { return params_; }
    size_t generated() const { return generated_.load(); }
    // Prompt tokens taken from a donor's cache rather than prefilled.
    size_t reused() const { return reused_.load(); }

private:
    friend class Scheduler;
    void push(uint32_t id) {
        std::lock_guard<std::mutex> lk(m_);
        out_.push_back(id);
        generated_.fetch_add(1);
        cv_.notify_all();
    }
    void end(const std::string& why, const std::string& err = "") {
        std::lock_guard<std::mutex> lk(m_);
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
    std::atomic<size_t> generated_{0};
    std::atomic<size_t> reused_{0};

    // Scheduler state.
    infer::Sequence seq_;
    std::string finish_pending_;   // set by a sampled end, acted on after the pass
    size_t need_ = 0;              // blocks reserved for it at admission
    size_t prompt_done_ = 0;
    uint32_t last_id_ = 0;
    std::vector<uint32_t> gen_;
    std::string decoded_;
    infer::RNG rng_;
    std::vector<float> logits_;
};

// The queue is full: the request is refused now rather than waiting.
struct QueueFull : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class Scheduler {
public:
    Scheduler(infer::Model& model, const bpe::Tokenizer& tok, size_t max_seqs, size_t ubatch, size_t max_queue)
        : model_(model), tok_(tok), max_seqs_(max_seqs ? max_seqs : 1), ubatch_(ubatch ? ubatch : 512),
          max_queue_(max_queue ? max_queue : 1) {}

    // Tokens one request may hold, prompt and reply together: the model
    // context or the KV pool, whichever is smaller.
    size_t token_limit() const {
        return std::min((size_t)model_.config().context_length, model_.kv_blocks_total() * model_.kv_block_tokens());
    }

    // Queue a request; the handle's channel delivers its tokens. A prompt
    // the limit cannot hold is refused here, before it waits, and so is a
    // request arriving at a full queue.
    std::shared_ptr<Request> submit(std::vector<uint32_t> prompt, SampleParams params) {
        if (prompt.empty()) throw std::runtime_error("server: empty prompt");
        if (params.max_tokens <= 0) throw std::runtime_error("server: max_tokens must be positive");
        if (prompt.size() + (size_t)params.max_tokens > token_limit())
            throw std::runtime_error("server: prompt plus max_tokens exceeds the " +
                                     std::to_string(token_limit()) + " tokens a request may hold");
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
        size_t active = 0, queued = 0, donors = 0, prefix_hits = 0, prefix_tokens = 0;
    };
    Stats stats() const {
        std::lock_guard<std::mutex> lk(m_);
        return Stats{active_count_.load(), queue_.size(), donors_.size(), prefix_hits_, prefix_tokens_};
    }

    // The loop, in the caller's thread, until stop(). The only caller of
    // Model::forward for this model.
    void run() {
        std::vector<std::shared_ptr<Request>> active;
        std::vector<infer::BatchEntry> entries;
        std::vector<std::shared_ptr<Request>> wanting;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return stopping_ || !queue_.empty() || !active.empty(); });
                if (stopping_) break;
                // Admission, in queue order, by the pool's budget: every
                // admitted request reserves the blocks its prompt and
                // max_tokens can reach, whether or not it holds them yet.
                // Donors give theirs up, oldest first, when a request
                // needs them.
                while (!queue_.empty() && active.size() < max_seqs_) {
                    const auto& r = queue_.front();
                    if (r->cancel_.load()) { r->end("cancel"); queue_.pop_front(); continue; }
                    const size_t tokens = r->prompt_.size() + (size_t)r->params_.max_tokens;
                    const size_t bt = model_.kv_block_tokens();
                    const size_t need = (tokens + bt - 1) / bt;
                    while (reserved_ + need > model_.kv_blocks_total() && !donors_.empty()) drop_donor();
                    if (reserved_ + need > model_.kv_blocks_total()) break;
                    reserved_ += need;
                    r->need_ = need;
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
                // The whole prompt's extent, reused prefix included, so its slices take the kernels one pass over it would.
                entries.back().extent = r->prompt_.size();
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
                // A failed pass leaves every history as it was; the requests
                // in it end with the error rather than the loop.
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
    // A finished request kept for its cache: the tokens its history holds,
    // the sequence holding them and the blocks it has reserved. Shared
    // full blocks are immutable, so a fork of it is safe while it lives.
    struct Donor {
        std::vector<uint32_t> tokens;
        infer::Sequence seq;
        size_t blocks = 0;
    };

    // The donor sharing the longest run of full blocks with the prompt,
    // and the token count of that run; zero when no donor shares a block.
    // Tokens are compared, not hashed. Only whole blocks are shared since
    // a fork appends only into fresh blocks, and the last prompt token is
    // always prefilled so the request has logits to sample from.
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

    // A history for an admitted request: a fork of the best donor rolled
    // back to the shared blocks, or a fresh sequence. Under the lock.
    void admit(Request& r) {
        size_t shared = 0;
        const size_t d = best_donor(r.prompt_, shared);
        if (shared) {
            r.seq_ = model_.fork(donors_[d].seq);
            model_.truncate(r.seq_, shared);
            r.prompt_done_ = shared;
            r.reused_.store(shared);
            ++prefix_hits_;
            prefix_tokens_ += shared;
        } else {
            r.seq_ = model_.make_sequence();
        }
        r.rng_.seed(r.params_.seed);
    }

    // The oldest donor's blocks back to the pool. Under the lock.
    void drop_donor() {
        Donor& d = donors_.front();
        try { model_.reset(d.seq); } catch (const std::exception&) {}
        reserved_ -= d.blocks;
        donors_.pop_front();
    }

    // How many prompt tokens of r were in this pass.
    static size_t ubatch_slice(const std::shared_ptr<Request>& r, const std::vector<infer::BatchEntry>& entries) {
        for (const auto& e : entries)
            if (e.seq == &r->seq_ && e.ids != &r->last_id_) return e.n;
        return 0;
    }

    // One sampled token for a request whose logits are in: pushed to its
    // channel unless it ends the request; a request that ends is finished
    // after the pass, once every entry's logits have been read.
    void step(Request& r) {
        const bool has_eos = tok_.eos_id >= 0;
        const uint32_t id = infer::sample(r.logits_, r.params_.temp, r.params_.top_k, r.params_.top_p,
                                          r.params_.penalty, r.gen_, r.rng_);
        if (has_eos && id == (uint32_t)tok_.eos_id) { r.finish_pending_ = "eos"; return; }
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

    // A finished request becomes a donor when its history holds a full
    // block; otherwise its blocks go back at once. A donor keeps only the
    // blocks it holds reserved, and there are at most max_seqs donors, the
    // oldest going when a newcomer needs the room.
    void finish(std::vector<std::shared_ptr<Request>>& active, size_t i, const std::string& why,
                const std::string& err = "") {
        auto r = active[i];
        active.erase(active.begin() + (std::ptrdiff_t)i);
        const size_t bt = model_.kv_block_tokens();
        const size_t held = r->seq_.length();
        if (why != "error" && held >= bt) {
            std::lock_guard<std::mutex> lk(m_);
            while (donors_.size() >= max_seqs_) drop_donor();
            Donor d;
            d.tokens.assign(r->prompt_.begin(), r->prompt_.begin() + (std::ptrdiff_t)r->prompt_done_);
            d.tokens.insert(d.tokens.end(), r->gen_.begin(), r->gen_.end());
            d.tokens.resize(std::min(d.tokens.size(), held));
            d.seq = std::move(r->seq_);
            d.blocks = (held + bt - 1) / bt;
            reserved_ -= r->need_ - d.blocks;
            r->need_ = 0;
            donors_.push_back(std::move(d));
        } else {
            release(*r);
        }
        r->seq_ = infer::Sequence{};
        r->end(why, err);
        active_count_.store(active.size());
    }
    // reset waits for the last pass that touched the sequence, so its
    // blocks return to the pool only once the device is done with them.
    void release(Request& r) {
        try { model_.reset(r.seq_); } catch (const std::exception&) {}
        reserved_ -= r.need_;
        r.need_ = 0;
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
    size_t reserved_ = 0;   // blocks promised to admitted requests and held by donors
    bool stopping_ = false;
};

} // namespace server
