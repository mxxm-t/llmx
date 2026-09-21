#pragma once
// The scheduler of docs/SERVER.md: one thread drives the model for every
// request, one Model::forward per iteration carrying every decoding
// request's next token and a slice of a prefilling request's prompt, the
// logits sampled on the host per request into that request's channel.
// Connection threads submit requests and drain channels; nothing else
// touches the model. Admission is by the KV pool's budget: a request is
// taken when the pool can hold its prompt and its max_tokens, in queue
// order, and nothing admitted is ever evicted.
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

class Scheduler {
public:
    Scheduler(infer::Model& model, const bpe::Tokenizer& tok, size_t max_seqs, size_t ubatch)
        : model_(model), tok_(tok), max_seqs_(max_seqs ? max_seqs : 1), ubatch_(ubatch ? ubatch : 512) {}

    // Queue a request; the handle's channel delivers its tokens. A prompt
    // the context cannot hold is refused here, before it waits.
    std::shared_ptr<Request> submit(std::vector<uint32_t> prompt, SampleParams params) {
        if (prompt.empty()) throw std::runtime_error("server: empty prompt");
        if (params.max_tokens <= 0) throw std::runtime_error("server: max_tokens must be positive");
        const size_t context = (size_t)model_.config().context_length;
        if (prompt.size() + (size_t)params.max_tokens > context)
            throw std::runtime_error("server: prompt plus max_tokens exceeds the context of " +
                                     std::to_string(context) + " tokens");
        auto r = std::make_shared<Request>(std::move(prompt), std::move(params));
        {
            std::lock_guard<std::mutex> lk(m_);
            queue_.push_back(r);
        }
        cv_.notify_all();
        return r;
    }

    struct Stats {
        size_t active = 0, queued = 0;
    };
    Stats stats() const {
        std::lock_guard<std::mutex> lk(m_);
        return Stats{active_count_.load(), queue_.size()};
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
                while (!queue_.empty() && active.size() < max_seqs_) {
                    const auto& r = queue_.front();
                    if (r->cancel_.load()) { r->end("cancel"); queue_.pop_front(); continue; }
                    const size_t tokens = r->prompt_.size() + (size_t)r->params_.max_tokens;
                    const size_t bt = model_.kv_block_tokens();
                    const size_t need = (tokens + bt - 1) / bt;
                    if (reserved_ + need > model_.kv_blocks_total()) break;
                    reserved_ += need;
                    r->need_ = need;
                    r->seq_ = model_.make_sequence();
                    r->rng_.seed(r->params_.seed);
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
        for (auto& r : active) r->end("cancel");
        std::lock_guard<std::mutex> lk(m_);
        for (auto& r : queue_) r->end("cancel");
        queue_.clear();
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

    void finish(std::vector<std::shared_ptr<Request>>& active, size_t i, const std::string& why,
                const std::string& err = "") {
        auto r = active[i];
        active.erase(active.begin() + (std::ptrdiff_t)i);
        // reset waits for the last pass that touched the sequence, so its
        // blocks return to the pool only once the device is done with them.
        try { model_.reset(r->seq_); } catch (const std::exception&) {}
        r->seq_ = infer::Sequence{};
        reserved_ -= r->need_;
        r->end(why, err);
        active_count_.store(active.size());
    }

    infer::Model& model_;
    const bpe::Tokenizer& tok_;
    size_t max_seqs_, ubatch_;
    infer::ExecContext ctx_;
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Request>> queue_;
    std::atomic<size_t> active_count_{0};
    size_t reserved_ = 0;   // blocks promised to admitted requests
    bool stopping_ = false;
};

} // namespace server
