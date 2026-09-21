#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "backends/backend.hpp"

namespace infer {

// Logical KV bookkeeping, identical on every backend (docs/KV-CACHE.md).
// Physical blocks live in a backend::KVStorage; this layer only hands out ids
// and counts references. Ids are dense from zero, so storage can grow toward
// the budget on demand instead of allocating it up front.
//
// Every vector is reserved to the budget when configured, so alloc, release,
// abort and reset publish their state without an allocation that could throw
// half way. Sequences hold the pool's address, so a pool stays where it was
// constructed: it is neither copyable nor movable, and it can be configured
// only while nothing is allocated from it.
class BlockPool {
public:
    BlockPool() = default;
    explicit BlockPool(size_t max_blocks) { configure(max_blocks); }
    BlockPool(const BlockPool&) = delete;
    BlockPool& operator=(const BlockPool&) = delete;

    void configure(size_t max_blocks) {
        if (in_use()) throw std::logic_error("KV cache: pool reconfigured while blocks are held");
        std::vector<int32_t> f;
        std::vector<uint32_t> r;
        f.reserve(max_blocks);
        r.reserve(max_blocks);
        free_.swap(f);
        refs_.swap(r);
        max_ = max_blocks;
        next_ = 0;
    }

    int32_t alloc() {
        if (!free_.empty()) {
            const int32_t id = free_.back();
            free_.pop_back();
            refs_[(size_t)id] = 1;
            return id;
        }
        if (next_ >= max_) throw std::runtime_error("KV cache: block budget exhausted");
        refs_.push_back(1);
        return (int32_t)next_++;
    }

    void retain(int32_t id) {
        uint32_t& r = refs_.at((size_t)id);
        if (r == 0) throw std::logic_error("KV cache: retain of a free block");
        if (r == std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("KV cache: block reference count overflow");
        ++r;
    }

    // On the eager CPU backend every reader of a block has finished by the
    // time it is released; an async backend must also wait for completion
    // before the id is reused.
    void release(int32_t id) {
        uint32_t& r = refs_.at((size_t)id);
        if (r == 0) throw std::logic_error("KV cache: block released twice");
        if (--r == 0) free_.push_back(id);
    }

    size_t in_use() const { return next_ - free_.size(); }
    size_t max_blocks() const { return max_; }

private:
    size_t max_ = 0, next_ = 0;
    std::vector<int32_t> free_;
    std::vector<uint32_t> refs_;
};

// One sequence's block table and committed length. Blocks for a step are
// taken before the write and committed after the whole step succeeds, so a
// step that fails leaves the previous history valid and returns the blocks it
// took. A sequence owns its blocks: it cannot be copied, and destroying or
// moving from it returns them to the pool.
class KVSequence {
public:
    KVSequence() = default;
    KVSequence(BlockPool* pool, size_t block_tokens)
        : pool_(pool), block_tokens_(block_tokens) {
        if (!pool || block_tokens == 0)
            throw std::invalid_argument("KV cache: sequence needs a pool and a block size");
        blocks_.reserve(pool->max_blocks());
    }
    KVSequence(const KVSequence&) = delete;
    KVSequence& operator=(const KVSequence&) = delete;
    KVSequence(KVSequence&& o) noexcept { swap(o); }
    KVSequence& operator=(KVSequence&& o) noexcept {
        if (this != &o) { release_all(); swap(o); }
        return *this;
    }
    ~KVSequence() { release_all(); }

    size_t length() const { return length_; }
    size_t n_blocks() const { return blocks_.size(); }

    // Make positions [length, length + n) addressable.
    void prepare(size_t n) {
        if (!pool_) throw std::logic_error("KV cache: sequence is not bound to a pool");
        if (pending_) throw std::logic_error("KV cache: step already in progress");
        if (n > std::numeric_limits<size_t>::max() - length_)
            throw std::runtime_error("KV cache: sequence length overflow");
        const size_t total = length_ + n;
        const size_t need = total / block_tokens_ + (total % block_tokens_ != 0);
        // The pool may have been reconfigured larger since this sequence was
        // bound; reserve to its current budget before taking any id, so no
        // push_back below can throw with an unrecorded id in hand.
        blocks_.reserve(pool_->max_blocks());
        try {
            while (blocks_.size() < need) blocks_.push_back(pool_->alloc());
        } catch (...) {
            abort();
            throw;
        }
        pending_ = n;
    }

    void commit() noexcept {
        length_ += pending_;
        pending_ = 0;
    }

    void abort() noexcept {
        shrink_to(length_ / block_tokens_ + (length_ % block_tokens_ != 0));
        pending_ = 0;
    }

    // Roll the committed history back to `length`, returning the blocks
    // beyond it. Used when a multi-step operation fails part way.
    void truncate(size_t length) noexcept {
        if (length < length_) length_ = length;
        pending_ = 0;
        shrink_to(length_ / block_tokens_ + (length_ % block_tokens_ != 0));
    }

    // Start a new history. Blocks return to the pool; the backend keeps the
    // physical storage they occupied, so a reused sequence does not reallocate.
    void reset() noexcept { truncate(0); }

    // The rows prepared and not yet committed are this pass's queries.
    backend::KVView view(backend::KVStorage* storage) const {
        return {storage, blocks_.data(), blocks_.size(), length_, pending_};
    }

private:
    void shrink_to(size_t keep) noexcept {
        while (blocks_.size() > keep) {
            pool_->release(blocks_.back());
            blocks_.pop_back();
        }
    }
    void release_all() noexcept {
        if (pool_) shrink_to(0);
        length_ = pending_ = 0;
    }
    void swap(KVSequence& o) noexcept {
        std::swap(pool_, o.pool_);
        std::swap(block_tokens_, o.block_tokens_);
        std::swap(length_, o.length_);
        std::swap(pending_, o.pending_);
        blocks_.swap(o.blocks_);
    }

    BlockPool* pool_ = nullptr;
    size_t block_tokens_ = 1, length_ = 0, pending_ = 0;
    std::vector<int32_t> blocks_;
};

} // namespace infer
