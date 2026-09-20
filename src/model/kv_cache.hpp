#pragma once
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "backends/backend.hpp"

namespace infer {

// Logical KV bookkeeping, identical on every backend (docs/KV-CACHE.md).
// Physical blocks live in a backend::KVStorage; this layer only hands out ids
// and counts references. Ids are dense from zero, so storage can grow toward
// the budget on demand instead of allocating it up front.
class BlockPool {
public:
    BlockPool() = default;
    explicit BlockPool(size_t max_blocks) : max_(max_blocks) {}

    int32_t alloc() {
        int32_t id;
        if (!free_.empty()) {
            id = free_.back();
            free_.pop_back();
        } else if (next_ < max_) {
            id = (int32_t)next_++;
            refs_.push_back(0);
        } else {
            throw std::runtime_error("KV cache: block budget exhausted");
        }
        refs_[(size_t)id] = 1;
        return id;
    }

    void retain(int32_t id) { ++refs_.at((size_t)id); }

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
// taken before the write and committed after it, so a step that fails leaves
// the previous history valid and returns the blocks it took.
class KVSequence {
public:
    KVSequence() = default;
    KVSequence(BlockPool* pool, size_t block_tokens)
        : pool_(pool), block_tokens_(block_tokens) {}

    size_t length() const { return length_; }
    size_t n_blocks() const { return blocks_.size(); }

    // Make positions [length, length + n) addressable.
    void prepare(size_t n) {
        if (pending_) throw std::logic_error("KV cache: step already in progress");
        const size_t need = (length_ + n + block_tokens_ - 1) / block_tokens_;
        try {
            while (blocks_.size() < need) blocks_.push_back(pool_->alloc());
        } catch (...) {
            abort();
            throw;
        }
        pending_ = n;
    }

    void commit() {
        length_ += pending_;
        pending_ = 0;
    }

    void abort() {
        shrink_to((length_ + block_tokens_ - 1) / block_tokens_);
        pending_ = 0;
    }

    // Start a new history. Blocks return to the pool; the backend keeps the
    // physical storage they occupied, so a reused sequence does not reallocate.
    void reset() {
        shrink_to(0);
        length_ = 0;
        pending_ = 0;
    }

    backend::KVView view(backend::KVStorage* storage) const {
        return {storage, blocks_.data(), blocks_.size(), length_};
    }

private:
    void shrink_to(size_t keep) {
        while (blocks_.size() > keep) {
            pool_->release(blocks_.back());
            blocks_.pop_back();
        }
    }

    BlockPool* pool_ = nullptr;
    size_t block_tokens_ = 1, length_ = 0, pending_ = 0;
    std::vector<int32_t> blocks_;
};

} // namespace infer
