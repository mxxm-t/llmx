#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "backends/backend.hpp"

namespace infer {

// Logical KV bookkeeping, identical on every backend (docs/KV-CACHE.md).
// Physical blocks live in a backend::KVStorage; this layer only hands out ids and counts references.
// Ids are dense from zero, so storage can grow toward the budget on demand instead of allocating it up front.
//
// Every vector is reserved to the budget when configured, so alloc, release, abort and reset publish their state without an allocation that could throw half way.
// Sequences hold the pool's address, so a pool stays where it was constructed: it is neither copyable nor movable, and it can be configured only while nothing is allocated from it.
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

    // On the eager CPU backend every reader of a block has finished by the time it is released; an async backend must also wait for completion before the id is reused.
    void release(int32_t id) {
        uint32_t& r = refs_.at((size_t)id);
        if (r == 0) throw std::logic_error("KV cache: block released twice");
        if (--r == 0) free_.push_back(id);
    }

    size_t in_use() const { return next_ - free_.size(); }
    size_t max_blocks() const { return max_; }
    uint32_t refs(int32_t id) const { return refs_.at((size_t)id); }

private:
    size_t max_ = 0, next_ = 0;
    std::vector<int32_t> free_;
    std::vector<uint32_t> refs_;
};

// One sequence's block table and committed length.
// Blocks for a step are taken before the write and committed after the whole step succeeds, so a step that fails leaves the previous history valid and returns the blocks it took.
// A sequence owns its blocks: it cannot be copied, moving it transfers them, and destroying it or moving another sequence into it returns them to the pool.
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
    // A block another sequence shares is read-only, so a history that was truncated into a shared block cannot be appended to.
    void prepare(size_t n) {
        if (!pool_) throw std::logic_error("KV cache: sequence is not bound to a pool");
        if (pending_) throw std::logic_error("KV cache: step already in progress");
        if (n > std::numeric_limits<size_t>::max() - length_)
            throw std::runtime_error("KV cache: sequence length overflow");
        if (n && length_ % block_tokens_ && pool_->refs(blocks_[length_ / block_tokens_]) > 1)
            throw std::logic_error("KV cache: append into a block shared with another sequence");
        const size_t need = backend::blocks_for(length_ + n, block_tokens_);
        // The pool may have been reconfigured larger since this sequence was bound; reserve to its current budget before taking any id, so no push_back below can throw with an unrecorded id in hand.
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
        shrink_to(backend::blocks_for(length_, block_tokens_));
        pending_ = 0;
    }

    // Roll the committed history back to `length`, returning the blocks beyond it.
    // Used when a multi-step operation fails part way.
    void truncate(size_t length) noexcept {
        if (length < length_) length_ = length;
        pending_ = 0;
        shrink_to(backend::blocks_for(length_, block_tokens_));
    }

    // Start a new history.
    // Blocks return to the pool; the backend keeps the physical storage they occupied, so a reused sequence does not reallocate.
    void reset() noexcept { truncate(0); }

    // A second history holding our first `length` committed tokens, a whole number of blocks: every block below `length` is shared, read-only from now on, with no new physical blocks or copies of their contents; the logical block table is allocated.
    // A failure part way leaves nothing retained, since the fork releases what it holds as it unwinds.
    KVSequence fork(size_t length) const {
        if (!pool_) throw std::logic_error("KV cache: sequence is not bound to a pool");
        if (pending_) throw std::logic_error("KV cache: fork during a step");
        if (length > length_ || length % block_tokens_)
            throw std::logic_error("KV cache: a fork takes whole blocks of the history");
        KVSequence f(pool_, block_tokens_);
        for (size_t i = 0; i < length / block_tokens_; ++i) {
            pool_->retain(blocks_[i]);
            f.blocks_.push_back(blocks_[i]);
        }
        f.length_ = length;
        return f;
    }

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

// The slots of a model's recurrent state, each the same slot in every state storage: `live` of them for the sequences that hold a state at once, and `checkpoints` more for states kept at a position (docs/SPECULATIVE.md, section 1).
// The two sides share the slots and are counted apart, so a live slot is always there for a sequence the live side admits, whatever the checkpoints hold.
// A slot holds nothing a sequence must clear: a history of length 0 reads a zero state whatever its slot holds.
// Sequences hold the pool's address, so it is neither copied nor moved.
class SlotPool {
public:
    SlotPool() = default;
    SlotPool(const SlotPool&) = delete;
    SlotPool& operator=(const SlotPool&) = delete;

    void configure(size_t live, size_t checkpoints) {
        if (live_held_ || kept_held_) throw std::logic_error("state slots reconfigured while some are held");
        if (checkpoints > std::numeric_limits<size_t>::max() - live) throw std::runtime_error("inference: more state slots than a size holds");
        const size_t slots = live + checkpoints;
        free_.clear();
        free_.reserve(slots);
        for (size_t s = slots; s-- > 0;) free_.push_back(s);
        refs_.assign(slots, 0);
        live_ = live;
        kept_ = checkpoints;
    }
    size_t acquire() {
        if (live_held_ == live_) throw std::runtime_error("inference: every recurrent state slot is held");
        ++live_held_;
        return take();
    }
    void release(size_t s) noexcept {
        free_.push_back(s);
        --live_held_;
    }
    size_t available() const { return live_ - live_held_; }

    // A checkpoint's slot, counted by reference: a sequence and every fork reading it hold one each.
    size_t acquire_kept() {
        if (kept_held_ == kept_) throw std::runtime_error("inference: every checkpoint slot is held");
        ++kept_held_;
        const size_t s = take();
        refs_[s] = 1;
        return s;
    }
    void retain(size_t s) noexcept { ++refs_[s]; }
    void release_kept(size_t s) noexcept {
        if (--refs_[s]) return;
        free_.push_back(s);
        --kept_held_;
    }
    // A live slot becomes a checkpoint's, its state kept where it is, when the checkpoint side has room; the live side then has one more to give.
    bool keep_live(size_t s) noexcept {
        if (kept_held_ == kept_) return false;
        --live_held_;
        ++kept_held_;
        refs_[s] = 1;
        return true;
    }
    size_t kept_available() const { return kept_ - kept_held_; }

private:
    size_t take() {
        const size_t s = free_.back();
        free_.pop_back();
        return s;
    }
    std::vector<size_t> free_;   // taken from the back, slot 0 first
    std::vector<uint32_t> refs_;
    size_t live_ = 0, kept_ = 0, live_held_ = 0, kept_held_ = 0;
};

// A sequence's hold on one live slot of a SlotPool, returned when it is released, moved over or destroyed.
class StateSlot {
public:
    StateSlot() = default;
    StateSlot(const StateSlot&) = delete;
    StateSlot& operator=(const StateSlot&) = delete;
    StateSlot(StateSlot&& o) noexcept : pool_(o.pool_), slot_(o.slot_) { o.pool_ = nullptr; }
    StateSlot& operator=(StateSlot&& o) noexcept {
        if (this != &o) {
            release();
            pool_ = o.pool_;
            slot_ = o.slot_;
            o.pool_ = nullptr;
        }
        return *this;
    }
    ~StateSlot() { release(); }
    size_t slot() const { return slot_; }
    bool held() const { return pool_ != nullptr; }
    void take(SlotPool& pool) {
        if (pool_) return;
        slot_ = pool.acquire();
        pool_ = &pool;
    }
    void release() noexcept {
        if (pool_) pool_->release(slot_);
        pool_ = nullptr;
    }
    // The slot handed to the checkpoint side (SlotPool::keep_live), so this hold ends without returning it.
    void forget() noexcept { pool_ = nullptr; }

private:
    SlotPool* pool_ = nullptr;
    size_t slot_ = 0;
};

// A state kept at a position: one reference to a checkpoint's slot, copied by taking another, returned when the last goes.
class Checkpoint {
public:
    Checkpoint() = default;
    Checkpoint(SlotPool& pool, size_t slot, size_t pos) : pool_(&pool), slot_(slot), pos_(pos) {}
    Checkpoint(const Checkpoint& o) noexcept : pool_(o.pool_), slot_(o.slot_), pos_(o.pos_) {
        if (pool_) pool_->retain(slot_);
    }
    Checkpoint& operator=(const Checkpoint& o) noexcept {
        if (this != &o) {
            Checkpoint c(o);
            swap(c);
        }
        return *this;
    }
    Checkpoint(Checkpoint&& o) noexcept { swap(o); }
    Checkpoint& operator=(Checkpoint&& o) noexcept {
        if (this != &o) {
            release();
            swap(o);
        }
        return *this;
    }
    ~Checkpoint() { release(); }
    bool held() const { return pool_ != nullptr; }
    size_t slot() const { return slot_; }
    size_t pos() const { return pos_; }
    void release() noexcept {
        if (pool_) pool_->release_kept(slot_);
        pool_ = nullptr;
    }

private:
    void swap(Checkpoint& o) noexcept {
        std::swap(pool_, o.pool_);
        std::swap(slot_, o.slot_);
        std::swap(pos_, o.pos_);
    }
    SlotPool* pool_ = nullptr;
    size_t slot_ = 0, pos_ = 0;
};

} // namespace infer
