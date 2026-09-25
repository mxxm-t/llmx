#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "backends/backend.hpp"

// The KV storage every backend's cache shares: per-layer K and V buffers grown through the backend's own alloc and copy, the byte accounting, the growth rule and the view checks (docs/src/backends-kv_storage.md).
// A backend derives its storage from it and keeps its block size, its limits, its kernels and its error prefix.

namespace backend {

class BlockKVStorage : public KVStorage {
public:
    // The blocks backed once block `id` is written, from `backed` backed now under a budget of `max_blocks`: through `id`, and at least double what is backed, capped at the budget.
    // The one growth rule, static so the kv-cache test checks it without a storage.
    static size_t growth_target(size_t backed, size_t id, size_t max_blocks) {
        return std::max(id + 1, std::min(max_blocks, backed * 2));
    }

    size_t max_blocks() const override { return max_; }
    size_t allocated_bytes() const override { return backed_ * (kblock_ + vblock_) * k_.size(); }
    size_t peak_bytes() const override { return peak_; }

    size_t layers() const { return k_.size(); }
    size_t heads() const { return heads_; }
    size_t dim() const { return dim_; }
    size_t block_tokens() const { return bt_; }
    KVType k_type() const { return kt_; }
    KVType v_type() const { return vt_; }
    // Block b of a layer's side starts at byte b times these and holds [kv_head][token][head_dim] in the side's type.
    size_t k_block_bytes() const { return kblock_; }
    size_t v_block_bytes() const { return vblock_; }
    bool backed(size_t id) const { return id < backed_; }
    const BufferPtr& k_buffer(size_t layer) const { return k_[layer]; }
    const BufferPtr& v_buffer(size_t layer) const { return v_[layer]; }

    // Backs blocks through `id` by the growth rule. The new buffers of every layer are allocated and hold the history before any is published, so a growth that throws leaves the storage and its accounting as they were, and a retry starts over.
    // alloc is zero-filled, which is what leaves a newly backed block reading as zeros.
    void ensure(size_t id) {
        if (id < backed_) return;
        if (id >= max_) fail("KV block outside the budget");
        const size_t want = growth_target(backed_, id, max_);
        const size_t kbytes = size_mul(want, kblock_), vbytes = size_mul(want, vblock_);
        // The old buffers and the new are held together while the copies run.
        const size_t peak = std::max(peak_, size_add(allocated_bytes(), size_mul(size_add(kbytes, vbytes), layers())));
        std::vector<BufferPtr> nk(layers()), nv(layers());
        try {
            for (size_t l = 0; l < layers(); ++l) {
                nk[l] = owner_->alloc(kbytes, Memory::device);
                nv[l] = owner_->alloc(vbytes, Memory::device);
                if (k_[l]) {
                    owner_->copy(*nk[l], 0, *k_[l], 0, k_[l]->size());
                    owner_->copy(*nv[l], 0, *v_[l], 0, v_[l]->size());
                }
            }
            for (size_t l = 0; l < layers(); ++l)
                if (k_[l]) {
                    retire(k_[l]);
                    retire(v_[l]);
                }
            after_growth(nk, nv);
        } catch (...) {
            // The new buffers can already be queued copy destinations, so the stream drains before they are released.
            owner_->sync();
            throw;
        }
        k_.swap(nk);
        v_.swap(nv);
        backed_ = want;
        peak_ = peak;
    }

    // Checks a view for an op on `layer`: the layer exists and the view's table covers its length plus its rows.
    // A writing op then backs the blocks its rows land in, in position order; a reading op requires every block the view reaches to be written.
    // Returns the blocks the view reaches.
    size_t check_view(const KVView& view, size_t layer, bool writing) {
        const size_t sequence = size_add(view.length, view.nq);
        const size_t used = blocks_for(sequence, bt_);
        if (layer >= layers() || used > view.n_blocks)
            fail(writing ? "KV write outside the view" : "attention outside the KV view");
        if (writing) {
            for (size_t t = view.length; t < sequence; t += bt_ - t % bt_)
                ensure((size_t)view.blocks[t / bt_]);
        } else {
            for (size_t b = 0; b < used; ++b)
                if (!backed((size_t)view.blocks[b])) fail("attention over unwritten KV blocks");
        }
        return used;
    }

protected:
    // Nothing is backed yet. The whole budget, `max_tokens` rounded up to whole blocks of every layer's K and V at their types, is checked here in bytes, so every size the storage computes later is in range and every kv_alloc refuses a budget that overflows.
    // `prefix` starts every error the storage throws, the backend's own.
    BlockKVStorage(Backend& owner, const char* prefix, size_t block_tokens, size_t layers, size_t heads, size_t dim,
                   size_t max_tokens, KVType kt, KVType vt)
        : owner_(&owner), prefix_(prefix), bt_(block_tokens), heads_(heads), dim_(dim),
          max_(blocks_for(max_tokens, block_tokens)), kt_(kt), vt_(vt),
          kblock_(size_mul(size_mul(size_mul(heads, block_tokens), dim), kv_elem_bytes(kt))),
          vblock_(size_mul(size_mul(size_mul(heads, block_tokens), dim), kv_elem_bytes(vt))),
          k_(layers), v_(layers) {
        size_mul(size_mul(layers, max_), size_add(kblock_, vblock_));
    }

    Backend& owner() const { return *owner_; }

    // An old buffer a growth has copied from, handed over before the new ones are published: a backend whose copies run after they are enqueued keeps it until they retire.
    virtual void retire(const BufferPtr& old) = 0;
    // The grown buffers, holding the history, before they replace the old ones: a backend that caches something of them takes it here.
    // An override changes its own state only once nothing it does can throw, since a throw leaves the storage as it was.
    virtual void after_growth(const std::vector<BufferPtr>& keys, const std::vector<BufferPtr>& values) {
        (void)keys;
        (void)values;
    }

private:
    [[noreturn]] void fail(const char* what) const {
        throw std::runtime_error(std::string(prefix_) + ": " + what);
    }

    Backend* owner_;
    const char* prefix_;
    size_t bt_, heads_, dim_, max_, backed_ = 0, peak_ = 0;
    KVType kt_, vt_;
    size_t kblock_, vblock_;
    std::vector<BufferPtr> k_, v_;
};

} // namespace backend
