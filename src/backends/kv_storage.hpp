#pragma once
#include <atomic>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "backends/backend.hpp"

// The KV storage every backend's cache shares: per-layer K, V and optional auxiliary buffers grown through the backend's own alloc and copy, the byte accounting, the growth rule and the view checks (docs/src/backends-kv_storage.md).
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
    size_t allocated_bytes() const override { return size_mul(backed_.load(), total_block_bytes_); }
    size_t peak_bytes() const override { return peak_.load(); }
    // One growth to the whole budget, which copies nothing on a fresh storage.
    void back_all() override {
        if (max_) ensure(max_ - 1);
    }

    // Fixed when the storage is made, so a thread may read it while another grows the buffers on the device's thread.
    size_t layers() const { return layers_; }
    size_t heads() const { return heads_; }
    size_t dim() const { return dim_; }
    size_t block_tokens() const { return bt_; }
    KVType k_type() const { return kt_; }
    KVType v_type() const { return vt_; }
    // Block b of a layer's side starts at byte b times these and holds [kv_head][token][head_dim] in the side's type.
    size_t k_block_bytes() const { return kblock_; }
    size_t v_block_bytes() const { return vblock_; }
    size_t auxiliary_width(size_t layer) const { return aux_widths_.at(layer); }
    size_t auxiliary_block_bytes(size_t layer) const { return auxblock_.at(layer); }
    // A layer's auxiliary block holds its tokens' rows, then with a pool of c one row of the width for each complete group of c tokens, which select_kv writes as the group completes.
    // Those rows follow from the tokens' rows and their positions, so whatever moves a token's position later must drop or recompute them; nothing does today.
    size_t auxiliary_pool(size_t layer) const { return aux_pools_.at(layer); }
    size_t auxiliary_rows_bytes(size_t layer) const { return size_mul(size_mul(bt_, aux_widths_.at(layer)), sizeof(float)); }
    size_t total_block_bytes() const { return total_block_bytes_; }
    bool backed(size_t id) const { return id < backed_; }
    const BufferPtr& k_buffer(size_t layer) const { return k_[layer]; }
    const BufferPtr& v_buffer(size_t layer) const { return v_[layer]; }
    const BufferPtr& auxiliary_buffer(size_t layer) const { return auxiliary_.at(layer); }

    // Backs blocks through `id` by the growth rule: the new buffers of every layer are allocated and hold the history before any is published, so a growth that throws leaves the storage and its accounting as they were, and a retry starts over.
    // alloc is zero-filled, which is what leaves a newly backed block reading as zeros.
    void ensure(size_t id) {
        if (id < backed_) return;
        if (id >= max_) fail("KV block outside the budget");
        const size_t want = growth_target(backed_, id, max_);
        const size_t kbytes = size_mul(want, kblock_), vbytes = size_mul(want, vblock_);
        // The old buffers and the new are held together while the copies run.
        const size_t peak = std::max(peak_.load(), size_add(allocated_bytes(), size_mul(want, total_block_bytes_)));
        std::vector<BufferPtr> nk(layers()), nv(layers()), na(layers());
        try {
            for (size_t l = 0; l < layers(); ++l) {
                nk[l] = owner_->alloc(kbytes, Memory::device);
                nv[l] = owner_->alloc(vbytes, Memory::device);
                if (auxblock_[l]) na[l] = owner_->alloc(size_mul(want, auxblock_[l]), Memory::device);
                if (k_[l]) {
                    owner_->copy(*nk[l], 0, *k_[l], 0, k_[l]->size());
                    owner_->copy(*nv[l], 0, *v_[l], 0, v_[l]->size());
                    if (auxiliary_[l]) owner_->copy(*na[l], 0, *auxiliary_[l], 0, auxiliary_[l]->size());
                }
            }
            for (size_t l = 0; l < layers(); ++l)
                if (k_[l]) {
                    retire(k_[l]);
                    retire(v_[l]);
                    if (auxiliary_[l]) retire(auxiliary_[l]);
                }
            after_growth(nk, nv);
        } catch (...) {
            // The new buffers can already be queued copy destinations, so the stream drains before they are released.
            owner_->sync();
            throw;
        }
        k_.swap(nk);
        v_.swap(nv);
        auxiliary_.swap(na);
        backed_ = want;
        peak_ = peak;
    }

    // Checks a view for an op on `layer`: the layer exists and the view's table covers its length plus its rows.
    // A writing op then backs the blocks its rows land in, in position order, and a reading op requires every block the view reaches to be backed, so which positions hold data is the caller's to know; it returns the blocks the view reaches.
    size_t check_view(const KVView& view, size_t layer, bool writing) {
        const size_t used = validate_view(view, layer, writing);
        if (writing) {
            const size_t sequence = size_add(view.length, view.nq);
            for (size_t t = view.length; t < sequence; t += bt_ - t % bt_)
                ensure((size_t)view.blocks[t / bt_]);
        } else {
            for (size_t b = 0; b < used; ++b)
                if (!backed((size_t)view.blocks[b])) fail("attention over unwritten KV blocks");
        }
        return used;
    }

    size_t validate_write_view(const KVView& view, size_t layer) const { return validate_view(view, layer, true); }

    void validate_auxiliary(size_t layer, CSlice auxiliary, size_t rows) const {
        const size_t width = aux_widths_.at(layer);
        if (!width) {
            if (auxiliary.buffer) fail("auxiliary KV input supplied to a layer without auxiliary storage");
            return;
        }
        if (!auxiliary.buffer) fail("auxiliary KV input missing");
        span(*auxiliary.buffer, size_mul(auxiliary.offset, sizeof(float)),
             size_mul(size_mul(rows, width), sizeof(float)));
    }

    void write_aux(size_t layer, const KVView* views, size_t n_views, CSlice auxiliary) {
        const size_t width = aux_widths_.at(layer);
        if (!width) return;
        size_t row0 = 0;
        const size_t row_bytes = size_mul(width, sizeof(float));
        for (size_t vi = 0; vi < n_views; ++vi) {
            const KVView& view = views[vi];
            size_t written = 0;
            while (written < view.nq) {
                const size_t token = view.length + written;
                const size_t in_block = token % bt_;
                const size_t count = std::min(view.nq - written, bt_ - in_block);
                const size_t id = (size_t)view.blocks[token / bt_];
                const size_t source_row = size_add(row0, written);
                const size_t source_off = size_mul(size_add(auxiliary.offset, size_mul(source_row, width)), sizeof(float));
                const size_t destination_off = size_add(size_mul(id, auxblock_[layer]), size_mul(in_block, row_bytes));
                owner_->copy(*auxiliary_[layer], destination_off, *auxiliary.buffer, source_off,
                             size_mul(count, row_bytes));
                written += count;
            }
            row0 = size_add(row0, view.nq);
        }
    }

protected:
    // Nothing is backed yet: the whole budget, `max_tokens` rounded up to whole blocks of every layer's K, V and auxiliary rows, is checked here in bytes.
    // `prefix` starts every error the storage throws, the backend's own.
    BlockKVStorage(Backend& owner, const char* prefix, size_t block_tokens, size_t layers, size_t heads, size_t dim,
                   size_t max_tokens, KVType kt, KVType vt, const std::vector<size_t>& auxiliary_widths = {}, const std::vector<size_t>& auxiliary_pools = {})
        : owner_(&owner), prefix_(prefix), bt_(block_tokens), heads_(heads), dim_(dim), layers_(layers),
          max_(blocks_for(max_tokens, block_tokens)), kt_(kt), vt_(vt),
          kblock_(size_mul(size_mul(size_mul(heads, block_tokens), dim), kv_elem_bytes(kt))),
          vblock_(size_mul(size_mul(size_mul(heads, block_tokens), dim), kv_elem_bytes(vt))),
          k_(layers), v_(layers), auxiliary_(layers), aux_widths_(layers), aux_pools_(layers), auxblock_(layers) {
        if (!auxiliary_widths.empty() && auxiliary_widths.size() != layers)
            fail("auxiliary width count differs from its layers");
        if (!auxiliary_pools.empty() && auxiliary_pools.size() != layers)
            fail("auxiliary pool count differs from its layers");
        size_t total = size_mul(layers, size_add(kblock_, vblock_));
        for (size_t l = 0; l < layers; ++l) {
            aux_widths_[l] = auxiliary_widths.empty() ? 0 : auxiliary_widths[l];
            aux_pools_[l] = auxiliary_pools.empty() ? 0 : auxiliary_pools[l];
            if (aux_pools_[l] && (!aux_widths_[l] || block_tokens % aux_pools_[l])) fail("an auxiliary pool needs auxiliary rows and a block of whole groups");
            auxblock_[l] = size_mul(size_mul(size_add(block_tokens, aux_pools_[l] ? block_tokens / aux_pools_[l] : 0), aux_widths_[l]), sizeof(float));
            total = size_add(total, auxblock_[l]);
        }
        total_block_bytes_ = total;
        size_mul(max_, total_block_bytes_);
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
    size_t validate_view(const KVView& view, size_t layer, bool writing) const {
        const size_t sequence = size_add(view.length, view.nq);
        const size_t used = blocks_for(sequence, bt_);
        if (layer >= layers() || used > view.n_blocks || (used && !view.blocks))
            fail(writing ? "KV write outside the view" : "attention outside the KV view");
        const size_t first = writing && view.nq ? view.length / bt_ : used;
        for (size_t b = first; b < used; ++b)
            if (view.blocks[b] < 0 || (size_t)view.blocks[b] >= max_) fail("KV block outside the budget");
        return used;
    }

    [[noreturn]] void fail(const char* what) const {
        throw std::runtime_error(std::string(prefix_) + ": " + what);
    }

    Backend* owner_;
    const char* prefix_;
    size_t bt_, heads_, dim_;
    const size_t layers_;
    size_t max_;
    // Growth runs on the device's thread, a stage's in a server, while the scheduler's thread reads what is backed for its accounting.
    std::atomic<size_t> backed_{0}, peak_{0};
    KVType kt_, vt_;
    size_t kblock_, vblock_, total_block_bytes_ = 0;
    std::vector<BufferPtr> k_, v_;
    std::vector<BufferPtr> auxiliary_;
    std::vector<size_t> aux_widths_, aux_pools_, auxblock_;
};

} // namespace backend
