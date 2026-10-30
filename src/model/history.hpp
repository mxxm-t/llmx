#pragma once
#include "model/runtime.hpp"

// A sequence's history, the one owner of every operation on it (docs/SPECULATIVE.md, section 1): fork, reset, retract, checkpoint, mark with its rerun, and an embedded drafter's chain of drafts past the history (section 7).
// Members of infer::Model, declared in its class (model/runtime.hpp), which includes this file after it.

namespace infer {

// Append IDs to the fixed-width tail, oldest first; reading an empty history uses padding, and an in-place shift reads ahead of its writes.
inline void Model::advance_tokens(uint32_t* dst, const uint32_t* previous, size_t history, const uint32_t* ids, size_t n) const noexcept {
    const size_t from_previous = n < token_width_ ? token_width_ - n : 0;
    for (size_t i = 0; i < from_previous; ++i) dst[i] = history ? previous[n + i] : token_padding_;
    for (size_t i = from_previous; i < token_width_; ++i) dst[i] = ids[n - (token_width_ - i)];
}

// A second history holding the first `length` tokens of `src`, which must be whole blocks in every storage: every block below `length` is shared and read-only from now on, so nothing is allocated or copied here.
// The fork inherits the tickets of the passes that wrote what it shares, and on a model that keeps a state it takes src's checkpoint at `length`, whose state its first pass reads in place.
inline Sequence Model::fork(const Sequence& src, size_t length) {
    if (src.owner_ != this) throw std::runtime_error("inference: sequence of another model");
    if (src.flights_) throw std::logic_error("inference: a fork of a sequence in flight");
    if (src.mark_.held()) throw std::logic_error("inference: a fork of a marked sequence");
    if (state_layers_ && (!src.kept_.held() || src.kept_.pos() != length))
        throw std::logic_error("inference: a fork of a model whose layers keep a recurrent state takes its source's checkpoint");
    if (length > src.length()) throw std::logic_error("KV cache: a fork takes whole blocks of the history");
    Sequence f;
    f.storage_of_ = src.storage_of_;
    f.length_.assign(stages_.size(), length);
    f.kv_.reserve(storages_.size());
    for (const KVSequence& kv : src.kv_) f.kv_.push_back(kv.fork(length));
    f.last_ = src.last_;
    f.owner_ = this;
    f.media_ = carried(src, length);
    // A fork of a history whose restore is still pending shares its record, so a failed copy is found in both.
    if (src.deferred_ && src.deferred_->restore) f.deferred_ = src.deferred_;
    f.from_.assign(stages_.size(), Sequence::kLive);
    if (state_layers_) {
        f.kept_ = src.kept_;
        std::fill(f.from_.begin(), f.from_.end(), f.kept_.slot());
    }
    return f;
}

// Start a new history, its blocks and state slot returned to their pools and their storage retained.
// Every pass ends in a submit or, on failure, a sync, so the sequence's last tickets cover everything that could still touch a block or slot: this waits for those and no more.
inline void Model::reset(Sequence& s) {
    settle(s, "a reset");
    s.deferred_.reset();
    drop_mark(s);
    for (auto& kv : s.kv_) kv.reset();
    s.media_.clear();
    std::fill(s.length_.begin(), s.length_.end(), 0);
    std::fill(s.from_.begin(), s.from_.end(), Sequence::kLive);
    s.kept_.release();
    s.state_.release();
}

// The history back to at most `length`, the one call that shortens it (docs/SPECULATIVE.md, section 1), returning the length reached: `length` wherever the caches hold it, else the sequence's checkpoint at or below it, else 0.
// The caller computes the rest again, and inside a mark it reaches `length` itself by running the state again from the mark's over the kept rows' saved inputs (Architecture::recur); a retract that throws keeps the mark.
inline size_t Model::retract(Sequence& s, size_t length) { return retract(s, length, {}); }

// With `defer` a rerun inside a mark goes to the caller as one stage item a device, while the lengths and blocks change here at once (docs/src/model-history.md).
inline size_t Model::retract(Sequence& s, size_t length, const Defer& defer) {
    settle(s, "a retract");
    if (s.mark_.held() && s.mark_.ran && length > s.mark_.pos && length < s.length()) {
        rerun(s, length - s.mark_.pos, defer);
        if (s.deferred_) {
            s.deferred_->hold = std::move(s.mark_.hold);
            s.mark_.ran = false;
        } else {
            drop_mark(s);
        }
        for (size_t& n : s.length_) n = std::min(n, length);
        for (auto& kv : s.kv_) kv.truncate(length);
        return length;
    }
    if (length >= s.length()) {
        if (s.mark_.ran) drop_mark(s);
        else if (s.mark_.held()) restore_mark(s);
        return s.length();
    }
    return rewind(s, length, defer ? &defer : nullptr);
}

// Whether a draft of the sequence may be asked for (draft): it holds a history and, on a model that keeps a state, a pass has fed it since its fork, since until then it reads its checkpoint in place and the drafter has no state to read.
inline bool Model::can_draft(const Sequence& s) const { return s.length() && (!keeps_state() || s.state_.held()); }

// Keep the sequence's state at its current length while one pass runs past it, so a retract into that pass reaches any of its rows exactly (docs/SPECULATIVE.md, section 1): a verify of drafts marks its history first.
// On a model that keeps a state the live slot becomes the mark's and the pass writes a fresh one, saving its rows' recurrent inputs; false when no mark is free, and a second mark is refused.
inline bool Model::mark(Sequence& s) {
    settle(s, "a mark");
    if (s.mark_.held()) throw std::logic_error("inference: a second mark");
    if (!state_layers_) return true;
    // Everything that may throw first, then the holds, which do not.
    const bool live = s.state_.held();
    std::vector<size_t> from = live ? std::vector<size_t>(stages_.size(), s.state_.slot()) : s.from_;
    if (!s.mark_.hold.take(slots_, live, live ? s.state_.slot() : 0)) return false;
    if (live) s.state_.forget();
    s.mark_.ran = false;
    s.mark_.pos = s.length();
    std::copy(from.begin(), from.end(), s.from_.begin());
    s.mark_.from.swap(from);
    return true;
}

// The sequence's state kept at its current length, between passes, as its checkpoint: a paused history keeps its state this way, its live slot becoming the checkpoint's with no copy.
// False when the model keeps no state or no checkpoint slot is free; the older checkpoint it replaces goes first, so its slot serves.
inline bool Model::keep(Sequence& s) {
    settle(s, "a keep");
    if (s.mark_.held()) throw std::logic_error("inference: a keep of a marked sequence");
    if (!state_layers_) return false;
    if (s.kept_.held() && s.kept_.pos() == s.length()) return true;
    if (!s.state_.held()) return false;
    s.kept_.release();
    if (!slots_.keep_live(s.state_.slot())) return false;
    s.kept_ = Checkpoint(slots_, s.state_.slot(), s.length());
    std::fill(s.from_.begin(), s.from_.end(), s.state_.slot());
    s.state_.forget();
    return true;
}

// The position of the sequence's checkpoint, which a fork of it takes and a retract reaches.
inline std::vector<size_t> Model::kv_free() const {
    std::vector<size_t> f;
    for (const Device* d : storages_) f.push_back(d->pool.max_blocks() - d->pool.in_use());
    return f;
}

inline std::vector<size_t> Model::kv_held(const Sequence& q) const {
    std::vector<size_t> h(storages_.size(), 0);
    for (size_t s = 0; s < h.size() && s < q.kv_.size(); ++s) h[s] = q.kv_[s].n_blocks();
    return h;
}

// Each pool's references from the set are counted against the pool's own, so a block another sequence holds, or the fork still to be made would share, is not freed.
inline Model::Freed Model::freed(const std::vector<const Sequence*>& set, const Sequence* kept, size_t kept_len) const {
    Freed f;
    f.blocks.assign(storages_.size(), 0);
    for (size_t s = 0; s < storages_.size(); ++s) {
        const BlockPool& pool = storages_[s]->pool;
        std::vector<uint32_t> refs(pool.max_blocks(), 0);
        for (const Sequence* q : set)
            if (q && s < q->kv_.size())
                for (int32_t id : q->kv_[s].blocks()) ++refs[(size_t)id];
        if (kept && s < kept->kv_.size()) {
            const std::vector<int32_t>& b = kept->kv_[s].blocks();
            for (size_t i = 0; i < b.size() && i < backend::blocks_for(kept_len, kv_pool_block_tokens(s)); ++i) refs[(size_t)b[i]] = 0;
        }
        for (size_t id = 0; id < refs.size(); ++id) f.blocks[s] += refs[id] && refs[id] == pool.refs((int32_t)id);
    }
    if (!state_layers_) return f;
    std::vector<uint32_t> held(slots_.refs_size(), 0);
    for (const Sequence* q : set)
        if (q && q->kept_.held()) ++held[q->kept_.slot()];
    if (kept && kept->kept_.held()) held[kept->kept_.slot()] = 0;
    for (size_t slot = 0; slot < held.size(); ++slot) f.slots += held[slot] && held[slot] == slots_.refs(slot);
    return f;
}

inline std::optional<size_t> Model::checkpoint(const Sequence& s) const {
    if (!s.kept_.held()) return std::nullopt;
    return s.kept_.pos();
}

// The host memory a copy of `length` tokens takes (HostHistory::held): per device, whole blocks of K, V and auxiliary rows, and one slot of state, in whole slabs.
// Without `blocks` it is the slot alone, and from `first` on and without `state` the blocks of a range alone.
inline size_t Model::host_bytes(size_t length, bool blocks, size_t first, bool state) const {
    size_t n = 0;
    for (const auto& d : devices_) {
        size_t bytes = 0;
        if (const auto* st = blocks ? dynamic_cast<const backend::BlockKVStorage*>(d->storage.get()) : nullptr)
            bytes += (backend::blocks_for(length, st->block_tokens()) - first / st->block_tokens()) * st->total_block_bytes();
        if (state && d->states)
            for (size_t l = 0; l < d->states->layers(); ++l) bytes = backend::size_add(bytes, d->states->slot_bytes(l));
        if (state && d->carry && state_layers_) bytes += plan_.residual * sizeof(float);
        if (state && token_width_ && d.get() == devices_[stages_[token_stage_].device].get()) bytes = backend::size_add(bytes, backend::size_mul(token_width_, sizeof(uint32_t)));
        n += backend::blocks_for(bytes, kHostSlab) * kHostSlab;
    }
    return n;
}

namespace detail {
// The host image's ordered KV runs: each layer's K, V and auxiliary planes, joining consecutive physical blocks.
template<class F> inline void block_runs(const backend::BlockKVStorage& st, const int32_t* ids, size_t n, F&& take) {
    for (size_t l = 0; l < st.layers(); ++l)
        for (unsigned side = 0; side < 3; ++side) {
            const size_t block = side == 0 ? st.k_block_bytes() : side == 1 ? st.v_block_bytes() : st.auxiliary_block_bytes(l);
            if (!block) continue;
            for (size_t i = 0; i < n;) {
                size_t e = i + 1;
                while (e < n && ids[e] == ids[e - 1] + 1) ++e;
                take(l, side, (size_t)ids[i] * block, (e - i) * block);
                i = e;
            }
        }
}

// One complete state slot per layer, including that layer's extra carried rows.
template<class F> inline void state_runs(backend::StateStorage& states, size_t slot, F&& take) {
    for (size_t l = 0; l < states.layers(); ++l) {
        const size_t bytes = states.slot_bytes(l);
        take(states.layer(l), slot * bytes, bytes);
    }
}

// Copies between device storage and a run of host slabs of `slab` bytes each, at a byte offset into the run, split where a copy crosses from one slab into the next.
struct HostSpan {
    backend::Backend& b;
    std::vector<backend::BufferPtr>& slabs;
    size_t slab;
    bool to_host;
    size_t at = 0;
    void copy(backend::Buffer& device, size_t off, size_t bytes) {
        while (bytes) {
            backend::Buffer& host = *slabs[at / slab];
            const size_t in = at % slab, n = std::min(bytes, slab - in);
            if (to_host) b.copy(host, in, device, off, n);
            else b.copy(device, off, host, in, n);
            at += n;
            off += n;
            bytes -= n;
        }
    }
    // `n` blocks from `ids` on: each layer's K, V and auxiliary blocks, with a run of consecutive blocks in one copy.
    void blocks(backend::BlockKVStorage& st, const int32_t* ids, size_t n) {
        block_runs(st, ids, n, [&](size_t l, unsigned side, size_t off, size_t bytes) {
            copy(*(side == 0 ? st.k_buffer(l) : side == 1 ? st.v_buffer(l) : st.auxiliary_buffer(l)), off, bytes);
        });
    }
    // One slot of every layer of `states`.
    void slot(backend::StateStorage& states, size_t slot) {
        state_runs(states, slot, [&](backend::Buffer& buffer, size_t off, size_t bytes) { copy(buffer, off, bytes); });
    }
    // Transfer the host IDs before enqueueing device copies; host read/write can retire earlier device work.
    void tokens(uint32_t* data, size_t count, size_t pos) {
        size_t bytes = backend::size_mul(count, sizeof(uint32_t)), off = 0;
        while (bytes) {
            backend::Buffer& host = *slabs[pos / slab];
            const size_t in = pos % slab, n = std::min(bytes, slab - in);
            if (to_host) b.write(host, in, (const uint8_t*)data + off, n);
            else b.read(host, in, (uint8_t*)data + off, n);
            pos += n;
            off += n;
            bytes -= n;
        }
    }
};

// The idle slabs to free on each device before a copy that needs `need` slabs on each, the devices holding `idle` idle and `alive` slabs in all.
// None where the idle slabs cover every need, else those past a device's need, latest devices first, until the slabs alive after the shortfall is allocated are within `limit`; nullopt where freeing all of them is not enough.
inline std::optional<std::vector<size_t>> slabs_to_free(const std::vector<size_t>& need, const std::vector<size_t>& idle, size_t alive, size_t limit) {
    std::vector<size_t> drop(need.size(), 0);
    size_t short_by = 0;
    for (size_t i = 0; i < need.size(); ++i) short_by += need[i] > idle[i] ? need[i] - idle[i] : 0;
    if (!short_by) return drop;
    for (size_t i = need.size(); i-- > 0 && alive + short_by > limit;) {
        const size_t spare = idle[i] > need[i] ? idle[i] - need[i] : 0;
        drop[i] = std::min(spare, alive + short_by - limit);
        alive -= drop[i];
    }
    if (alive + short_by > limit) return std::nullopt;
    return drop;
}

// What a copy needing `need` slabs on each device takes, as alloc_host takes it: the idle slabs it frees first (slabs_to_free), how many those are, and the slabs it then allocates anew; nullopt past the limit.
struct CopySlabs {
    std::vector<size_t> drop;
    size_t dropped = 0, fresh = 0;
};
inline std::optional<CopySlabs> copy_slabs(const std::vector<size_t>& need, const std::vector<size_t>& idle, size_t alive, size_t limit) {
    const auto drop = slabs_to_free(need, idle, alive, limit);
    if (!drop) return std::nullopt;
    CopySlabs c{*drop};
    for (size_t i = 0; i < need.size(); ++i) {
        c.dropped += c.drop[i];
        const size_t left = idle[i] - c.drop[i];
        c.fresh += need[i] > left ? need[i] - left : 0;
    }
    return c;
}

// Whether `bytes` more of host memory leave the host `reserve` out of `free`, what it has free now; an unknown figure refuses nothing.
inline bool host_room(std::optional<size_t> free, size_t bytes, size_t reserve) {
    return !free || (bytes <= *free && *free - bytes >= reserve);
}
} // namespace detail

// Whether some KV or state storage sits on a device other than the CPU, so a copy to host memory frees memory a copy on the host would not.
inline bool Model::caches_on_devices() const {
    for (const auto& d : devices_)
        if ((d->storage || d->states) && !d->b->is_cpu()) return true;
    return false;
}

// The host memory the slabs for copies take, idle or holding a copy.
inline size_t Model::host_allocated() const {
    size_t n = 0;
    for (size_t a : host_allocated_) n += a;
    return n * kHostSlab;
}

// The bytes a copy of `length` tokens takes on each device: its blocks from `first` on where `blocks`, and with `state` its state and the drafter's carried row.
inline std::vector<size_t> Model::host_copy_bytes(size_t length, bool blocks, size_t first, bool state) const {
    std::vector<size_t> bytes(devices_.size(), 0);
    for (size_t i = 0; i < devices_.size(); ++i) {
        const Device& d = *devices_[i];
        if (const auto* st = blocks ? dynamic_cast<const backend::BlockKVStorage*>(d.storage.get()) : nullptr)
            bytes[i] = (length - first) / st->block_tokens() * st->total_block_bytes();
        if (state && d.states)
            for (size_t l = 0; l < d.states->layers(); ++l) bytes[i] = backend::size_add(bytes[i], d.states->slot_bytes(l));
        if (state && d.carry && state_layers_) bytes[i] += plan_.residual * sizeof(float);
        if (state && token_width_ && i == stages_[token_stage_].device) bytes[i] = backend::size_add(bytes[i], backend::size_mul(token_width_, sizeof(uint32_t)));
    }
    return bytes;
}

// The slabs a whole copy of `length` tokens would allocate anew now: each device's need past the idle slabs it holds, which another device's idle slabs do not serve.
inline size_t Model::host_fresh(size_t length) const {
    const std::vector<size_t> bytes = host_copy_bytes(length, true, 0, true);
    size_t fresh = 0;
    for (size_t i = 0; i < devices_.size(); ++i) {
        const size_t need = backend::blocks_for(bytes[i], kHostSlab), idle = i < host_slabs_.size() ? host_slabs_[i].size() : 0;
        fresh += need > idle ? need - idle : 0;
    }
    return fresh;
}

// Whether a whole copy of `length` tokens would get its slabs now, as alloc_host takes them under `limit`, with `freed` idle slabs more on each device.
// Fresh slabs must leave the host's reserve, read as if the idle slabs freed for the limit were given back.
inline Model::HostFit Model::host_copy_fits(size_t length, size_t limit, const std::vector<size_t>& freed) const {
    const std::vector<size_t> bytes = host_copy_bytes(length, true, 0, true);
    std::vector<size_t> need(devices_.size(), 0), idle(devices_.size(), 0);
    size_t alive = host_allocated() / kHostSlab;
    for (size_t i = 0; i < devices_.size(); ++i) {
        need[i] = backend::blocks_for(bytes[i], kHostSlab);
        idle[i] = (i < host_slabs_.size() ? host_slabs_[i].size() : 0) + (i < freed.size() ? freed[i] : 0);
    }
    const auto take = detail::copy_slabs(need, idle, alive, limit / kHostSlab);
    if (!take) return HostFit::limit;
    if (!take->fresh) return HostFit::fits;
    const std::optional<size_t> free = host_available();
    return detail::host_room(free ? std::optional<size_t>(*free + take->dropped * kHostSlab) : std::nullopt, take->fresh * kHostSlab, host_reserve_) ? HostFit::fits : HostFit::memory;
}

// Slabs for a copy of `length` tokens (HostHistory) as save_host takes them, holding nothing yet: within `limit` and the host reserve, idle slabs reused and other devices' freed first.
// From `first` on and without `state` they cover the blocks of a range alone, `out` is whole or released on a throw, and with `deferred` or `defer` the allocation and frees are left to each device's stage thread (copy_out).
inline void Model::alloc_host(size_t length, HostHistory& out, size_t limit, bool blocks, size_t first, bool state, std::vector<size_t>* deferred, const Defer* defer) {
    release_host(out);
    // Each sized on its own, so one that fails to grow leaves the other's check to grow it next time.
    if (host_allocated_.size() != devices_.size()) host_allocated_.resize(devices_.size(), 0);
    if (host_slabs_.size() != devices_.size()) host_slabs_.resize(devices_.size());
    const std::vector<size_t> bytes = host_copy_bytes(length, blocks, first, state);
    std::vector<size_t> need(devices_.size(), 0), idle(devices_.size(), 0);
    for (size_t i = 0; i < devices_.size(); ++i) {
        need[i] = backend::blocks_for(bytes[i], kHostSlab);
        idle[i] = host_slabs_[i].size();
    }
    // Idle slabs are retired (release_host), so freeing them waits for nothing.
    const auto take = detail::copy_slabs(need, idle, host_allocated() / kHostSlab, limit / kHostSlab);
    if (!take) throw std::runtime_error("inference: no host memory for the copy within its limit");
    std::vector<std::vector<backend::BufferPtr>> dropped(devices_.size());
    for (size_t i = 0; i < devices_.size(); ++i)
        for (size_t k = 0; k < take->drop[i]; ++k) {
            dropped[i].push_back(std::move(host_slabs_[i].back()));
            host_slabs_[i].pop_back();
            --host_allocated_[i];
        }
    if (defer) free_slabs(std::move(dropped), *defer);
    dropped.clear();
    const size_t fresh = take->fresh;
    // The slabs it allocates must leave the host the reserve the fit kept on it, read once the idle slabs above are freed.
    if (fresh && !host_admits(fresh * kHostSlab))
        throw std::runtime_error("inference: no host memory for the copy beside the reserve the host keeps");
    HostHistory h;
    h.owner = this;
    h.length = length;
    h.blocks = blocks;
    h.first = first;
    h.state = state;
    h.device_bytes = bytes;
    h.slabs.resize(devices_.size());
    h.tickets.assign(devices_.size(), 0);
    if (deferred) deferred->assign(devices_.size(), 0);
    try {
        for (size_t i = 0; i < devices_.size(); ++i) {
            if (!bytes[i]) continue;
            Device& d = *devices_[i];
            std::vector<backend::BufferPtr>& slabs = h.slabs[i];
            while (slabs.size() + (deferred ? (*deferred)[i] : 0) < need[i]) {
                if (host_slabs_[i].empty()) {
                    host_slabs_[i].reserve(host_allocated_[i] + 1);
                    if (deferred) ++(*deferred)[i];
                    else slabs.push_back(d.b->alloc(kHostSlab, backend::Memory::host_visible));
                    ++host_allocated_[i];
                } else {
                    slabs.push_back(std::move(host_slabs_[i].back()));
                    host_slabs_[i].pop_back();
                }
                h.held += kHostSlab;
            }
            h.bytes += bytes[i];
        }
    } catch (...) {
        if (deferred)
            for (size_t i = 0; i < devices_.size(); ++i) host_allocated_[i] -= (*deferred)[i];
        release_host(h);
        throw;
    }
    out = std::move(h);
}

// Slabs freed on their devices' stages' threads, which may be recording on those devices.
inline void Model::free_slabs(std::vector<std::vector<backend::BufferPtr>> slabs, const Defer& defer) {
    for (size_t i = 0; i < slabs.size(); ++i) {
        if (slabs[i].empty()) continue;
        auto held = std::make_shared<std::vector<backend::BufferPtr>>(std::move(slabs[i]));
        defer(StageItem{stage_of_device(i), [held] { held->clear(); }});
    }
}

// The device parts of a copy to host memory (save_host, save_host_blocks), one stage item a device: each allocates its fresh slabs, records its copies and submits on its stage's thread, and reports to h's pending record.
// What a part reads is taken now, the block ids, the token tail and the slot: a block or slot the caller releases after this is reused only by a pass queued after the part on the same device, so the part reads it first.
inline void Model::copy_out(HostHistory& h, const std::vector<size_t>& fresh, std::vector<std::vector<int32_t>> ids, std::vector<uint32_t> tail, size_t tail_at, size_t slot, bool state,
                            const Defer& defer) {
    auto p = std::make_shared<HostPending>();
    p->wanted = fresh;
    p->fresh.resize(devices_.size());
    p->tickets.assign(devices_.size(), 0);
    std::vector<StageItem> items;
    for (size_t i = 0; i < devices_.size(); ++i) {
        if (h.slabs[i].empty() && !fresh[i]) continue;
        const bool tokens = state && token_width_ && i == stages_[token_stage_].device;
        items.push_back(StageItem{stage_of_device(i), [this, p, i, reused = h.slabs[i], n = fresh[i], ids = std::move(ids[i]), tail = tokens ? tail : std::vector<uint32_t>{}, tail_at, slot,
                                                       state]() mutable {
            Device& d = *devices_[i];
            std::vector<backend::BufferPtr> made;
            backend::Ticket t = 0;
            std::string error;
            try {
                std::vector<backend::BufferPtr> slabs = reused;
                for (size_t k = 0; k < n; ++k) {
                    made.push_back(d.b->alloc(kHostSlab, backend::Memory::host_visible));
                    slabs.push_back(made.back());
                }
                detail::HostSpan span{*d.b, slabs, kHostSlab, true};
                if (!tail.empty()) span.tokens(tail.data(), tail.size(), tail_at);
                if (!ids.empty()) span.blocks(*dynamic_cast<backend::BlockKVStorage*>(d.storage.get()), ids.data(), ids.size());
                if (state && d.states) span.slot(*d.states, slot);
                if (state && d.carry && state_layers_) span.copy(*d.carry, slot * plan_.residual * sizeof(float), plan_.residual * sizeof(float));
                if (!ids.empty() || !tail.empty() || state) t = d.b->submit();
            } catch (const std::exception& e) {
                error = e.what();
                // Nothing else records on the device on its stage's thread, so what the part recorded retires here before its slabs can go.
                try { d.b->sync(); } catch (...) {}
            }
            std::lock_guard<std::mutex> lk(p->m);
            p->fresh[i] = std::move(made);
            p->tickets[i] = t;
            if (!error.empty() && p->error.empty()) p->error = error;
            if (!--p->left) p->cv.notify_all();
        }});
    }
    p->left = items.size();
    h.pending = p;
    for (StageItem& it : items) defer(std::move(it));
}

// A copy whose parts are pending made whole once they have run: their fresh slabs joined to its own and their tickets and error taken, true; false where `wait` is not given and a part has yet to run.
inline bool Model::host_settle(const HostHistory& h, bool wait) noexcept {
    const std::shared_ptr<HostPending> p = h.pending;
    if (!p || h.owner != this) return true;
    {
        std::unique_lock<std::mutex> lk(p->m);
        if (p->left && !wait) return false;
        if (p->left) {
            if (stage_wait_) stage_wait_(true);
            p->cv.wait(lk, [&] { return p->left == 0; });
            if (stage_wait_) stage_wait_(false);
        }
    }
    for (size_t i = 0; i < devices_.size() && i < h.slabs.size(); ++i) {
        // A part whose allocation failed made fewer than the pool counted.
        host_allocated_[i] -= p->wanted[i] - p->fresh[i].size();
        for (auto& b : p->fresh[i]) h.slabs[i].push_back(std::move(b));
        if (p->tickets[i]) h.tickets[i] = p->tickets[i];
    }
    if (!p->error.empty()) h.error = p->error;
    h.pending.reset();
    return true;
}

// A copy read back into the devices whole: its parts run, and none of them failed.
inline void Model::settled_whole(const HostHistory& h) {
    host_settle(h, true);
    if (!h.error.empty()) throw std::runtime_error("inference: a copy to host memory failed (" + h.error + ")");
}

// The history's first `length` tokens (whole blocks of every storage) copied to host memory, with its checkpoint's slot at `length` on a model that keeps a state; without `blocks` only the state is copied.
// Slabs stay within `limit` and the host's reserve and `out` is whole or released on failure; with `defer` the device parts go to the stages' threads (copy_out) and `out` is whole once host_settle has taken them.
inline void Model::save_host(Sequence& s, size_t length, HostHistory& out, size_t limit, bool blocks, const Defer& defer) {
    settle(s, "a copy to host memory");
    if (s.deferred_ && !s.deferred_->error.empty()) throw std::runtime_error("inference: a copy to host memory of a history whose restore failed (" + s.deferred_->error + ")");
    release_host(out);
    if (s.mark_.held()) throw std::logic_error("inference: a copy to host memory of a marked sequence");
    if (length > s.length()) throw std::logic_error("inference: a copy to host memory past the history");
    if (state_layers_ && (!s.kept_.held() || s.kept_.pos() != length))
        throw std::logic_error("inference: a copy to host memory of a model whose layers keep a recurrent state takes its checkpoint");
    if (!blocks && !state_layers_) throw std::logic_error("inference: a state alone copied to host memory on a model whose layers keep none");
    for (const Device* d : storages_) {
        if (!dynamic_cast<const backend::BlockKVStorage*>(d->storage.get())) throw std::runtime_error("inference: a KV storage that cannot be copied to host memory");
        if (length % d->b->kv_layout().block_tokens) throw std::logic_error("inference: a copy to host memory takes whole blocks of the history");
    }
    HostHistory h;
    if (defer) {
        std::vector<size_t> fresh;
        alloc_host(length, h, limit, blocks, 0, true, &fresh, &defer);
        std::vector<std::vector<int32_t>> ids(devices_.size());
        std::vector<uint32_t> tail;
        size_t tail_at = 0;
        for (size_t i = 0; i < devices_.size(); ++i) {
            Device& d = *devices_[i];
            if (token_width_ && i == stages_[token_stage_].device) {
                tail.assign(token_tails_.begin() + (std::ptrdiff_t)(s.kept_.slot() * token_width_), token_tails_.begin() + (std::ptrdiff_t)((s.kept_.slot() + 1) * token_width_));
                tail_at = h.device_bytes[i] - token_width_ * sizeof(uint32_t);
            }
            if (const auto* st = blocks ? dynamic_cast<const backend::BlockKVStorage*>(d.storage.get()) : nullptr) {
                const int32_t* b = s.kv_[(size_t)d.storage_index].view(nullptr).blocks;
                ids[i].assign(b, b + length / st->block_tokens());
            }
        }
        copy_out(h, fresh, std::move(ids), std::move(tail), tail_at, state_layers_ ? s.kept_.slot() : 0, true, defer);
        for (const Sequence::Media& m : carried(s, length)) h.media.push_back(m.run);
        out = std::move(h);
        return;
    }
    alloc_host(length, h, limit, blocks);
    try {
        for (size_t i = 0; i < devices_.size(); ++i) {
            if (h.slabs[i].empty()) continue;
            Device& d = *devices_[i];
            detail::HostSpan span{*d.b, h.slabs[i], kHostSlab, true};
            if (token_width_ && i == stages_[token_stage_].device)
                span.tokens(token_tails_.data() + s.kept_.slot() * token_width_, token_width_, h.device_bytes[i] - token_width_ * sizeof(uint32_t));
            auto* st = blocks ? dynamic_cast<backend::BlockKVStorage*>(d.storage.get()) : nullptr;
            if (st) span.blocks(*st, s.kv_[(size_t)d.storage_index].view(nullptr).blocks, length / st->block_tokens());
            if (d.states) span.slot(*d.states, s.kept_.slot());
            // An embedded drafter's carried row goes with the checkpoint it sits beside, so the history promoted back drafts as one never evicted.
            if (d.carry && state_layers_) span.copy(*d.carry, s.kept_.slot() * plan_.residual * sizeof(float), plan_.residual * sizeof(float));
            h.tickets[i] = d.b->submit();
        }
    } catch (...) {
        // Copies may be enqueued past the tickets h holds, so every device retires them before the slabs go back.
        for (auto& d : devices_) d->b->sync();
        release_host(h);
        throw;
    }
    for (const Sequence::Media& m : carried(s, length)) h.media.push_back(m.run);
    out = std::move(h);
}

// The blocks of the history's tokens from `first` to `length` copied to host memory alone, with no state: what a turn added to a history whose earlier blocks are kept elsewhere (docs/DISK-TIER.md, Entries written as what changed).
// Enqueued and limited as save_host's copies are; `out` is whole or, on a throw, released.
inline void Model::save_host_blocks(Sequence& s, size_t first, size_t length, HostHistory& out, size_t limit, const Defer& defer) {
    settle(s, "a copy to host memory");
    if (s.deferred_ && !s.deferred_->error.empty()) throw std::runtime_error("inference: a copy to host memory of a history whose restore failed (" + s.deferred_->error + ")");
    release_host(out);
    if (s.mark_.held()) throw std::logic_error("inference: a copy to host memory of a marked sequence");
    if (first >= length || length > s.length()) throw std::logic_error("inference: a copy to host memory of blocks outside the history");
    for (const Device* d : storages_) {
        if (!dynamic_cast<const backend::BlockKVStorage*>(d->storage.get())) throw std::runtime_error("inference: a KV storage that cannot be copied to host memory");
        if (first % d->b->kv_layout().block_tokens || length % d->b->kv_layout().block_tokens) throw std::logic_error("inference: a copy to host memory takes whole blocks of the history");
    }
    HostHistory h;
    if (defer) {
        std::vector<size_t> fresh;
        alloc_host(length, h, limit, true, first, false, &fresh, &defer);
        std::vector<std::vector<int32_t>> ids(devices_.size());
        for (size_t i = 0; i < devices_.size(); ++i) {
            Device& d = *devices_[i];
            if (const auto* st = dynamic_cast<const backend::BlockKVStorage*>(d.storage.get())) {
                const int32_t* b = s.kv_[(size_t)d.storage_index].view(nullptr).blocks;
                ids[i].assign(b + first / st->block_tokens(), b + length / st->block_tokens());
            }
        }
        copy_out(h, fresh, std::move(ids), {}, 0, 0, false, defer);
        out = std::move(h);
        return;
    }
    alloc_host(length, h, limit, true, first, false);
    try {
        for (size_t i = 0; i < devices_.size(); ++i) {
            if (h.slabs[i].empty()) continue;
            Device& d = *devices_[i];
            detail::HostSpan span{*d.b, h.slabs[i], kHostSlab, true};
            auto* st = dynamic_cast<backend::BlockKVStorage*>(d.storage.get());
            span.blocks(*st, s.kv_[(size_t)d.storage_index].view(nullptr).blocks + first / st->block_tokens(), (length - first) / st->block_tokens());
            h.tickets[i] = d.b->submit();
        }
    } catch (...) {
        for (auto& d : devices_) d->b->sync();
        release_host(h);
        throw;
    }
    for (const Sequence::Media& m : carried(s, length)) h.media.push_back(m.run);
    out = std::move(h);
}

// Where tokens `first` to `length` lie in host history `h`: per device, each layer's K, V and auxiliary blocks, in the order a copy of that range holds them end to end.
inline std::vector<HostRange> Model::host_ranges(const HostHistory& h, size_t first, size_t length) const {
    if (h.owner != this || !h.blocks || first < h.first || first >= length || length > h.length) throw std::logic_error("inference: a range outside the host history");
    std::vector<HostRange> out;
    for (size_t i = 0; i < devices_.size(); ++i) {
        const auto* st = dynamic_cast<const backend::BlockKVStorage*>(devices_[i]->storage.get());
        if (!st) continue;
        const size_t bt = st->block_tokens(), held = (h.length - h.first) / bt, skip = (first - h.first) / bt, n = (length - first) / bt;
        if (first % bt || length % bt) throw std::logic_error("inference: a range of a host history takes whole blocks");
        size_t at = 0;
        for (size_t l = 0; l < st->layers(); ++l)
            for (const size_t block : {st->k_block_bytes(), st->v_block_bytes(), st->auxiliary_block_bytes(l)}) {
                if (!block) continue;
                out.push_back(HostRange{i, at + skip * block, n * block});
                at += held * block;
            }
    }
    return out;
}

// Where the state lies in host history `h`, which holds one: per device, the bytes after its blocks.
inline std::vector<HostRange> Model::host_state_ranges(const HostHistory& h) const {
    if (h.owner != this || !h.state) throw std::logic_error("inference: a host history without a state");
    std::vector<HostRange> out;
    for (size_t i = 0; i < devices_.size(); ++i) {
        size_t blocks = 0;
        if (const auto* st = h.blocks ? dynamic_cast<const backend::BlockKVStorage*>(devices_[i]->storage.get()) : nullptr)
            blocks = (h.length - h.first) / st->block_tokens() * st->total_block_bytes();
        if (h.device_bytes[i] > blocks) out.push_back(HostRange{i, blocks, h.device_bytes[i] - blocks});
    }
    return out;
}

// Copies out of `h`'s slabs a failure stopped part way retire before they are released: devices before `submitted` wait on h's tickets, those up to `recorded` submit and wait.
// A device the copy never touched is left alone, since another thread may be recording on it.
inline void Model::drain_recorded(const HostHistory& h, size_t submitted, size_t recorded) noexcept {
    for (size_t i = 0; i < devices_.size() && i < recorded; ++i) {
        if (h.slabs[i].empty()) continue;
        backend::Backend& b = *devices_[i]->b;
        if (i < submitted) {
            b.wait(h.tickets[i]);
        } else {
            try {
                b.wait(b.submit());
            } catch (...) {
                // This sync assumes no other thread records on the device, which the caller's hold of it gives (the server's quieted()).
                b.sync();
            }
        }
    }
}

// Whether every device's ticket of `tickets` (a host history's, HostHistory::tickets) has retired, asked through Backend::done, which a device backend answers without waiting.
inline bool Model::copies_retired(const std::vector<backend::Ticket>& tickets) const noexcept {
    for (size_t i = 0; i < devices_.size() && i < tickets.size(); ++i)
        if (tickets[i] && !devices_[i]->b->done(tickets[i])) return false;
    return true;
}

// Whether the parts a pending record waits for have run and every ticket they took has retired.
inline bool Model::copies_retired(const HostPending& p) const noexcept {
    std::lock_guard<std::mutex> lk(const_cast<HostPending&>(p).m);
    return !p.left && copies_retired(p.tickets);
}

// The token tail of `h`'s state written into checkpoint slot `slot`, read from its slabs' host memory once its copies into them have retired.
inline void Model::read_tail(const HostHistory& h, size_t slot) {
    if (!token_width_) return;
    const size_t i = stages_[token_stage_].device;
    size_t at = h.device_bytes[i] - token_width_ * sizeof(uint32_t), off = 0, bytes = token_width_ * sizeof(uint32_t);
    auto* out = reinterpret_cast<uint8_t*>(token_tails_.data() + slot * token_width_);
    while (bytes) {
        const size_t in = at % kHostSlab, n = std::min(bytes, kHostSlab - in);
        std::memcpy(out + off, static_cast<const uint8_t*>(h.slabs[i][at / kHostSlab]->host_ptr()) + in, n);
        at += n;
        off += n;
        bytes -= n;
    }
}

// The device parts of a copy from host memory into `s` (restore_host, fork with a state), one stage item a device: each records its copies and submits on its stage's thread, behind which the history's passes are queued.
// Each part's ticket goes to both `h`, whose slabs it reads, and `s`, whose next settle folds it; a part that fails throws after it has reported, so its stage's thread reports the restore (StageItem::restore).
inline void Model::copy_in(HostHistory& h, Sequence& s, std::vector<std::vector<int32_t>> ids, size_t slot, const Defer& defer) {
    auto p = std::make_shared<HostPending>();
    p->wanted.assign(devices_.size(), 0);
    p->fresh.resize(devices_.size());
    p->tickets.assign(devices_.size(), 0);
    auto q = std::make_shared<Sequence::Deferred>();
    q->last.assign(devices_.size(), 0);
    q->restore = true;
    std::vector<StageItem> items;
    for (size_t i = 0; i < devices_.size(); ++i) {
        Device& d = *devices_[i];
        const bool state = h.state && (d.states || (d.carry && state_layers_));
        if (ids[i].empty() && !state) continue;
        items.push_back(StageItem{stage_of_device(i), [this, p, q, i, slabs = h.slabs[i], ids = std::move(ids[i]), slot]() mutable {
            Device& d = *devices_[i];
            backend::Ticket t = 0;
            std::string error;
            try {
                detail::HostSpan span{*d.b, slabs, kHostSlab, false};
                auto* st = dynamic_cast<backend::BlockKVStorage*>(d.storage.get());
                // The storage grows to the blocks taken here, on the device's thread, ahead of the copies into them.
                if (!ids.empty()) st->ensure((size_t)*std::max_element(ids.begin(), ids.end()));
                if (!ids.empty()) span.blocks(*st, ids.data(), ids.size());
                if (d.states) span.slot(*d.states, slot);
                if (d.carry && state_layers_) span.copy(*d.carry, slot * plan_.residual * sizeof(float), plan_.residual * sizeof(float));
                t = d.b->submit();
            } catch (const std::exception& e) {
                error = e.what();
                // Nothing else records on the device on its stage's thread, so what the part recorded retires here before the slabs can go.
                try { d.b->sync(); } catch (...) {}
            }
            // The host copy read from stays whole, so only the history copied into takes the error.
            {
                std::lock_guard<std::mutex> lk(p->m);
                p->tickets[i] = t;
                if (!--p->left) p->cv.notify_all();
            }
            {
                std::lock_guard<std::mutex> lk(q->m);
                q->last[i] = t;
                if (!error.empty() && q->error.empty()) q->error = error;
                if (!--q->left) q->cv.notify_all();
            }
            if (!error.empty()) throw std::runtime_error(error);
        }, q});
    }
    p->left = q->left = items.size();
    h.pending = p;
    s.deferred_ = q;
    size_t sent = 0;
    try {
        for (; sent < items.size(); ++sent) defer(std::move(items[sent]));
    } catch (...) {
        // A part run in the call that threw reported itself; those after it are never handed over, so nothing waits for them.
        const size_t unsent = items.size() - sent - 1;
        {
            std::lock_guard<std::mutex> lk(p->m);
            p->left -= unsent;
            p->cv.notify_all();
        }
        {
            std::lock_guard<std::mutex> lk(q->m);
            q->left -= unsent;
            q->cv.notify_all();
        }
        throw;
    }
}

// A fresh history holding what `h` copied, copied back into blocks and, on a model that keeps a state, a checkpoint slot at h.length, so it continues with the bits of the history it was copied from.
// The copies are enqueued ahead of any pass of the history; a failure leaves nothing held, and with `defer` the device copies go to the stages' threads (copy_in) with the history's passes queued behind them.
inline Sequence Model::restore_host(HostHistory& h, const Defer& defer) {
    if (h.owner != this) throw std::runtime_error("inference: a history copied to host memory by another model");
    settled_whole(h);
    if (h.slabs.size() != devices_.size() || !h.blocks || h.first || (state_layers_ && !h.state)) throw std::logic_error("inference: a host history of another layout");
    Sequence s = make_sequence();
    for (const MediaRun& run : h.media) s.media_.push_back(Sequence::Media{run, std::vector<backend::BufferPtr>(devices_.size())});
    size_t slot = 0;
    if (state_layers_) {
        slot = slots_.acquire_kept();
        s.kept_ = Checkpoint(slots_, slot, h.length);
        std::fill(s.from_.begin(), s.from_.end(), slot);
    }
    if (defer) {
        std::vector<std::vector<int32_t>> ids(devices_.size());
        for (size_t i = 0; i < devices_.size(); ++i) {
            Device& d = *devices_[i];
            auto* st = dynamic_cast<backend::BlockKVStorage*>(d.storage.get());
            if (d.storage && !st) throw std::runtime_error("inference: a KV storage that cannot be copied from host memory");
            if ((st || d.states) && h.slabs[i].empty()) throw std::logic_error("inference: a host history of another layout");
            if (!st) continue;
            // A tensor group's members copy into the blocks its first member, before them, took from the pool they share.
            const size_t n = h.length / st->block_tokens();
            KVSequence& kv = s.kv_[(size_t)d.storage_index];
            if (!d.member) kv.prepare(h.length);
            const int32_t* blocks = kv.view(nullptr).blocks;
            ids[i].assign(blocks, blocks + n);
            if (!d.member) kv.commit();
        }
        if (state_layers_) {
            wait_host(h);
            read_tail(h, slot);
        }
        copy_in(h, s, std::move(ids), slot, defer);
        std::fill(s.length_.begin(), s.length_.end(), h.length);
        return s;
    }
    size_t submitted = 0, recorded = 0;
    try {
        for (size_t i = 0; i < devices_.size(); ++i) {
            Device& d = *devices_[i];
            auto* st = dynamic_cast<backend::BlockKVStorage*>(d.storage.get());
            if (d.storage && !st) throw std::runtime_error("inference: a KV storage that cannot be copied from host memory");
            if ((st || d.states) && h.slabs[i].empty()) throw std::logic_error("inference: a host history of another layout");
            recorded = i + 1;
            detail::HostSpan span{*d.b, h.slabs[i], kHostSlab, false};
            if (token_width_ && i == stages_[token_stage_].device)
                span.tokens(token_tails_.data() + slot * token_width_, token_width_, h.device_bytes[i] - token_width_ * sizeof(uint32_t));
            if (st) {
                // A tensor group's members copy into the blocks its first member, before them, took from the pool they share.
                const size_t n = h.length / st->block_tokens();
                KVSequence& kv = s.kv_[(size_t)d.storage_index];
                if (!d.member) kv.prepare(h.length);
                const int32_t* blocks = kv.view(nullptr).blocks;
                if (n) st->ensure((size_t)*std::max_element(blocks, blocks + n));
                span.blocks(*st, blocks, n);
                if (!d.member) kv.commit();
            }
            if (d.states) span.slot(*d.states, slot);
            if (d.carry && state_layers_) span.copy(*d.carry, slot * plan_.residual * sizeof(float), plan_.residual * sizeof(float));
        }
        for (size_t i = 0; i < devices_.size(); ++i) {
            if (!h.slabs[i].empty()) h.tickets[i] = s.last_[i] = devices_[i]->b->submit();
            submitted = i + 1;
        }
    } catch (...) {
        // Copies out of h's slabs may be enqueued past its tickets, so they retire before h can be released.
        drain_recorded(h, submitted, recorded);
        throw;
    }
    std::fill(s.length_.begin(), s.length_.end(), h.length);
    return s;
}

// A fresh history for a restore of `length` tokens copied in piece by piece, its blocks and checkpoint slot taken as restore_host takes them, with per device where each byte of the image goes.
// The runs follow HostSpan's walk, so bytes copied run by run give restore_host's bits (docs/src/model-history.md); with `defer` each storage grows on its device's stage thread.
inline RestoreInto Model::restore_into(size_t length, const Defer& defer) {
    RestoreInto r;
    r.length = length;
    r.seq = make_sequence();
    r.runs.resize(devices_.size());
    for (size_t i = 0; i < devices_.size(); ++i) {
        r.backends.push_back(devices_[i]->b.get());
        r.stages.push_back(stage_of_device(i));
    }
    size_t slot = 0;
    if (state_layers_) {
        slot = slots_.acquire_kept();
        r.seq.kept_ = Checkpoint(slots_, slot, length);
        std::fill(r.seq.from_.begin(), r.seq.from_.end(), slot);
    }
    for (size_t i = 0; i < devices_.size(); ++i) {
        Device& d = *devices_[i];
        auto* st = dynamic_cast<backend::BlockKVStorage*>(d.storage.get());
        if (d.storage && !st) throw std::runtime_error("inference: a KV storage that cannot be copied from host memory");
        std::vector<RestoreInto::Run>& runs = r.runs[i];
        if (st) {
            if (length % st->block_tokens()) throw std::logic_error("inference: a restore takes whole blocks");
            // A tensor group's members copy into the blocks its first member, before them, took from the pool they share.
            const size_t n = length / st->block_tokens();
            KVSequence& kv = r.seq.kv_[(size_t)d.storage_index];
            if (!d.member) kv.prepare(length);
            const int32_t* blocks = kv.view(nullptr).blocks;
            if (n) {
                const size_t most = (size_t)*std::max_element(blocks, blocks + n);
                if (defer) defer(StageItem{stage_of_device(i), [st, most] { st->ensure(most); }});
                else st->ensure(most);
            }
            detail::block_runs(*st, blocks, n, [&](size_t l, unsigned side, size_t off, size_t bytes) {
                runs.push_back({nullptr, st, l, side, off, bytes});
            });
            if (!d.member) kv.commit();
        }
        if (d.states) detail::state_runs(*d.states, slot, [&](backend::Buffer& buffer, size_t off, size_t bytes) {
            runs.push_back({&buffer, nullptr, 0, 0, off, bytes});
        });
        if (d.carry && state_layers_) runs.push_back({d.carry.get(), nullptr, 0, false, slot * plan_.residual * sizeof(float), plan_.residual * sizeof(float)});
        if (token_width_ && i == stages_[token_stage_].device)
            runs.push_back({nullptr, nullptr, 0, 0, 0, token_width_ * sizeof(uint32_t), reinterpret_cast<uint8_t*>(token_tails_.data() + slot * token_width_)});
    }
    return r;
}

// The layout of a whole copy of `length` tokens with no memory behind it, which host_ranges and host_state_ranges read to say where an image's bytes lie (a streamed restore's files).
inline HostHistory Model::host_layout(size_t length) const {
    HostHistory h;
    h.owner = this;
    h.length = length;
    h.device_bytes = host_copy_bytes(length, true, 0, true);
    return h;
}

// A restore copied into whole: its stages without KV hold its length from here, and its passes go behind `last`, each device's ticket of its last copy into it.
inline void Model::restored(RestoreInto& r, const std::vector<backend::Ticket>& last) {
    for (size_t i = 0; i < devices_.size() && i < last.size(); ++i)
        if (last[i]) r.seq.last_[i] = last[i];
    std::fill(r.seq.length_.begin(), r.seq.length_.end(), r.length);
}

// A second history holding the first `length` tokens of `src`, whose blocks it shares as fork does, and the state at `length` that `state` holds (Model::save_host without blocks, docs/SPECULATIVE.md, section 2, Host tier).
// The state is copied into a checkpoint slot of its own, on the stages' threads with `defer` (copy_in), behind the passes that wrote what it shares; a throw leaves nothing held.
inline Sequence Model::fork(const Sequence& src, size_t length, HostHistory& state, const Defer& defer) {
    if (src.owner_ != this || state.owner != this) throw std::runtime_error("inference: sequence or host state of another model");
    settled_whole(state);
    if (src.flights_) throw std::logic_error("inference: a fork of a sequence in flight");
    if (src.mark_.held()) throw std::logic_error("inference: a fork of a marked sequence");
    if (!state_layers_ || state.blocks || state.length != length || state.slabs.size() != devices_.size())
        throw std::logic_error("inference: a fork with a state takes a state alone at its length, on a model whose layers keep one");
    if (length > src.length()) throw std::logic_error("KV cache: a fork takes whole blocks of the history");
    Sequence f;
    f.storage_of_ = src.storage_of_;
    f.length_.assign(stages_.size(), length);
    f.kv_.reserve(storages_.size());
    for (const KVSequence& kv : src.kv_) f.kv_.push_back(kv.fork(length));
    f.last_ = src.last_;
    f.owner_ = this;
    f.media_ = carried(src, length);
    const size_t slot = slots_.acquire_kept();
    f.kept_ = Checkpoint(slots_, slot, length);
    f.from_.assign(stages_.size(), slot);
    if (defer) {
        for (size_t i = 0; i < devices_.size(); ++i)
            if ((devices_[i]->states || devices_[i]->carry) && state.slabs[i].empty()) throw std::logic_error("inference: a host state of another layout");
        wait_host(state);
        read_tail(state, slot);
        copy_in(state, f, std::vector<std::vector<int32_t>>(devices_.size()), slot, defer);
        return f;
    }
    size_t submitted = 0, recorded = 0;
    try {
        for (size_t i = 0; i < devices_.size(); ++i) {
            Device& d = *devices_[i];
            if (!d.states && !d.carry) continue;
            if (state.slabs[i].empty()) throw std::logic_error("inference: a host state of another layout");
            recorded = i + 1;
            detail::HostSpan span{*d.b, state.slabs[i], kHostSlab, false};
            if (token_width_ && i == stages_[token_stage_].device)
                span.tokens(token_tails_.data() + slot * token_width_, token_width_, state.device_bytes[i] - token_width_ * sizeof(uint32_t));
            if (d.states) span.slot(*d.states, slot);
            // An embedded drafter's carried row comes back with the state it was saved beside, so the fork drafts as the history it was taken from did.
            if (d.carry) span.copy(*d.carry, slot * plan_.residual * sizeof(float), plan_.residual * sizeof(float));
            state.tickets[i] = f.last_[i] = d.b->submit();
            submitted = i + 1;
        }
    } catch (...) {
        // Copies out of the state's slabs may be enqueued past its tickets, so they retire before it can be released.
        drain_recorded(state, submitted, recorded);
        throw;
    }
    return f;
}

// What a history copied to host memory holds and how, beyond the model file and the build: each device's identity, effective dtype and runs, the cache types, the placement and the row classes up to the most a history holds.
// Two models of one file and one build with equal identities hold the same bytes for the same history, so a copy written by one is restored by the other (docs/DISK-TIER.md, The entry file).
inline std::string Model::host_identity() const {
    std::string s = token_width_ ? "llmx-host-history 2\n" : "llmx-host-history 1\n";
    if (token_width_) s += "token history " + std::to_string(token_stage_) + " " + std::to_string(token_width_) + " " + std::to_string(token_padding_) + "\n";
    const auto join = [](const std::vector<int>& v) {
        std::string t;
        for (int x : v) t += (t.empty() ? "" : ",") + std::to_string(x);
        return t;
    };
    s += "kv " + std::string(backend::kv_type_name(options_.kv_k)) + " " + backend::kv_type_name(options_.kv_v) + "\n";
    for (size_t i = 0; i < devices_.size(); ++i) {
        const Device& d = *devices_[i];
        const backend::Dtype dtype = options_.device_dtypes.empty() ? options_.dtype : options_.device_dtypes[i];
        s += "device " + std::to_string(i) + " " + d.b->identity() + "; dtype " + backend::dtype_name(dtype) + "; used " + std::to_string(d.used);
        if (const auto* st = dynamic_cast<const backend::BlockKVStorage*>(d.storage.get())) {
            s += "; kv " + std::to_string(st->layers()) + " layers, " + std::to_string(st->block_tokens()) + " tokens a block, " + std::to_string(st->k_block_bytes()) + "+" +
                 std::to_string(st->v_block_bytes()) + " bytes";
            bool auxiliary = false;
            for (size_t l = 0; l < st->layers(); ++l) auxiliary |= st->auxiliary_width(l) != 0;
            if (auxiliary) {
                s += "; auxiliary f32 widths";
                for (size_t l = 0; l < st->layers(); ++l) s += " " + std::to_string(st->auxiliary_width(l));
                s += "; pooled every";
                for (size_t l = 0; l < st->layers(); ++l) s += " " + std::to_string(st->auxiliary_pool(l));
            }
        }
        if (d.states) {
            const size_t core = d.states->shape().slot_floats() * sizeof(float);
            s += "; state " + std::to_string(d.states->layers()) + " layers, " + std::to_string(core) + " bytes";
            bool tails = false;
            for (size_t l = 0; l < d.states->layers(); ++l) tails |= d.states->slot_bytes(l) != core;
            if (tails) {
                s += "; state strides";
                for (size_t l = 0; l < d.states->layers(); ++l) s += " " + std::to_string(d.states->slot_bytes(l));
            }
        }
        if (d.carry && state_layers_) s += "; carry " + std::to_string(plan_.residual * sizeof(float)) + " bytes";
        s += "\n";
    }
    s += "placement mixer " + join(place_.mixer_device) + " ffn " + join(place_.ffn_device) + " embed " + std::to_string(place_.embed_device) + " output " +
         std::to_string(place_.output_device) + " stream " + std::to_string(place_.stream_from) + " width " + std::to_string(place_.width) + "\n";
    const size_t limit = std::min((size_t)context_length(), kv_tokens_total());
    std::vector<size_t> last;
    s += "classes";
    for (size_t e = 1; e <= limit; ++e) {
        std::vector<size_t> c = row_class(e);
        if (c == last) continue;
        s += " " + std::to_string(e) + ":";
        for (size_t k = 0; k < c.size(); ++k) s += (k ? "," : "") + std::to_string(c[k]);
        last.swap(c);
    }
    return s + "\n";
}

// The slabs `h` holds left for the next copy on their devices once every copy into or out of them has retired; the pools' capacity is reserved as slabs are allocated, so this allocates nothing.
inline void Model::release_host(HostHistory& h) noexcept {
    host_settle(h, true);
    if (h.owner == this)
        for (size_t i = 0; i < h.slabs.size() && i < devices_.size(); ++i) {
            if (h.slabs[i].empty()) continue;
            wait_ticket(i, h.tickets[i]);
            for (auto& b : h.slabs[i]) host_slabs_[i].push_back(std::move(b));
        }
    h = HostHistory{};
}

// The pool's one admission rule: whether `bytes` of fresh slabs leave the host the pool's reserve, read now; twice the reserve for `optional` growth, which nothing waits for.
// Every fresh slab passes it, a copy's, a read's and an idle step's alike (alloc_host), and it is a sampled guard: another process may take the memory between the reading and the slab.
inline bool Model::host_admits(size_t bytes, bool optional) const {
    return detail::host_room(host_available(), bytes, host_reserve_ * (optional ? kHostOptional : 1));
}

// Whether the host is below the pool's reserve while the pool holds idle slabs it could give back.
inline bool Model::host_short() const {
    const std::optional<size_t> free = host_available();
    if (!free || *free >= host_reserve_) return false;
    return std::any_of(host_slabs_.begin(), host_slabs_.end(), [](const std::vector<backend::BufferPtr>& idle) { return !idle.empty(); });
}

// Idle slabs given back to a host below the pool's reserve, as many as cover what it is short by; the bytes freed.
// Slabs a copy holds are not idle and stay, so every history in host memory and every read or write on one is as it was.
inline size_t Model::host_yield(const Defer& defer) {
    const std::optional<size_t> free = host_available();
    if (!free || *free >= host_reserve_) return 0;
    size_t idle = 0;
    for (const auto& slabs : host_slabs_) idle += slabs.size();
    const size_t give = std::min(idle, backend::blocks_for(host_reserve_ - *free, kHostSlab)) * kHostSlab;
    if (defer) trim_host(host_allocated() - give, defer);
    else trim_host(host_allocated() - give);
    return give;
}

// The same, each slab freed on its device's stage's thread (free_slabs).
inline void Model::trim_host(size_t limit, const Defer& defer) {
    std::vector<std::vector<backend::BufferPtr>> freed(host_slabs_.size());
    for (size_t i = host_slabs_.size(); i-- > 0;)
        while (!host_slabs_[i].empty() && host_allocated() > limit) {
            freed[i].push_back(std::move(host_slabs_[i].back()));
            host_slabs_[i].pop_back();
            --host_allocated_[i];
        }
    free_slabs(std::move(freed), defer);
}

// Idle slabs freed until the slabs alive are within `limit`: what a copy allocated beyond its caller's own limit gives back once it is released.
inline void Model::trim_host(size_t limit) noexcept {
    for (size_t i = host_slabs_.size(); i-- > 0;)
        while (!host_slabs_[i].empty() && host_allocated() > limit) {
            host_slabs_[i].pop_back();
            --host_allocated_[i];
        }
}

// Waits for the copies into and out of h's slabs, so the host can read its bytes, as the disk tier's writes do.
inline void Model::wait_host(const HostHistory& h) noexcept {
    host_settle(h, true);
    if (h.owner == this)
        for (size_t i = 0; i < h.slabs.size() && i < devices_.size(); ++i)
            if (!h.slabs[i].empty()) wait_ticket(i, h.tickets[i]);
}

// The caller waits for device `device`'s ticket `t`, told to on_stage_wait where the work has not yet retired, so a server counts every wait of its thread on a device.
inline void Model::wait_ticket(size_t device, backend::Ticket t) const noexcept {
    if (!t || devices_[device]->b->done(t)) return;
    if (stage_wait_) stage_wait_(true);
    devices_[device]->b->wait(t);
    if (stage_wait_) stage_wait_(false);
}

// A history whose retract's parts have all run takes their tickets and gives the mark's hold back now, without waiting, so another verify can take that mark; true where nothing is left pending.
// A restore's record stays, its failure reported by its stage (StageItem::restore), and so does a part's failure, which its request's end meets.
inline bool Model::release_settled(Sequence& s) noexcept {
    const std::shared_ptr<Sequence::Deferred> p = s.deferred_;
    if (!p) return true;
    if (p->restore) return false;
    {
        std::lock_guard<std::mutex> lk(p->m);
        if (p->left || !p->error.empty()) return false;
    }
    for (size_t d = 0; d < p->last.size() && d < s.last_.size(); ++d) s.last_[d] = std::max(s.last_[d], p->last[d]);
    p->hold.release();
    s.deferred_.reset();
    return true;
}

// A sequence out of flight whose passes have retired, before it gives blocks or slots back.
inline void Model::settle(Sequence& s, const char* what) {
    if (s.owner_ != this) throw std::runtime_error("inference: sequence of another model");
    if (s.flights_) throw std::logic_error(std::string("inference: ") + what + " of a sequence in flight");
    // Parts on the stages' threads run first; a failed part's request ends by its stage's report, so its error is not thrown here.
    if (const std::shared_ptr<Sequence::Deferred> p = std::move(s.deferred_)) {
        std::unique_lock<std::mutex> lk(p->m);
        if (p->left) {
            if (stage_wait_) stage_wait_(true);
            p->cv.wait(lk, [&] { return p->left == 0; });
            if (stage_wait_) stage_wait_(false);
        }
        for (size_t d = 0; d < p->last.size() && d < s.last_.size(); ++d) s.last_[d] = std::max(s.last_[d], p->last[d]);
        p->hold.release();
        // A restore that failed stays named on the history, so its stage's report finds every holder of it (Sequence::restoring).
        if (p->restore && !p->error.empty()) s.deferred_ = p;
    }
    // A sequence moved from has no tickets, nothing of it being left to wait on.
    for (size_t d = 0; d < devices_.size() && d < s.last_.size(); ++d)
        if (devices_[d]->used) wait_ticket(d, s.last_[d]);
}

// A history back to `length`, or on a model that keeps a state to its checkpoint at or below it, else 0, returning the length reached: every stage and storage at it, the blocks and a checkpoint beyond it returned.
// The state is then the checkpoint's, or zero, so the live slot goes back too, and the next pass takes one: a donor parked at its checkpoint holds none of the slots admission counts on.
inline size_t Model::rewind(Sequence& s, size_t length, const Defer* defer) noexcept {
    // At or past a mark the state is the mark's, and the history goes back to its position; below it the mark is not needed.
    if (s.mark_.held() && length >= s.mark_.pos) {
        const size_t to = s.mark_.pos;
        restore_mark(s);
        for (size_t& n : s.length_) n = std::min(n, to);
        for (auto& kv : s.kv_) kv.truncate(to);
        return to;
    }
    drop_mark(s);
    size_t to = length;
    if (state_layers_) {
        if (s.kept_.held() && s.kept_.pos() > length) s.kept_.release();
        to = s.kept_.held() ? s.kept_.pos() : 0;
        std::fill(s.from_.begin(), s.from_.end(), s.kept_.held() ? s.kept_.slot() : Sequence::kLive);
        s.state_.release();
    }
    for (size_t& n : s.length_) n = std::min(n, to);
    for (auto& kv : s.kv_) kv.truncate(to);
    drop_media(s, to, defer);
    return to;
}

// Where row 0 of saved item `item` of local state layer `layer` of mark buffer `buffer` sits in device d's saved buffer, in floats; its rows follow one another, each as wide as a member's share of the item on a tensor group.
inline size_t Model::saved_at(const Device& d, const LayerPlan& lp, size_t buffer, size_t layer, size_t item) const {
    size_t before = 0;
    for (size_t i = 0; i < item; ++i) before += shard::saved(lp.saved[i], width_).width;
    return ((buffer * (size_t)d.state_layers + layer) * d.saved_floats + before) * options_.mark_rows;
}

// After state layer l's mixer on device `dev`: each marked entry's rows of the inputs its state's update read, copied into its mark's buffer.
inline void Model::save(ExecContext& ctx, const Pass& p, size_t dev, int l) {
    const Device& d = *devices_[dev];
    const LayerPlan& lp = plan_.layers[(size_t)l];
    const ExecContext::Scratch& sc = ctx.scratch[dev];
    for (size_t e = 0; e < p.entries.size(); ++e) {
        const Sequence::Mark& m = p.entries[e].seq->mark_;
        if (!m.held()) continue;
        const size_t r0 = e ? p.runs[e - 1].end : 0, n = p.entries[e].n;
        for (size_t i = 0; i < lp.saved.size(); ++i) {
            const Saved v = shard::saved(lp.saved[i], width_);
            d.b->copy(*d.saved, saved_at(d, lp, m.hold.buffer(), (size_t)d.local_layer[(size_t)l], i) * sizeof(float), *sc.arena,
                      sc.offset[v.slot] + (v.plane * p.rows + r0) * v.width * sizeof(float), n * v.width * sizeof(float));
        }
    }
}

// After an embedded drafter's context rows on the head's device, or each member of its group: each marked entry's normed rows, the row each kept row would carry, copied into its mark's room.
inline void Model::save_h(ExecContext& ctx, const Pass& p) {
    const size_t E = plan_.residual;
    for (size_t o = (size_t)place_.output_device, end = o + width_; o < end; ++o) {
        const Device& d = *devices_[o];
        const backend::Slice hn = slot(ctx, o, plan_.draft_h);
        for (size_t e = 0; e < p.entries.size(); ++e) {
            const Sequence::Mark& m = p.entries[e].seq->mark_;
            if (!m.held()) continue;
            const size_t r0 = e ? p.runs[e - 1].end : 0, n = p.entries[e].n;
            d.b->copy(*d.saved_h, m.hold.buffer() * options_.mark_rows * E * sizeof(float), *hn.buffer, (hn.offset + r0 * E) * sizeof(float), n * E * sizeof(float));
        }
    }
}

// The state after the first `rows` rows of the pass past the sequence's mark, into its live slot: each state layer's update (Architecture::recur) run from the mark's state over its saved inputs, in one submission on each device.
// An embedded drafter's carried row is the last kept row's, copied from the mark's room into the live slot's in the same submission, and a failure drains every device and leaves the mark, which a retry reads again.
inline void Model::rerun(Sequence& s, size_t rows, const Defer& defer) {
    const Sequence::Mark& m = s.mark_;
    // What every device's part reads, taken now: a deferred part runs once the mark has gone to the history's Deferred.
    const Rerun a{rows, m.pos, m.from, s.state_.slot(), m.hold.buffer()};
    if (!defer) {
        try {
            ensure(ctx_, rows, 0, handoffs(1));
            for (size_t dev = 0; dev < devices_.size(); ++dev)
                if (reruns_on(dev)) {
                    rerun_on(dev, a);
                    s.last_[dev] = devices_[dev]->b->submit();
                }
        } catch (...) {
            // A failure inside a phase leaves its device recording without barriers; the mode is set before any barrier is recorded, so it is off even where recording that barrier fails.
            for (auto& d : devices_) try { d->b->unordered(false); } catch (...) {}
            retire();
            throw;
        }
    } else {
        // The context takes the most rows a mark saves, so no later retract grows it under a part still running.
        {
            std::lock_guard<std::mutex> lk(context_m_);
            ensure(ctx_, options_.mark_rows, 0, handoffs(1));
        }
        auto p = std::make_shared<Sequence::Deferred>();
        p->last.assign(devices_.size(), 0);
        std::vector<size_t> on;
        for (size_t dev = 0; dev < devices_.size(); ++dev)
            if (reruns_on(dev)) on.push_back(dev);
        p->left = on.size();
        s.deferred_ = p;
        for (size_t dev : on)
            defer({stage_of_device(dev), [this, dev, a, p] {
                Device& d = *devices_[dev];
                std::string error;
                try {
                    {
                        // Parts on different stages' threads share the context's scratch run list.
                        std::lock_guard<std::mutex> lk(context_m_);
                        rerun_on(dev, a);
                    }
                    const backend::Ticket t = d.b->submit();
                    std::lock_guard<std::mutex> lk(p->m);
                    p->last[dev] = t;
                } catch (const std::exception& e) {
                    try { d.b->unordered(false); d.b->sync(); } catch (...) {}
                    error = e.what();
                }
                {
                    std::lock_guard<std::mutex> lk(p->m);
                    if (!error.empty() && p->error.empty()) p->error = error;
                    --p->left;
                }
                p->cv.notify_all();
                if (!error.empty()) throw std::runtime_error(error);
            }});
    }
    if (token_width_) {
        const size_t src = a.from[token_stage_] == Sequence::kLive ? a.live : a.from[token_stage_];
        advance_tokens(token_tails_.data() + a.live * token_width_, token_tails_.data() + src * token_width_,
                       a.pos, marked_tokens_.data() + a.buffer * options_.mark_rows, rows);
    }
}

// Whether a retract's rerun records on device dev: one holding state layers, or the output device of a model with an embedded drafter, whose carried row it restores.
inline bool Model::reruns_on(size_t dev) const {
    const Device& d = *devices_[dev];
    return d.states || (plan_.drafter && dev - d.member == (size_t)place_.output_device);
}

// The stage whose passes record on device dev.
inline size_t Model::stage_of_device(size_t dev) const {
    const size_t home = dev - devices_[dev]->member;
    for (size_t st = 0; st < stages_.size(); ++st)
        if (stages_[st].device == home) return st;
    throw std::logic_error("inference: a device no stage records on");
}

// Device dev's part of a rerun, recorded and not submitted: each state layer's update over the kept rows' saved inputs, from the mark's state into the live slot, and the drafter's carried row.
inline void Model::rerun_on(size_t dev, const Rerun& a) {
    Device& d = *devices_[dev];
    // The update's calls take the mark's rows, which a plane of its saved inputs holds, and their views the rows kept.
    const backend::RowRun run{options_.mark_rows, 1};
    std::vector<size_t> offsets(plan_.slots.size());
    int phases = 0;
    for (const LayerPlan& lp : plan_.layers)
        if (lp.cache == Cache::state) phases = std::max(phases, lp.recur_phases);
    // Nothing a phase runs reads what another layer's writes: each layer reads its own saved inputs and slot, and writes its own room and slot.
    for (int phase = 0; d.states && phase < phases; ++phase) {
        d.b->unordered(true);
        for (size_t st = 0; st < stages_.size(); ++st) {
            if (stages_[st].device != dev - d.member) continue;
            const size_t src = a.from[st] == Sequence::kLive ? a.live : a.from[st];
            const backend::StateView view{d.states.get(), src, a.live, a.pos, a.rows};
            for (int l = stages_[st].first; l < stages_[st].end; ++l) {
                const LayerPlan& lp = plan_.layers[(size_t)l];
                if (lp.cache != Cache::state || phase >= lp.recur_phases) continue;
                const size_t layer = (size_t)d.local_layer[(size_t)l];
                size_t at = d.rerun_base + layer * options_.mark_rows * d.rerun_floats;
                std::fill(offsets.begin(), offsets.end(), 0);
                for (size_t slot : lp.recur_writes) {
                    offsets[slot] = at * sizeof(float);
                    at += options_.mark_rows * plan_.slots[slot];
                }
                for (size_t i = 0; i < lp.saved.size(); ++i)
                    if (!lp.saved[i].plane) offsets[lp.saved[i].slot] = saved_at(d, lp, a.buffer, layer, i) * sizeof(float);
                Step step = part(ctx_, dev, home_row(d.member, (size_t)l), lp.kind, 0, options_.mark_rows, {&run, 1});
                step.width = width_;
                step.arena = d.saved.get();
                step.offsets = offsets.data();
                step.x = {d.saved.get(), 0};
                step.states = &view;
                step.n_views = 1;
                step.state_layer = layer;
                step.phase = phase;
                arch_->recur(step);
            }
        }
        d.b->unordered(false);
    }
    if (plan_.drafter && dev - d.member == (size_t)place_.output_device) {
        const size_t E = plan_.residual;
        d.b->copy(*d.carry, a.live * E * sizeof(float), *d.saved_h, (a.buffer * options_.mark_rows + a.rows - 1) * E * sizeof(float), E * sizeof(float));
    }
}

// The state at the mark's position as the sequence's again: the mark's slot its live slot where the mark took it from there, else read from where it was then; the live slot a pass wrote returned and the mark gone.
inline void Model::restore_mark(Sequence& s) noexcept {
    Sequence::Mark& m = s.mark_;
    if (!m.held()) return;
    if (m.hold.owns()) {
        s.state_.adopt(slots_, m.hold.give_slot());
        std::fill(s.from_.begin(), s.from_.end(), Sequence::kLive);
    } else {
        s.state_.release();
        std::copy(m.from.begin(), m.from.end(), s.from_.begin());
    }
    m.hold.release();
    m.ran = false;
}

// The mark gone with its slot, the state staying where the passes after it left it.
inline void Model::drop_mark(Sequence& s) noexcept {
    s.mark_.hold.release();
    s.mark_.ran = false;
}

// Up to `k` drafts of the tokens after `last`, each history's last pick not yet fed, from an embedded drafter (docs/SPECULATIVE.md, section 7), in `out`, for every sequence asked: one submission on the head's device of a step a draft.
// Step m takes the sequences whose chains are longer than m, longest first, each reading the token drafted before it, and a chain ends at the first id outside the vocabulary or, on a tensor group, one its members disagree on.
inline void Model::draft(DraftAsk* asks, size_t n) {
    std::unique_ptr<Chain> c = draft_begin(asks, n);
    if (!c) return;
    try {
        draft_record(*c);
    } catch (...) {
        draft_end(*c);
        throw;
    }
    draft_end(*c);
}

// The drafter's room for chains of `seqs` sequences of `steps` steps: the ids of every step, the first member's logits of every step and the others' of one.
inline void Model::draft_buffers(size_t seqs, size_t steps) {
    const size_t o = (size_t)place_.output_device, W = width_, V = plan_.vocab;
    if (draft_id_rows_ >= seqs * (steps + 1) && draft_rows_ >= seqs * steps && draft_seqs_ >= seqs) return;
    const size_t id_rows = std::max(draft_id_rows_, seqs * (steps + 1)), rows = std::max(draft_rows_, seqs * steps), most = std::max(draft_seqs_, seqs);
    std::vector<backend::BufferPtr> ids(W), logits(W);
    for (size_t m = 0; m < W; ++m) {
        backend::Backend& b = *devices_[o + m]->b;
        ids[m] = b.alloc(backend::size_mul(id_rows, sizeof(float)), backend::Memory::host_visible);
        logits[m] = m ? b.alloc(backend::size_mul(most, backend::size_mul(V, sizeof(float))))
                      : b.alloc(backend::size_mul(rows, backend::size_mul(V, sizeof(float))), backend::Memory::host_visible);
    }
    draft_ids_ = std::move(ids);
    draft_logits_ = std::move(logits);
    draft_id_rows_ = id_rows;
    draft_rows_ = rows;
    draft_seqs_ = most;
}

// The fit counts both (Footprint::context_rows, Footprint::drafter_cache).
inline void Model::reserve_drafting() {
    if (!options_.mark_slots) return;
    const size_t rows = std::max(state_layers_ ? options_.mark_rows : 0, plan_.drafter ? options_.mark_slots : 0);
    if (!rows) return;
    std::lock_guard<std::mutex> context(context_m_);
    ensure(ctx_, rows, 0, handoffs(1));
    if (plan_.drafter && options_.mark_rows > 1) draft_buffers(options_.mark_slots, options_.mark_rows - 1);
}

// The chain's part on the thread that owns the histories: each history settled, the KV rows its drafts take, every step's rows, positions and views, and its sequences in flight; null where no ask drafts.
// With `behind` the chain is recorded on the head stage's thread behind the histories' rerun parts, and a caller without reserve_drafting's room grows the context and the drafter's buffers here.
inline std::unique_ptr<Model::Chain> Model::draft_begin(const DraftAsk* asks, size_t n, bool behind) {
    if (!plan_.drafter) throw std::logic_error("inference: a draft without an embedded drafter");
    const size_t S = stages_.size();
    auto c = std::make_unique<Chain>();
    c->asks.assign(asks, asks + n);
    for (size_t i = 0; i < n; ++i) {
        const DraftAsk& a = asks[i];
        if (!behind) settle(*a.seq, "a draft");
        a.out->clear();
        if (!a.k) continue;
        if (a.seq->flights_) throw std::logic_error("inference: a draft of a sequence in flight");
        const size_t L = history(*a.seq);
        if (!L || !a.seq->state_.held()) throw std::logic_error("inference: a draft of a history no pass has fed");
        if (a.k > plan_.context_length - std::min(plan_.context_length, L))
            throw std::runtime_error("inference: context length exceeded (" + std::to_string(plan_.context_length) + " tokens)");
        c->order.push_back(i);
        c->steps = std::max(c->steps, a.k);
    }
    if (c->order.empty()) return nullptr;
    std::stable_sort(c->order.begin(), c->order.end(), [&](size_t a, size_t b) { return asks[a].k > asks[b].k; });
    const size_t R = c->order.size(), o = (size_t)place_.output_device, W = width_, steps = c->steps;
    Device& d = *devices_[o];
    {
        std::lock_guard<std::mutex> context(context_m_);
        ensure(ctx_, R, 0, handoffs(1));
        draft_buffers(R, steps);
    }
    c->last.resize(R);
    c->carried.resize(R);
    c->kv.resize(R);
    std::vector<size_t> length(R);
    for (size_t p = 0; p < R; ++p) {
        Sequence& s = *asks[c->order[p]].seq;
        c->last[p] = asks[c->order[p]].last;
        length[p] = history(s);
        c->carried[p] = (uint32_t)(s.from_[S - 1] == Sequence::kLive ? s.state_.slot() : s.from_[S - 1]);
        c->kv[p] = &s.kv_[(size_t)d.storage_index];
    }
    try {
        for (; c->prepared < R; ++c->prepared) c->kv[c->prepared]->prepare(asks[c->order[c->prepared]].k);
        c->rows.resize(steps);
        c->pos.assign(backend::size_mul(steps, R), 0);
        c->views.resize(backend::size_mul(backend::size_mul(steps, W), R));
        for (size_t m = 0, rows = R; m < steps; ++m) {
            while (asks[c->order[rows - 1]].k <= m) --rows;
            c->rows[m] = rows;
            for (size_t p = 0; p < rows; ++p) {
                for (size_t g = 0; g < W; ++g) {
                    backend::KVView& v = c->views[(m * W + g) * R + p];
                    v = c->kv[p]->view(devices_[o + g]->storage.get());
                    v.length = length[p] + m;
                    v.nq = 1;
                    v.extent = 1;
                }
                // A drafted token is text, at its index less what the history's images saved.
                positions(*asks[c->order[p]].seq, length[p] + m, 1, false, &c->pos[m * R + p]);
            }
        }
    } catch (...) {
        for (size_t p = 0; p < c->prepared; ++p) c->kv[p]->abort();
        throw;
    }
    for (size_t p = 0; p < R; ++p) ++asks[c->order[p]].seq->flights_;
    return c;
}

// The chain on the head's devices, on the head stage's thread in a server: one submission a member of a step a draft, waited for, and the drafted ids read back.
// Step m takes the sequences whose chains are longer than m, longest first, each reading the token drafted before it.
inline void Model::draft_record(Chain& c) {
    const size_t R = c.order.size(), o = (size_t)place_.output_device, W = width_, V = plan_.vocab, steps = c.steps;
    // A server records the chain beside a deferred rerun's parts on other stages, which share the context.
    std::lock_guard<std::mutex> context(context_m_);
    try {
        std::vector<backend::Slice> x(W);
        for (size_t g = 0; g < W; ++g) {
            devices_[o + g]->b->write(*draft_ids_[g], 0, c.last.data(), R * sizeof(uint32_t));
            x[g] = slot(ctx_, o + g, plan_.draft_x);
        }
        for (size_t m = 0; m < steps; ++m) {
            const size_t rows = c.rows[m];
            const backend::RowRun run{rows, 1};
            // One device runs a step whole; a group runs it in three calls a member with a sum after each of the first two (DraftStep).
            for (int phase = W > 1 ? 0 : -1; phase < (W > 1 ? 3 : 0); ++phase) {
                for (size_t g = 0; g < W; ++g) {
                    const backend::Slice hn = slot(ctx_, o + g, plan_.draft_h);
                    DraftStep step{part(ctx_, o + g, drafter_row(g), plan_.drafter->kind, 0, rows, {&run, 1}),
                                   {draft_ids_[g].get(), m * R},
                                   m ? backend::CSlice{hn} : backend::CSlice{devices_[o + g]->carry.get(), 0},
                                   m ? nullptr : c.carried.data(),
                                   {draft_ids_[g].get(), (m + 1) * R},
                                   hn,
                                   {draft_logits_[g].get(), g ? 0 : m * R * V}};
                    step.views = &c.views[(m * W + g) * R];
                    step.n_views = rows;
                    step.kv_layer = drafter_kv_;
                    step.pos = &c.pos[m * R];
                    step.phase = phase;
                    if (W > 1) {
                        step.width = W;
                        step.partial = ctx_.collectives[o]->partial(g);
                        if (phase == 1 && has_experts(*plan_.drafter)) clear_partial(ctx_, o + g, step.partial, rows);
                    }
                    arch_->draft(step);
                }
                if (phase == 0 || phase == 1) ctx_.collectives[o]->sum_into(x, rows, plan_.residual);
            }
        }
        c.sent.assign(W, 0);
        for (size_t g = 0; g < W; ++g) c.sent[g] = devices_[o + g]->b->submit();
        for (size_t g = 0; g < W; ++g) devices_[o + g]->b->wait(c.sent[g]);
        // The ids buffers serve the next chain, which a server may record before this one ends.
        const size_t per = R * (steps + 1);
        c.ids.resize(per * W);
        for (size_t g = 0; g < W; ++g) std::memcpy(c.ids.data() + g * per, draft_ids_[g]->host_ptr(), per * sizeof(uint32_t));
    } catch (...) {
        // Only the head's devices recorded the chain, and another stage's thread may be recording on the rest.
        for (size_t g = 0; g < W; ++g) try { devices_[o + g]->b->sync(); } catch (...) {}
        throw;
    }
    c.recorded = true;
}

// The chain's end on the thread that owns the histories: the drafted rows given back, the sequences out of flight and, once recorded, each history's drafts.
// A chain ends at the first id outside the vocabulary or, on a tensor group, one its members disagree on.
inline void Model::draft_end(Chain& c) {
    const size_t R = c.order.size(), o = (size_t)place_.output_device, W = width_, V = plan_.vocab, per = R * (c.steps + 1);
    for (size_t p = 0; p < c.prepared; ++p) c.kv[p]->abort();
    c.prepared = 0;
    for (size_t p = 0; p < R; ++p) --c.asks[c.order[p]].seq->flights_;
    draft_order_.assign(c.asks.size(), R);
    draft_k_.assign(c.asks.size(), 0);
    draft_width_ = R;
    if (!c.recorded) return;
    for (size_t p = 0; p < R; ++p) {
        draft_order_[c.order[p]] = p;
        draft_k_[c.order[p]] = c.asks[c.order[p]].k;
        for (size_t g = 0; g < W; ++g) c.asks[c.order[p]].seq->last_[o + g] = c.sent[g];
    }
    const uint32_t* ids = c.ids.data();
    const auto agreed = [&](size_t i) {
        for (size_t g = 1; g < W; ++g)
            if (ids[g * per + i] != ids[i]) return false;
        return true;
    };
    for (size_t p = 0; p < R; ++p) {
        const DraftAsk& a = c.asks[c.order[p]];
        for (size_t m = 1; m <= a.k && ids[m * R + p] < V && agreed(m * R + p); ++m) a.out->push_back(ids[m * R + p]);
    }
}

// An image given to a sequence (docs/src/model-history.md): one it carries from a fork or a host copy, given again as the same image with its rows, or one after every image it holds.
// Rows are needed where the history has not read the image whole, and are held until the pass reading the last of them commits.
inline void Model::add_media(Sequence& s, MediaRun run) {
    if (s.owner_ != this) throw std::runtime_error("inference: sequence of another model");
    if (s.flights_) throw std::logic_error("inference: an image added to a sequence in flight");
    if (plan_.image_streams.empty()) throw std::runtime_error("inference: the model's architecture takes no image rows");
    if (!run.rows || size_t(run.grid_h) * run.grid_w != run.rows) throw std::runtime_error("inference: an image's rows do not fill its grid");
    const size_t end = run.first + run.rows;
    if (run.encoded && (run.from >= run.rows || run.encoded->size() != backend::size_mul(run.rows - run.from, embedding_width())))
        throw std::runtime_error("inference: an image's rows are not " + std::to_string(run.rows) + " rows of the embedding's " + std::to_string(embedding_width()) + " floats from row " +
                                 std::to_string(run.from));
    if (!run.encoded && end > s.length()) throw std::runtime_error("inference: an image the history has not read given without its rows");
    if (run.encoded && run.from && run.first + run.from > s.length()) throw std::runtime_error("inference: an image given without rows the history has not read");
    if (run.first > plan_.context_length || run.rows > plan_.context_length - run.first)
        throw std::runtime_error("inference: context length exceeded (" + std::to_string(plan_.context_length) + " tokens)");
    for (Sequence::Media& m : s.media_)
        if (m.run.first == run.first) {
            if (m.run.rows != run.rows || m.run.grid_h != run.grid_h || m.run.grid_w != run.grid_w || m.run.digest != run.digest)
                throw std::logic_error("inference: an image given again is not the image the history holds there");
            if (end > s.length()) {
                m.run.encoded = std::move(run.encoded);
                m.run.from = run.from;
            }
            return;
        }
    if (!s.media_.empty() && run.first < s.media_.back().run.first + s.media_.back().run.rows)
        throw std::logic_error("inference: an image before or over an image the history holds");
    if (end <= s.length()) run.encoded.reset();
    s.media_.push_back(Sequence::Media{std::move(run), std::vector<backend::BufferPtr>(devices_.size())});
}

// Rows [first, first + n) of the history at the positions HF's get_rope_index gives Qwen's images (docs/src/model-history.md): a text row at its index less what the images below it saved, an image's row at (p, p + row, p + column).
// With `streams` each row takes three entries, a text row its position thrice; without, one, and no row may be an image's.
inline void Model::positions(const Sequence& s, size_t first, size_t n, bool streams, uint32_t* out) const {
    const size_t k = streams ? 3 : 1;
    size_t next = 0, saved = 0;   // the first image not wholly below the row, and what the images below it saved
    for (size_t i = first; i < first + n; ++i) {
        while (next < s.media_.size() && s.media_[next].run.first + s.media_[next].run.rows <= i) {
            const MediaRun& r = s.media_[next++].run;
            saved += r.rows - std::max<size_t>(r.grid_h, r.grid_w);
        }
        uint32_t* o = out + (i - first) * k;
        if (next < s.media_.size() && s.media_[next].run.first <= i) {
            if (!streams) throw std::logic_error("inference: an image's row without its position streams");
            const MediaRun& r = s.media_[next].run;
            const size_t p = r.first - saved, at = i - r.first;
            o[0] = (uint32_t)p;
            o[1] = (uint32_t)(p + at / r.grid_w);
            o[2] = (uint32_t)(p + at % r.grid_w);
        } else {
            for (size_t j = 0; j < k; ++j) o[j] = (uint32_t)(i - saved);
        }
    }
}

// The images below `length` a fork or a host copy carries: their places, grids and digests, not their rows.
inline std::vector<Sequence::Media> Model::carried(const Sequence& s, size_t length) const {
    std::vector<Sequence::Media> out;
    for (const Sequence::Media& m : s.media_)
        if (m.run.first < length) {
            MediaRun run = m.run;
            run.encoded.reset();
            run.from = 0;
            out.push_back(Sequence::Media{std::move(run), std::vector<backend::BufferPtr>(devices_.size())});
        }
    return out;
}

// The rows of the images the history has read whole dropped, each device's copy freed once the submission that last read it there has retired (retiring_).
inline void Model::release_media(Sequence& s) {
    for (Sequence::Media& m : s.media_) {
        if (m.run.first + m.run.rows > s.length()) break;
        m.run.encoded.reset();
        for (size_t d = 0; d < m.on.size(); ++d)
            if (m.on[d]) retiring_.push_back(Retiring{d, s.last_[d], std::move(m.on[d])});
    }
}

// The images whose first row is at or past `length`, which the history no longer reaches, dropped.
// With `defer` each device's rows are freed on its stage's thread, which may be recording on that device.
inline void Model::drop_media(Sequence& s, size_t length, const Defer* defer) noexcept {
    std::vector<std::vector<backend::BufferPtr>> rows(devices_.size());
    while (!s.media_.empty() && s.media_.back().run.first >= length) {
        if (defer)
            for (size_t d = 0; d < s.media_.back().on.size() && d < rows.size(); ++d)
                if (s.media_.back().on[d]) rows[d].push_back(std::move(s.media_.back().on[d]));
        s.media_.pop_back();
    }
    if (defer) try { free_slabs(std::move(rows), *defer); } catch (...) {}
}

} // namespace infer
