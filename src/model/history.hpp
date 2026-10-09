#pragma once
#include "model/runtime.hpp"

// A sequence's history, the one owner of every operation on it (docs/SPECULATIVE.md, section 1): fork, reset, retract, checkpoint, mark with its rerun, and an embedded drafter's chain of drafts past the history (section 7).
// Members of infer::Model, declared in its class (model/runtime.hpp), which includes this file after it.

namespace infer {

// A second history holding the first `length` tokens of `src`, which must be whole blocks in every storage: every block below `length` is shared and read-only from now on, so nothing is allocated or copied here.
// The fork inherits the tickets of the passes that wrote what it shares, and on a model that keeps a state it takes src's checkpoint at `length`, whose state its first pass reads in place.
inline Sequence Model::fork(const Sequence& src, size_t length) {
    if (src.owner_ != this) throw std::runtime_error("inference: sequence of another model");
    if (src.in_flight_) throw std::logic_error("inference: a fork of a sequence in flight");
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
    drop_mark(s);
    for (auto& kv : s.kv_) kv.reset();
    std::fill(s.length_.begin(), s.length_.end(), 0);
    std::fill(s.from_.begin(), s.from_.end(), Sequence::kLive);
    s.kept_.release();
    s.state_.release();
}

// The history back to at most `length`, the one call that shortens it (docs/SPECULATIVE.md, section 1), returning the length reached: `length` wherever the caches hold it, else the sequence's checkpoint at or below it, else 0.
// The caller computes the rest again, and inside a mark it reaches `length` itself by running the state again from the mark's over the kept rows' saved inputs (Architecture::recur); a retract that throws keeps the mark.
inline size_t Model::retract(Sequence& s, size_t length) {
    settle(s, "a retract");
    if (s.mark_.held() && s.mark_.ran && length > s.mark_.pos && length < s.length()) {
        rerun(s, length - s.mark_.pos);
        drop_mark(s);
        for (size_t& n : s.length_) n = std::min(n, length);
        for (auto& kv : s.kv_) kv.truncate(length);
        return length;
    }
    if (length >= s.length()) {
        if (s.mark_.ran) drop_mark(s);
        else if (s.mark_.held()) restore_mark(s);
        return s.length();
    }
    return rewind(s, length);
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
inline std::optional<size_t> Model::checkpoint(const Sequence& s) const {
    if (!s.kept_.held()) return std::nullopt;
    return s.kept_.pos();
}

// The host memory a copy of `length` tokens takes (HostHistory::held): per device, whole blocks of its KV storage's layers, K and V, and one slot of its state storage, in whole slabs.
// Without `blocks` it is the slot alone, and from `first` on and without `state` the blocks of a range alone.
inline size_t Model::host_bytes(size_t length, bool blocks, size_t first, bool state) const {
    size_t n = 0;
    for (const auto& d : devices_) {
        size_t bytes = 0;
        if (const auto* st = blocks ? dynamic_cast<const backend::BlockKVStorage*>(d->storage.get()) : nullptr)
            bytes += (backend::blocks_for(length, st->block_tokens()) - first / st->block_tokens()) * (st->k_block_bytes() + st->v_block_bytes()) * st->layers();
        if (state && d->states) bytes += d->states->layers() * d->states->shape().slot_floats() * sizeof(float);
        if (state && d->carry && state_layers_) bytes += plan_.residual * sizeof(float);
        n += backend::blocks_for(bytes, kHostSlab) * kHostSlab;
    }
    return n;
}

namespace detail {
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
    // `n` blocks of a sequence in `st`, from `ids` on, each layer's K blocks then its V blocks, a run of consecutive blocks in one copy.
    void blocks(backend::BlockKVStorage& st, const int32_t* ids, size_t n) {
        for (size_t l = 0; l < st.layers(); ++l)
            for (int side = 0; side < 2; ++side) {
                backend::Buffer& buf = side ? *st.v_buffer(l) : *st.k_buffer(l);
                const size_t block = side ? st.v_block_bytes() : st.k_block_bytes();
                for (size_t i = 0; i < n;) {
                    size_t e = i + 1;
                    while (e < n && ids[e] == ids[e - 1] + 1) ++e;
                    copy(buf, (size_t)ids[i] * block, (e - i) * block);
                    i = e;
                }
            }
    }
    // One slot of every layer of `states`.
    void slot(backend::StateStorage& states, size_t slot) {
        const size_t bytes = states.shape().slot_floats() * sizeof(float);
        for (size_t l = 0; l < states.layers(); ++l) copy(states.layer(l), slot * bytes, bytes);
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

// Whether `bytes` more of host memory leave the host the reserve the fit keeps on it (CpuBackend::host_reserve) out of `free`, what it has free now; an unknown figure refuses nothing.
inline bool host_room(std::optional<size_t> free, size_t bytes) {
    return !free || (bytes <= *free && *free - bytes >= backend::CpuBackend::host_reserve(*free));
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

// Slabs for a copy of `length` tokens of this model's layout (HostHistory), as save_host takes them, holding nothing yet: within `limit` and the host reserve, idle slabs reused and those of other devices freed first.
// From `first` on and without `state` they are for the blocks of a range alone (save_host_blocks), and `out` is whole or, on a throw, released.
inline void Model::alloc_host(size_t length, HostHistory& out, size_t limit, bool blocks, size_t first, bool state) {
    release_host(out);
    // Each sized on its own, so one that fails to grow leaves the other's check to grow it next time.
    if (host_allocated_.size() != devices_.size()) host_allocated_.resize(devices_.size(), 0);
    if (host_slabs_.size() != devices_.size()) host_slabs_.resize(devices_.size());
    std::vector<size_t> bytes(devices_.size(), 0), need(devices_.size(), 0), idle(devices_.size(), 0);
    for (size_t i = 0; i < devices_.size(); ++i) {
        const Device& d = *devices_[i];
        if (const auto* st = blocks ? dynamic_cast<const backend::BlockKVStorage*>(d.storage.get()) : nullptr)
            bytes[i] = (length - first) / st->block_tokens() * (st->k_block_bytes() + st->v_block_bytes()) * st->layers();
        if (state && d.states) bytes[i] += d.states->layers() * d.states->shape().slot_floats() * sizeof(float);
        if (state && d.carry && state_layers_) bytes[i] += plan_.residual * sizeof(float);
        need[i] = backend::blocks_for(bytes[i], kHostSlab);
        idle[i] = host_slabs_[i].size();
    }
    // Idle slabs are retired (release_host), so freeing them waits for nothing.
    const auto drop = detail::slabs_to_free(need, idle, host_allocated() / kHostSlab, limit / kHostSlab);
    if (!drop) throw std::runtime_error("inference: no host memory for the copy within its limit");
    size_t fresh = 0;
    for (size_t i = 0; i < devices_.size(); ++i) {
        for (size_t k = 0; k < (*drop)[i]; ++k) {
            host_slabs_[i].pop_back();
            --host_allocated_[i];
        }
        fresh += need[i] > host_slabs_[i].size() ? need[i] - host_slabs_[i].size() : 0;
    }
    // The slabs it allocates must leave the host the reserve the fit kept on it, read once the idle slabs above are freed.
    if (fresh && !detail::host_room(core::host_memory_available(), fresh * kHostSlab))
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
    try {
        for (size_t i = 0; i < devices_.size(); ++i) {
            if (!bytes[i]) continue;
            Device& d = *devices_[i];
            std::vector<backend::BufferPtr>& slabs = h.slabs[i];
            while (slabs.size() < need[i]) {
                if (host_slabs_[i].empty()) {
                    host_slabs_[i].reserve(host_allocated_[i] + 1);
                    slabs.push_back(d.b->alloc(kHostSlab, backend::Memory::host_visible));
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
        release_host(h);
        throw;
    }
    out = std::move(h);
}

// The history's first `length` tokens (whole blocks of every storage) copied to host memory, with its checkpoint's slot at `length` on a model that keeps a state; without `blocks` only the state is copied.
// The copies are enqueued behind the passes that wrote the history and nothing waits for them, the slabs stay within `limit` and the host's reserve, and `out` is whole or, on a throw, released (docs/SPECULATIVE.md, section 2, Host tier).
inline void Model::save_host(Sequence& s, size_t length, HostHistory& out, size_t limit, bool blocks) {
    settle(s, "a copy to host memory");
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
    alloc_host(length, h, limit, blocks);
    try {
        for (size_t i = 0; i < devices_.size(); ++i) {
            if (h.slabs[i].empty()) continue;
            Device& d = *devices_[i];
            detail::HostSpan span{*d.b, h.slabs[i], kHostSlab, true};
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
    out = std::move(h);
}

// The blocks of the history's tokens from `first` to `length` copied to host memory alone, with no state: what a turn added to a history whose earlier blocks are kept elsewhere (docs/DISK-TIER.md, Entries written as what changed).
// Enqueued and limited as save_host's copies are; `out` is whole or, on a throw, released.
inline void Model::save_host_blocks(Sequence& s, size_t first, size_t length, HostHistory& out, size_t limit) {
    settle(s, "a copy to host memory");
    release_host(out);
    if (s.mark_.held()) throw std::logic_error("inference: a copy to host memory of a marked sequence");
    if (first >= length || length > s.length()) throw std::logic_error("inference: a copy to host memory of blocks outside the history");
    for (const Device* d : storages_) {
        if (!dynamic_cast<const backend::BlockKVStorage*>(d->storage.get())) throw std::runtime_error("inference: a KV storage that cannot be copied to host memory");
        if (first % d->b->kv_layout().block_tokens || length % d->b->kv_layout().block_tokens) throw std::logic_error("inference: a copy to host memory takes whole blocks of the history");
    }
    HostHistory h;
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
    out = std::move(h);
}

// Where the blocks of tokens `first` to `length` lie in host history `h`, which holds them: per device, a layer's K blocks then its V blocks, in the order a copy of that range alone holds them end to end.
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
            for (const size_t block : {st->k_block_bytes(), st->v_block_bytes()}) {
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
            blocks = (h.length - h.first) / st->block_tokens() * (st->k_block_bytes() + st->v_block_bytes()) * st->layers();
        if (h.device_bytes[i] > blocks) out.push_back(HostRange{i, blocks, h.device_bytes[i] - blocks});
    }
    return out;
}

// A fresh history holding what `h` copied, copied back into blocks and, on a model that keeps a state, a checkpoint slot of this model at h.length, so a fork or the history itself continues with the bits of the history it was copied from.
// The copies are enqueued ahead of any pass of the history, whose tickets cover them, and a throw leaves nothing held.
inline Sequence Model::restore_host(HostHistory& h) {
    if (h.owner != this) throw std::runtime_error("inference: a history copied to host memory by another model");
    if (h.slabs.size() != devices_.size() || !h.blocks || h.first || (state_layers_ && !h.state)) throw std::logic_error("inference: a host history of another layout");
    Sequence s = make_sequence();
    size_t slot = 0;
    if (state_layers_) {
        slot = slots_.acquire_kept();
        s.kept_ = Checkpoint(slots_, slot, h.length);
        std::fill(s.from_.begin(), s.from_.end(), slot);
    }
    try {
        for (size_t i = 0; i < devices_.size(); ++i) {
            Device& d = *devices_[i];
            auto* st = dynamic_cast<backend::BlockKVStorage*>(d.storage.get());
            if (d.storage && !st) throw std::runtime_error("inference: a KV storage that cannot be copied from host memory");
            if ((st || d.states) && h.slabs[i].empty()) throw std::logic_error("inference: a host history of another layout");
            detail::HostSpan span{*d.b, h.slabs[i], kHostSlab, false};
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
    } catch (...) {
        // Copies out of h's slabs may be enqueued past its tickets, so they retire before h can be released.
        for (auto& d : devices_) d->b->sync();
        throw;
    }
    std::fill(s.length_.begin(), s.length_.end(), h.length);
    for (size_t i = 0; i < devices_.size(); ++i)
        if (!h.slabs[i].empty()) h.tickets[i] = s.last_[i] = devices_[i]->b->submit();
    return s;
}

// A second history holding the first `length` tokens of `src`, whose blocks it shares as fork does, and the state at `length` that `state` holds (Model::save_host without blocks, docs/SPECULATIVE.md, section 2, Host tier).
// The state is copied back into a checkpoint slot of its own, which the fork's first pass reads in place, enqueued behind the passes that wrote what it shares; a throw leaves nothing held.
inline Sequence Model::fork(const Sequence& src, size_t length, HostHistory& state) {
    if (src.owner_ != this || state.owner != this) throw std::runtime_error("inference: sequence or host state of another model");
    if (src.in_flight_) throw std::logic_error("inference: a fork of a sequence in flight");
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
    const size_t slot = slots_.acquire_kept();
    f.kept_ = Checkpoint(slots_, slot, length);
    f.from_.assign(stages_.size(), slot);
    try {
        for (size_t i = 0; i < devices_.size(); ++i) {
            Device& d = *devices_[i];
            if (!d.states && !d.carry) continue;
            if (state.slabs[i].empty()) throw std::logic_error("inference: a host state of another layout");
            detail::HostSpan span{*d.b, state.slabs[i], kHostSlab, false};
            if (d.states) span.slot(*d.states, slot);
            // An embedded drafter's carried row comes back with the state it was saved beside, so the fork drafts as the history it was taken from did.
            if (d.carry) span.copy(*d.carry, slot * plan_.residual * sizeof(float), plan_.residual * sizeof(float));
            state.tickets[i] = f.last_[i] = d.b->submit();
        }
    } catch (...) {
        // Copies out of the state's slabs may be enqueued past its tickets, so they retire before it can be released.
        for (auto& d : devices_) d->b->sync();
        throw;
    }
    return f;
}

// What a history copied to host memory holds and how, beyond the model file and the build: each device's identity, effective dtype and runs, the cache types, the placement and the row classes up to the most a history holds.
// Two models of one file and one build with equal identities hold the same bytes for the same history, so a copy written by one is restored by the other (docs/DISK-TIER.md, The entry file).
inline std::string Model::host_identity() const {
    std::string s = "llmx-host-history 1\n";
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
        if (const auto* st = dynamic_cast<const backend::BlockKVStorage*>(d.storage.get()))
            s += "; kv " + std::to_string(st->layers()) + " layers, " + std::to_string(st->block_tokens()) + " tokens a block, " + std::to_string(st->k_block_bytes()) + "+" +
                 std::to_string(st->v_block_bytes()) + " bytes";
        if (d.states) s += "; state " + std::to_string(d.states->layers()) + " layers, " + std::to_string(d.states->shape().slot_floats() * sizeof(float)) + " bytes";
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
    if (h.owner == this)
        for (size_t i = 0; i < h.slabs.size() && i < devices_.size(); ++i) {
            if (h.slabs[i].empty()) continue;
            devices_[i]->b->wait(h.tickets[i]);
            for (auto& b : h.slabs[i]) host_slabs_[i].push_back(std::move(b));
        }
    h = HostHistory{};
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
inline void Model::wait_host(const HostHistory& h) const noexcept {
    if (h.owner == this)
        for (size_t i = 0; i < h.slabs.size() && i < devices_.size(); ++i)
            if (!h.slabs[i].empty()) devices_[i]->b->wait(h.tickets[i]);
}

// A sequence out of flight whose passes have retired, before it gives blocks or slots back.
inline void Model::settle(Sequence& s, const char* what) {
    if (s.owner_ != this) throw std::runtime_error("inference: sequence of another model");
    if (s.in_flight_) throw std::logic_error(std::string("inference: ") + what + " of a sequence in flight");
    // A sequence moved from has no tickets, nothing of it being left to wait on.
    for (size_t d = 0; d < devices_.size() && d < s.last_.size(); ++d)
        if (devices_[d]->used) devices_[d]->b->wait(s.last_[d]);
}

// A history back to `length`, or on a model that keeps a state to its checkpoint at or below it, else 0, returning the length reached: every stage and storage at it, the blocks and a checkpoint beyond it returned.
// The state is then the checkpoint's, or zero, so the live slot goes back too, and the next pass takes one: a donor parked at its checkpoint holds none of the slots admission counts on.
inline size_t Model::rewind(Sequence& s, size_t length) noexcept {
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
inline void Model::rerun(Sequence& s, size_t rows) {
    const Sequence::Mark& m = s.mark_;
    try {
        ensure(ctx_, rows, 0, handoffs(1));
        // The update's calls take the mark's rows, which a plane of its saved inputs holds, and their views the rows kept.
        const backend::RowRun run{options_.mark_rows, 1};
        std::vector<size_t> offsets(plan_.slots.size());
        int phases = 0;
        for (const LayerPlan& lp : plan_.layers)
            if (lp.cache == Cache::state) phases = std::max(phases, lp.recur_phases);
        for (size_t dev = 0; dev < devices_.size(); ++dev) {
            Device& d = *devices_[dev];
            const bool carry = plan_.drafter && dev - d.member == (size_t)place_.output_device;
            if (!d.states && !carry) continue;
            // Nothing a phase runs reads what another layer's writes: each layer reads its own saved inputs and slot, and writes its own room and slot.
            for (int phase = 0; d.states && phase < phases; ++phase) {
                d.b->unordered(true);
                for (size_t st = 0; st < stages_.size(); ++st) {
                    if (stages_[st].device != dev - d.member) continue;
                    const size_t src = m.from[st] == Sequence::kLive ? s.state_.slot() : m.from[st];
                    const backend::StateView view{d.states.get(), src, s.state_.slot(), m.pos, rows};
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
                            if (!lp.saved[i].plane) offsets[lp.saved[i].slot] = saved_at(d, lp, m.hold.buffer(), layer, i) * sizeof(float);
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
            if (carry) {
                const size_t E = plan_.residual;
                d.b->copy(*d.carry, s.state_.slot() * E * sizeof(float), *d.saved_h, (m.hold.buffer() * options_.mark_rows + rows - 1) * E * sizeof(float),
                          E * sizeof(float));
            }
            s.last_[dev] = d.b->submit();
        }
    } catch (...) {
        // A failure inside a phase leaves its device recording without barriers; the mode is set before any barrier is recorded, so it is off even where recording that barrier fails.
        for (auto& d : devices_) try { d->b->unordered(false); } catch (...) {}
        retire();
        throw;
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
    if (!plan_.drafter) throw std::logic_error("inference: a draft without an embedded drafter");
    const size_t S = stages_.size(), V = plan_.vocab;
    std::vector<size_t> order;
    size_t steps = 0;
    for (size_t i = 0; i < n; ++i) {
        DraftAsk& a = asks[i];
        settle(*a.seq, "a draft");
        a.out->clear();
        if (!a.k) continue;
        const size_t L = history(*a.seq);
        if (!L || !a.seq->state_.held()) throw std::logic_error("inference: a draft of a history no pass has fed");
        if (a.k > plan_.context_length - std::min(plan_.context_length, L))
            throw std::runtime_error("inference: context length exceeded (" + std::to_string(plan_.context_length) + " tokens)");
        order.push_back(i);
        steps = std::max(steps, a.k);
    }
    if (order.empty()) return;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return asks[a].k > asks[b].k; });
    const size_t R = order.size(), o = (size_t)place_.output_device, W = width_;
    Device& d = *devices_[o];
    ensure(ctx_, R, 0, handoffs(1));
    if (draft_id_rows_ < R * (steps + 1) || draft_rows_ < R * steps || draft_seqs_ < R) {
        const size_t id_rows = std::max(draft_id_rows_, R * (steps + 1)), rows = std::max(draft_rows_, R * steps), seqs = std::max(draft_seqs_, R);
        std::vector<backend::BufferPtr> ids(W), logits(W);
        for (size_t m = 0; m < W; ++m) {
            backend::Backend& b = *devices_[o + m]->b;
            ids[m] = b.alloc(backend::size_mul(id_rows, sizeof(float)), backend::Memory::host_visible);
            logits[m] = m ? b.alloc(backend::size_mul(seqs, backend::size_mul(V, sizeof(float))))
                          : b.alloc(backend::size_mul(rows, backend::size_mul(V, sizeof(float))), backend::Memory::host_visible);
        }
        draft_ids_ = std::move(ids);
        draft_logits_ = std::move(logits);
        draft_id_rows_ = id_rows;
        draft_rows_ = rows;
        draft_seqs_ = seqs;
    }
    draft_order_.assign(n, R);
    draft_k_.assign(n, 0);
    draft_width_ = R;
    std::vector<uint32_t> last(R), carried(R), pos(R);
    std::vector<size_t> length(R);
    std::vector<KVSequence*> kv(R);
    for (size_t p = 0; p < R; ++p) {
        Sequence& s = *asks[order[p]].seq;
        draft_order_[order[p]] = p;
        last[p] = asks[order[p]].last;
        length[p] = history(s);
        carried[p] = (uint32_t)(s.from_[S - 1] == Sequence::kLive ? s.state_.slot() : s.from_[S - 1]);
        kv[p] = &s.kv_[(size_t)d.storage_index];
    }
    size_t prepared = 0;
    try {
        for (; prepared < R; ++prepared) kv[prepared]->prepare(asks[order[prepared]].k);
        std::vector<backend::Slice> x(W);
        for (size_t g = 0; g < W; ++g) {
            devices_[o + g]->b->write(*draft_ids_[g], 0, last.data(), R * sizeof(uint32_t));
            x[g] = slot(ctx_, o + g, plan_.draft_x);
        }
        std::vector<std::vector<backend::KVView>> views(W, std::vector<backend::KVView>(R));
        for (size_t m = 0, rows = R; m < steps; ++m) {
            while (asks[order[rows - 1]].k <= m) --rows;
            for (size_t p = 0; p < rows; ++p) {
                for (size_t g = 0; g < W; ++g) {
                    views[g][p] = kv[p]->view(devices_[o + g]->storage.get());
                    views[g][p].length = length[p] + m;
                    views[g][p].nq = 1;
                    views[g][p].extent = 1;
                }
                pos[p] = (uint32_t)(length[p] + m);
            }
            const backend::RowRun run{rows, 1};
            // One device runs a step whole; a group runs it in three calls a member with a sum after each of the first two (DraftStep).
            for (int phase = W > 1 ? 0 : -1; phase < (W > 1 ? 3 : 0); ++phase) {
                for (size_t g = 0; g < W; ++g) {
                    const backend::Slice hn = slot(ctx_, o + g, plan_.draft_h);
                    DraftStep step{part(ctx_, o + g, drafter_row(g), plan_.drafter->kind, 0, rows, {&run, 1}),
                                   {draft_ids_[g].get(), m * R},
                                   m ? backend::CSlice{hn} : backend::CSlice{devices_[o + g]->carry.get(), 0},
                                   m ? nullptr : carried.data(),
                                   {draft_ids_[g].get(), (m + 1) * R},
                                   hn,
                                   {draft_logits_[g].get(), g ? 0 : m * R * V}};
                    step.views = views[g].data();
                    step.n_views = rows;
                    step.kv_layer = drafter_kv_;
                    step.pos = pos.data();
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
        std::vector<backend::Ticket> sent(W);
        for (size_t g = 0; g < W; ++g) {
            sent[g] = devices_[o + g]->b->submit();
            for (size_t p = 0; p < R; ++p) asks[order[p]].seq->last_[o + g] = sent[g];
        }
        for (size_t g = 0; g < W; ++g) devices_[o + g]->b->wait(sent[g]);
    } catch (...) {
        retire();
        for (size_t p = 0; p < prepared; ++p) kv[p]->abort();
        throw;
    }
    for (size_t p = 0; p < R; ++p) {
        kv[p]->abort();
        draft_k_[order[p]] = asks[order[p]].k;
    }
    const uint32_t* ids = (const uint32_t*)draft_ids_[0]->host_ptr();
    const auto agreed = [&](size_t i) {
        for (size_t g = 1; g < W; ++g)
            if (((const uint32_t*)draft_ids_[g]->host_ptr())[i] != ids[i]) return false;
        return true;
    };
    for (size_t p = 0; p < R; ++p) {
        const DraftAsk& a = asks[order[p]];
        for (size_t m = 1; m <= a.k && ids[m * R + p] < V && agreed(m * R + p); ++m) a.out->push_back(ids[m * R + p]);
    }
}

} // namespace infer
