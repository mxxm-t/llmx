#pragma once
#include "model/runtime.hpp"

// A sequence's history, the one owner of every operation on it (docs/SPECULATIVE.md, section 1): a fork, a reset, the one call that shortens it and what it goes through, the checkpoint a keep makes and the mark a verify takes, with the recurrent inputs a mark saves and the rerun that reads them.
// Members of infer::Model, declared in its class (model/runtime.hpp), which includes this file after it.

namespace infer {

// A second history holding the first `length` tokens of `src`, which must be whole blocks in every storage: every block below `length` is shared, read-only from now on, and the fork appends into fresh ones, so nothing is allocated or copied here.
// The fork inherits the tickets of the passes that wrote what it shares.
// A recurrent state exists only at the end of what it has read, so on a model that keeps one the fork takes src's checkpoint at `length`, whose state its first pass reads in place.
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

// Start a new history.
// Blocks and the state slot return to their pools; their storage is retained.
// Every pass ends in a submit or, on failure, a sync, so the sequence's last tickets cover everything that could still be touching a block or a slot: this waits for those and no more.
inline void Model::reset(Sequence& s) {
    settle(s, "a reset");
    drop_mark(s);
    for (auto& kv : s.kv_) kv.reset();
    std::fill(s.length_.begin(), s.length_.end(), 0);
    std::fill(s.from_.begin(), s.from_.end(), Sequence::kLive);
    s.kept_.release();
    s.state_.release();
}

// The history back to at most `length`, the one call that shortens it (docs/SPECULATIVE.md, section 1), and the length it reached: `length` wherever the caches hold it, else, on a model whose layers keep a state, the sequence's checkpoint at or below it, else 0.
// The caller computes the rest again, without sampling, as a resume does; blocks and a checkpoint past the length reached return to their pools.
// Inside a mark it reaches `length` itself: past the mark its state is run again from the mark's over the kept rows' saved inputs (Architecture::recur), at the mark it is the mark's, and the mark goes; a retract that throws keeps the mark, so it may be called again.
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

// Keep the sequence's state at its current length while one pass runs past it, so a retract into that pass reaches any of its rows exactly (docs/SPECULATIVE.md, section 1): a verify of drafts marks its history first.
// Nothing on a model that keeps no state, whose caches reach every length; on one that keeps a state, the live slot becomes the mark's and the pass writes a fresh one, saving its rows' recurrent inputs.
// False when no mark is free, and then nothing is marked; a second mark is refused.
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

// A sequence out of flight whose passes have retired, before it gives blocks or slots back.
inline void Model::settle(Sequence& s, const char* what) {
    if (s.owner_ != this) throw std::runtime_error("inference: sequence of another model");
    if (s.in_flight_) throw std::logic_error(std::string("inference: ") + what + " of a sequence in flight");
    // A sequence moved from has no tickets, nothing of it being left to wait on.
    for (size_t d = 0; d < devices_.size() && d < s.last_.size(); ++d)
        if (devices_[d]->used) devices_[d]->b->wait(s.last_[d]);
}

// A history back to `length`, or on a model that keeps a state to its checkpoint at or below `length`, else 0, whose live state a pass may have written: every stage and storage at the length reached, which is returned, the blocks and a checkpoint beyond it returned.
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

// Where row 0 of saved item `item` of local state layer `layer` of mark buffer `buffer` sits in device d's saved buffer, in floats; its rows follow one another.
inline size_t Model::saved_at(const Device& d, const LayerPlan& lp, size_t buffer, size_t layer, size_t item) const {
    size_t before = 0;
    for (size_t i = 0; i < item; ++i) before += lp.saved[i].width;
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
            const Saved& v = lp.saved[i];
            d.b->copy(*d.saved, saved_at(d, lp, m.hold.buffer(), (size_t)d.local_layer[(size_t)l], i) * sizeof(float), *sc.arena,
                      sc.offset[v.slot] + (v.plane * p.rows + r0) * v.width * sizeof(float), n * v.width * sizeof(float));
        }
    }
}

// The state after the first `rows` rows of the pass past the sequence's mark, into its live slot: on each device that keeps a state, every state layer's saved inputs copied back into the model's own arena and its update run from the mark's state (Architecture::recur), in each device's order after that pass.
// A failure drains every device and leaves the mark, which a retry reads again.
inline void Model::rerun(Sequence& s, size_t rows) {
    const Sequence::Mark& m = s.mark_;
    try {
        ensure(ctx_, rows, 0, handoffs(1));
        const backend::RowRun run{rows, 1};
        for (size_t st = 0; st < stages_.size(); ++st) {
            const size_t dev = stages_[st].device;
            Device& d = *devices_[dev];
            if (!d.states) continue;
            const ExecContext::Scratch& sc = ctx_.scratch[dev];
            const size_t src = m.from[st] == Sequence::kLive ? s.state_.slot() : m.from[st];
            const backend::StateView view{d.states.get(), src, s.state_.slot(), m.pos, rows};
            for (int l = stages_[st].first; l < stages_[st].end; ++l) {
                const LayerPlan& lp = plan_.layers[(size_t)l];
                if (lp.cache != Cache::state) continue;
                const size_t layer = (size_t)d.local_layer[(size_t)l];
                for (size_t i = 0; i < lp.saved.size(); ++i) {
                    const Saved& v = lp.saved[i];
                    d.b->copy(*sc.arena, sc.offset[v.slot] + v.plane * rows * v.width * sizeof(float), *d.saved,
                              saved_at(d, lp, m.hold.buffer(), layer, i) * sizeof(float), rows * v.width * sizeof(float));
                }
                Step step = part(ctx_, dev, home_[(size_t)l].data(), lp.kind, 0, rows, {&run, 1});
                step.states = &view;
                step.n_views = 1;
                step.state_layer = layer;
                arch_->recur(step);
            }
            s.last_[dev] = d.b->submit();
        }
    } catch (...) {
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

} // namespace infer
