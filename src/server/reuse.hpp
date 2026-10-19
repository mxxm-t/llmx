#pragma once
#include "server/scheduler.hpp"

// The cache's moves between its places, the one owner of each (docs/CACHE-PLAN.md, section 2): device donors, message boundaries, host entries, the disk tier's side and the restores from it.
// Members of server::Scheduler, declared in its class (server/scheduler.hpp), which includes this file after it.

namespace server {

// Takes fresh slabs of the host tier while the server has nothing to do, kHostGrow a round or the shortest copy's where that needs more, and gives them back at once to stand idle in the model's pool; whether more are to come.
// A fresh slab is zero-filled as it is taken, which a read from disk otherwise spends on this thread before it starts (docs/STATUS.md); the model counts the fresh ones device by device (Model::host_fresh), and this runs under the lock.
inline bool Scheduler::grow_host() {
    const size_t slab = infer::Model::host_slab_bytes(), alive = model_.host_allocated(), in_use = host_held_ + parked_held();
    if (alive + slab > host_cap_ || in_use > alive) return false;
    const size_t bt = model_.kv_block_tokens(), limit = token_limit();
    std::vector<infer::HostHistory> probes;
    try {
        // The round's fresh slabs are those the pool admits as growth nothing waits for, and each copy is asked for by the slabs it would itself take anew, the shortest copy's included.
        size_t fresh = kHostGrow;
        while (fresh && !model_.host_admits(fresh * slab, true)) --fresh;
        for (size_t room = host_cap_ - in_use; room && (fresh || probes.empty());) {
            const size_t n = grow_tokens(room, fresh, bt, limit, probes.empty(), [&](size_t t) { return model_.host_bytes(t); }, [&](size_t t) { return model_.host_fresh(t); });
            const size_t anew = n ? model_.host_fresh(n) : 0;
            if (!n || (anew && !model_.host_admits(anew * slab, true))) break;
            fresh -= std::min(fresh, anew);
            probes.emplace_back();
            quieted()->alloc_host(n, probes.back(), host_cap_ + parked_held());
            room -= std::min(room, probes.back().held);
        }
    } catch (const std::exception&) {}
    for (infer::HostHistory& h : probes) model_.release_host(h);
    return model_.host_allocated() > alive;
}

// Once every kHostCheck: idle slabs go back to a host below the pool's reserve (Model::host_yield), and the idle growth is tried again where none had to, which the pool refuses while the host is short.
// Under the lock.
inline void Scheduler::host_pressure() {
    const Clock::time_point now = Clock::now();
    if (!host_cap_ || now < host_check_) return;
    host_check_ = now + kHostCheck;
    const size_t freed = model_.host_short() ? quieted()->host_yield() : 0;
    host_yielded_ += freed;
    host_alive_ = model_.host_allocated();
    // A run of give-backs is said as it starts and, once the host has gone kHostQuiet without one, with what it gave back; growth waits for that quiet too.
    if (freed) {
        if (!host_run_bytes_) {
            host_run_since_ = now;
            std::fprintf(stderr, "server: giving idle host memory back, the host being short of memory\n");
        }
        host_run_bytes_ += freed;
        host_quiet_ = now + kHostQuiet;
    } else if (host_run_bytes_ && now >= host_quiet_) {
        std::fprintf(stderr, "server: %.0f MiB of idle host memory given back over %.0f s, the host being short of memory\n", (double)host_run_bytes_ / (1 << 20),
                     std::chrono::duration<double>(host_quiet_ - kHostQuiet - host_run_since_).count());
        host_run_bytes_ = 0;
    }
    host_growing_ = !freed && now >= host_quiet_;
}

// A resumed request's own donor, the one its pause left, when nothing has evicted it; donors_.size() when there is none.
inline size_t Scheduler::own_donor(const Request& r) const {
    for (size_t d = 0; r.donor_ && d < donors_.size(); ++d)
        if (donors_[d].id == r.donor_) return d;
    return donors_.size();
}

// The one count of the device's room (docs/CACHE-PLAN.md, section 4): what is free in each pool now, the model's free blocks less the room the running requests promised and have not taken.
// A request has promised its reservation (`need_`) less the blocks its sequence holds, a pass's prepared blocks among them, so one in flight counts at what its pass took and its step is not subtracted twice.
inline std::vector<size_t> Scheduler::room() const {
    std::vector<size_t> free = model_.kv_free();
    if (running_)
        for (const auto& r : *running_) {
            const std::vector<size_t> p = promised(*r);
            for (size_t s = 0; s < free.size(); ++s) free[s] -= std::min(free[s], p[s]);
        }
    return free;
}

inline std::vector<size_t> Scheduler::promised(const Request& r) const {
    std::vector<size_t> p(pools_.blocks.size(), 0);
    if (r.need_.empty()) return p;
    const std::vector<size_t> held = model_.kv_held(r.seq_);
    for (size_t s = 0; s < p.size(); ++s) p[s] = r.need_[s] > held[s] ? r.need_[s] - held[s] : 0;
    return p;
}

// make_room's gain over donors_ and `active`: what the model frees of every victim together, asked of the whole set and never victim by victim (Model::freed), with each paused request's promise.
// The donor `keep` a request forks keeps the `kept_len` tokens it shares, one it takes back (`take`) frees nothing, its blocks being the request's, and `also`, a sequence going in any case, counts for what it frees beside the victims.
inline Gain Scheduler::gain_of(const std::vector<std::shared_ptr<Request>>& active, size_t keep, size_t kept_len, bool take, const infer::Sequence* also) const {
    const std::vector<size_t> base = also ? model_.freed({also}).blocks : std::vector<size_t>();
    return [this, &active, keep, kept_len, take, also, base](const Taken& t) {
        std::vector<const infer::Sequence*> set;
        std::vector<size_t> g(pools_.blocks.size(), 0);
        if (also) set.push_back(also);
        bool forked = false;
        for (size_t d : t.donors) {
            if (d == keep && take) continue;
            forked = forked || (d == keep && kept_len);
            set.push_back(&donors_[d].seq);
        }
        for (const auto& [i, with] : t.paused) {
            add(g, promised(*active[i]));
            if (with) set.push_back(&active[i]->seq_);
        }
        add(g, model_.freed(set, forked ? &donors_[keep].seq : nullptr, kept_len).blocks);
        return base.empty() ? g : less(g, base);
    };
}

// Stats::reserved and Stats::donor_blocks as the device's room stands: each pool's blocks taken or promised, and those only donors and the histories being streamed in hold; under the lock.
inline void Scheduler::note_room() {
    const std::vector<size_t> free = room();
    reserved_now_.assign(free.size(), 0);
    for (size_t s = 0; s < free.size(); ++s) reserved_now_[s] = pools_.blocks[s] - free[s];
    std::vector<const infer::Sequence*> held;
    for (const Donor& d : donors_) held.push_back(&d.seq);
    for (const Streaming& st : streams_) held.push_back(&st.into->seq);
    donor_blocks_now_ = model_.freed(held).blocks;
}

// A free checkpoint slot for a keep beside `keeps` others the pass takes, the oldest donors that hold one giving theirs where none is free (make_room, checkpoint slots being one more pool).
// The donor `spare` a job forked keeps its slot, which its fork may still read, and a checkpoint is never forced, so false leaves it out.
inline bool Scheduler::checkpoint_room(size_t keeps, uint64_t spare) {
    if (model_.checkpoints_free() > keeps) return true;
    std::lock_guard<std::mutex> lk(m_);
    return checkpoint_room_held(keeps, spare);
}

// checkpoint_room under the lock.
inline bool Scheduler::checkpoint_room_held(size_t keeps, uint64_t spare) {
    if (model_.checkpoints_free() > keeps) return true;
    const size_t free = model_.checkpoints_free();
    std::vector<size_t> at;
    for (size_t d = 0; d < donors_.size(); ++d)
        if (!spare || donors_[d].id != spare) at.push_back(d);
    // A donor frees its slot only where no fork still reads it, which the model counts (Model::freed).
    const Taken t = make_room({0}, at.size(), npos, false, {}, 0, false, {keeps - free + 1}, [&](const Taken& p) {
        std::vector<const infer::Sequence*> set;
        for (size_t i : p.donors) set.push_back(&donors_[at[i]].seq);
        return std::vector<size_t>{model_.freed(set).slots};
    });
    if (!t.enough) return false;
    std::vector<size_t> gone;
    for (size_t i : t.donors) gone.push_back(at[i]);
    std::sort(gone.begin(), gone.end(), std::greater<size_t>());
    for (size_t i : gone) drop_donor(i, true);
    // A slot a fork still reads stays held after its donor goes.
    return model_.checkpoints_free() > keeps;
}

// The donor sharing the longest run of whole blocks with r's history, over rows computed as r's were or, at its first admission, would be, the newest where several share as much; the run's length goes to `tokens`, zero for none.
// On a model that keeps a state the run reaches only as far as the donor's checkpoint, and the last history token is never shared, since a pass must compute it to give logits.
inline size_t Scheduler::best_donor(const Request& r, size_t& tokens) const {
    size_t best = donors_.size();
    tokens = 0;
    for (size_t d = 0; d < donors_.size(); ++d) {
        const size_t n = shareable(r, donors_[d].tokens, donors_[d].classes, donors_[d].seq);
        if (n && n >= tokens) { tokens = n; best = d; }
    }
    return best;
}

// How much of r's history a fork of `seq`, holding the history `t` computed as `classes` record, can give it: whole blocks of the same tokens over rows computed as r's were or, at its first admission, would be.
// A job's rows are at the class every longer prompt takes, it is within what `seq` holds, and on a model that keeps a state only as far as the checkpoint.
inline size_t Scheduler::shareable(const Request& r, const std::vector<uint32_t>& t, const std::vector<RowClass>& classes, const infer::Sequence& seq) const {
    return shareable(r, t, classes, seq.length(), model_.checkpoint(seq));
}

// The same for a history of `held` tokens whose state, on a model that keeps one, is kept at `kept`.
inline size_t Scheduler::shareable(const Request& r, const std::vector<uint32_t>& t, const std::vector<RowClass>& classes, size_t held, std::optional<size_t> kept) const {
    const size_t bt = model_.kv_block_tokens(), h = history_tokens(r);
    const std::vector<RowClass> first{RowClass{r.prompt_.size(), r.prompt_.size()}};
    const std::vector<RowClass>& own = r.classes_.empty() ? first : r.classes_;
    size_t n = 0;
    // A request leaves its last token for the pass that gives its logits; a job wants none and shares every id a history holds.
    const size_t limit = std::min({t.size(), r.job_ ? h : h - 1, held});
    while (n < limit && t[n] == *token_ptr(r, n)) ++n;
    n = alike(own, classes, n);
    n = n / bt * bt;
    if (model_.keeps_state()) n = kept && *kept <= n && *kept % bt == 0 ? *kept : 0;
    return n;
}

// Job donor d's state kept as a boundary as its job completes (keep_boundary below).
inline void Scheduler::keep_boundary(Donor& d) { keep_boundary(d.seq, d.tokens, d.classes); }

// The boundary request r asked for, its pass retired: kept, or said not kept with what keep_boundary found; under the lock.
inline void Scheduler::asked_boundary(Request& r) {
    const std::optional<size_t> kept = model_.keeps_state() ? model_.checkpoint(r.seq_) : std::nullopt;
    const size_t before = bounds_.size(), bt = model_.kv_block_tokens();
    const uint64_t was = moves_[(size_t)Move::boundary_kept].load() + moves_[(size_t)Move::boundary_not_kept].load();
    keep_boundary(r.seq_, r.prompt_, r.classes_);
    if (!host_cap_ || !model_.keeps_state() || bounds_.size() > before || moves_[(size_t)Move::boundary_kept].load() + moves_[(size_t)Move::boundary_not_kept].load() != was) return;
    const char* why = !kept ? "its history has no state kept once its pass retired" : !*kept || *kept % bt ? "the state kept is not at a whole block" : *kept > r.prompt_.size() ? "the state kept is past its prompt" : nullptr;
    if (why) moved(Move::boundary_not_kept, "server: cache: the message boundary a request asked for was not kept (%s, at %zu)\n", why, kept.value_or(0));
}

// The checkpoint of `seq`, holding `tokens` computed as `classes` record, kept as a boundary, the copy enqueued behind the passes that wrote it, within the room the host tier and the host's free memory leave; one already kept is renewed.
// Superseded copies and then the longest unheard conversation's boundaries go first and, without a disk tier, a conversation's are thinned to kBoundaries, so the state survives the history's fate; under the lock.
inline void Scheduler::keep_boundary(infer::Sequence& seq, const std::vector<uint32_t>& tokens, const std::vector<RowClass>& classes) {
    if (!host_cap_ || !model_.keeps_state()) return;
    const std::optional<size_t> kept = model_.checkpoint(seq);
    const size_t bt = model_.kv_block_tokens();
    if (!kept || !*kept || *kept % bt || *kept > tokens.size()) return;
    const size_t n = *kept;
    // The conversation's boundaries: those whose tokens d's begin with.
    std::vector<size_t> mine;
    for (size_t i = 0; i < bounds_.size(); ++i) {
        const auto& t = bounds_[i].tokens;
        if (t.size() > n || !std::equal(t.begin(), t.end(), tokens.begin())) continue;
        if (t.size() == n && alike(bounds_[i].classes, classes, n) == n) {
            bounds_[i].used = std::filesystem::file_time_type::clock::now();
            if (bounds_[i].disk) disk_->renew(bounds_[i].disk, true);
            std::rotate(bounds_.begin() + (std::ptrdiff_t)i, bounds_.begin() + (std::ptrdiff_t)i + 1, bounds_.end());
            return;
        }
        mine.push_back(i);
    }
    // Thinned before the copy, by position: the one between its neighbours whose gap would grow least goes, never the first nor one a request being admitted has pinned.
    while (!disk_ && mine.size() + 1 > kBoundaries) {
        std::sort(mine.begin(), mine.end(), [&](size_t a, size_t b) { return bounds_[a].tokens.size() < bounds_[b].tokens.size(); });
        size_t best = 0, gap = std::numeric_limits<size_t>::max();
        for (size_t k = 1; k < mine.size(); ++k) {
            const size_t next = k + 1 < mine.size() ? bounds_[mine[k + 1]].tokens.size() : n;
            const size_t g = next - bounds_[mine[k - 1]].tokens.size();
            if (g < gap && bounds_[mine[k]].id != pinned_bound_) { gap = g; best = k; }
        }
        if (!best) break;
        const size_t gone = mine[best];
        drop_bound(gone);
        mine.erase(mine.begin() + (std::ptrdiff_t)best);
        for (size_t& i : mine) if (i > gone) --i;
    }
    const size_t bytes = model_.host_bytes(n, false);
    while (!host_.empty() && host_.front().superseded && host_held_ + bytes > host_cap_) drop_host(0);
    release_written(bytes);
    while (host_held_ + bytes > host_cap_ && drop_oldest_bound()) {}
    if (host_held_ + bytes > host_cap_) {
        moved(Move::boundary_not_kept, "server: a message boundary at %zu tokens was not kept in host memory (the host tier has no room it may give)\n", n);
        return;
    }
    // The conversation's boundaries take its age, behind every other conversation's, so the room the tier needs takes the longest unheard conversation's boundaries, its first among them, before any of one still going.
    std::stable_partition(bounds_.begin(), bounds_.end(), [&](const Boundary& o) {
        return !(o.tokens.size() < n && std::equal(o.tokens.begin(), o.tokens.end(), tokens.begin()));
    });
    // The entry is made before the copy, so nothing allocates between a copy enqueued and its entry: a throw before the copy, or from it, which releases what it took, leaves no entry.
    bounds_.emplace_back();
    Boundary& b = bounds_.back();
    try {
        b.id = ++donor_ids_;
        b.tokens.assign(tokens.begin(), tokens.begin() + (std::ptrdiff_t)n);
        b.classes = clip(classes, n);
        quieted()->save_host(seq, n, b.state, host_cap_ + parked_held(), false);
    } catch (const std::exception& e) {
        bounds_.pop_back();
        moved(Move::boundary_not_kept, "server: a message boundary at %zu tokens was not kept in host memory (%s)\n", n, e.what());
        return;
    }
    host_held_ += b.state.held;
    host_moved_ += b.state.bytes;
    moved(Move::boundary_kept, "server: a message boundary at %zu tokens kept in host memory, %.1f MiB\n", n, (double)b.state.bytes / (1 << 20));
    if (disk_) {
        refresh_disk();
        if (b.disk) disk_->renew(b.disk, true);
        write_ahead();
    }
}

// The boundary `id`, or none.
inline Scheduler::Boundary* Scheduler::find_bound(uint64_t id) {
    for (Boundary& b : bounds_)
        if (b.id == id) return &b;
    return nullptr;
}

// The oldest boundary but the one a request being admitted has pinned out of host memory, the one being written to disk only when no other is left (written_soon); false when there is none.
// Under the lock.
inline bool Scheduler::drop_oldest_bound() {
    for (const bool writing : {false, true})
        for (size_t i = 0; i < bounds_.size(); ++i)
            if (bounds_[i].id != pinned_bound_ && written_soon(true, bounds_[i].id) == writing) {
                drop_bound(i);
                return true;
            }
    return false;
}

// Boundary i out of host memory.
// Under the lock.
inline void Scheduler::drop_bound(size_t i) {
    const bool gone = !bounds_[i].disk && disk_ && !stopping_;
    if (!bounds_[i].disk) lost();
    // A state whose file is whole loses nothing, so it is counted and not given as a loss.
    if (bounds_[i].disk) moved(Move::boundary_dropped);
    else this->gone(bounds_[i].tokens, gone ? Move::boundary_lost : Move::boundary_dropped, bounds_[i].state.length, bounds_[i].state.length, gone ? "server: cache: the state at %zu tokens left host memory with no file\n" : nullptr,
               bounds_[i].state.length);
    host_held_ -= bounds_[i].state.held;
    release_entry(true, bounds_[i].id, bounds_[i].state);
    bounds_.erase(bounds_.begin() + (std::ptrdiff_t)i);
}

// The boundary r can use that shares the most whole blocks with its history, by best_donor's rule at the boundary's position; bounds_.size() when none does.
// A boundary is a state alone, so it is one r can use only where a donor or a host entry holds its rows; a longer one without them does not hide a shorter one with them.
inline size_t Scheduler::best_bound(const Request& r, size_t& tokens) const {
    size_t best = bounds_.size();
    tokens = 0;
    for (size_t i = 0; i < bounds_.size(); ++i) {
        const Boundary& b = bounds_[i];
        const size_t len = b.state.length;
        const size_t n = shareable(r, b.tokens, b.classes, len, std::optional<size_t>(len));
        if (n <= tokens) continue;
        const bool rows = std::any_of(donors_.begin(), donors_.end(), [&](const Donor& d) { return holds(b, d.tokens, d.classes, d.seq.length()); }) ||
                          std::any_of(host_.begin(), host_.end(), [&](const HostDonor& h) { return holds(b, h.tokens, h.classes, h.history.length); });
        if (rows) { tokens = n; best = i; }
    }
    return best;
}

// Whether a history of tokens `t` computed as `classes` record, `held` of them on hand, holds boundary b's rows: its tokens and row classes below b's position.
inline bool Scheduler::holds(const Boundary& b, const std::vector<uint32_t>& t, const std::vector<RowClass>& classes, size_t held) const {
    return holds(b.tokens, b.classes, b.state.length, t, classes, held);
}

// The same for the rows below `n` of tokens `bt` computed as `bc` record.
inline bool Scheduler::holds(const std::vector<uint32_t>& bt, const std::vector<RowClass>& bc, size_t n, const std::vector<uint32_t>& t, const std::vector<RowClass>& classes,
           size_t held) const {
    return held >= n && t.size() >= n && bt.size() >= n && std::equal(bt.begin(), bt.begin() + (std::ptrdiff_t)n, t.begin()) && alike(bc, classes, n) == n;
}

// The host donor sharing the most whole blocks with r's history, by best_donor's rule; host_.size() when none shares a block.
inline size_t Scheduler::best_host(const Request& r, size_t& tokens) const {
    size_t best = host_.size();
    tokens = 0;
    for (size_t i = 0; i < host_.size(); ++i) {
        const size_t len = host_[i].history.length;
        const size_t n = shareable(r, host_[i].tokens, host_[i].classes, len, model_.keeps_state() ? std::optional<size_t>(len) : std::nullopt);
        if (n > tokens) { tokens = n; best = i; }
    }
    return best;
}

// Donor d's history copied to host memory as it leaves the devices (enqueued, not waited for), its whole blocks up to its checkpoint on a model that keeps a state, where the tier's cap and the host's free memory allow.
// Superseded host donors and then the oldest go first, but for a donor whose conversation did not come back (below); under the lock.
inline size_t Scheduler::copy_length(const Donor& d) const {
    const size_t bt = model_.kv_block_tokens();
    size_t n = std::min(d.seq.length(), d.tokens.size()) / bt * bt;
    if (model_.keeps_state()) {
        const std::optional<size_t> kept = model_.checkpoint(d.seq);
        n = kept && *kept <= n && *kept % bt == 0 ? *kept : 0;
    }
    return n;
}

// What write_back does with a copy of `n` tokens, the one owner of whether a copy is kept: the room it takes, in the order write_back releases it, and host memory for the slabs it takes anew.
// An unreturned donor takes only the room of superseded or unreturned entries until the tier has refused as many in a row as it holds (docs/src/server.md).
inline Scheduler::HostRoom Scheduler::host_room(size_t n, bool back) const {
    HostRoom plan;
    const size_t bytes = model_.host_bytes(n);
    if (bytes > host_cap_) {
        plan.why = HostRoom::Why::larger;
        return plan;
    }
    size_t held = host_held_, left = host_.size(), parked = parked_held();
    std::vector<char> gone_bound(bounds_.size(), 0), gone_copy(host_.size(), 0);
    std::vector<size_t> freed;
    // An entry released leaves its slabs idle for the copy, or parked beside the tier's limit while its write stops (release_entry).
    const auto release = [&](const infer::HostHistory& h, bool bound, uint64_t id) {
        held -= std::min(held, h.held);
        plan.drops.push_back({bound, id});
        if (parks(bound, id)) {
            parked += h.held;
            return;
        }
        if (freed.size() < h.slabs.size()) freed.resize(h.slabs.size(), 0);
        for (size_t k = 0; k < h.slabs.size(); ++k) freed[k] += h.slabs[k].size();
    };
    const auto bound = [&](size_t i) { gone_bound[i] = 1; release(bounds_[i].state, true, bounds_[i].id); };
    const auto copy = [&](size_t i) { gone_copy[i] = 1; --left; release(host_[i].history, false, host_[i].id); };
    const auto short_of = [&] { return held + bytes > host_cap_; };
    for (size_t i = 0; i < host_.size() && host_[i].superseded && short_of(); ++i) copy(i);
    for (size_t i = 0; i < bounds_.size() && short_of(); ++i)
        if (bounds_[i].disk && bounds_[i].id != pinned_bound_) bound(i);
    for (size_t i = 0; i < host_.size() && short_of(); ++i)
        if (!gone_copy[i] && host_[i].disk && host_[i].id != promoting_ && !read_source(host_[i].id)) copy(i);
    for (const bool writing : {false, true})
        for (size_t i = 0; i < bounds_.size() && short_of(); ++i)
            if (!gone_bound[i] && bounds_[i].id != pinned_bound_ && written_soon(true, bounds_[i].id) == writing) bound(i);
    if (!back) {
        size_t room = host_cap_ - std::min(host_cap_, held);
        for (size_t i = 0; i < host_.size(); ++i)
            if (!gone_copy[i] && (host_[i].superseded || !host_[i].back)) room += host_[i].history.held;
        if (room < bytes && host_refused_ < left) {
            plan.why = HostRoom::Why::unreturned;
            return plan;
        }
        for (const bool writing : {false, true})
            for (size_t i = 0; i < host_.size() && short_of(); ++i)
                if (!gone_copy[i] && (host_[i].superseded || !host_[i].back) && written_soon(false, host_[i].id) == writing) copy(i);
    }
    for (const bool writing : {false, true})
        for (size_t i = 0; i < host_.size() && short_of(); ++i)
            if (!gone_copy[i] && host_[i].id != promoting_ && !read_source(host_[i].id) && written_soon(false, host_[i].id) == writing) copy(i);
    // Still short once every entry it may release is gone (a pinned boundary, the copy being promoted and a disk read's source stay), the copy would fail its slabs' limit.
    if (short_of()) {
        plan.why = HostRoom::Why::room;
        return plan;
    }
    const infer::Model::HostFit fit = model_.host_copy_fits(n, host_cap_ + parked, freed);
    if (fit == infer::Model::HostFit::limit) plan.why = HostRoom::Why::room;
    else if (fit == infer::Model::HostFit::memory) plan.why = HostRoom::Why::memory;
    return plan;
}

inline void Scheduler::write_back(Donor& d) {
    if (!host_cap_ || d.superseded) return;
    const size_t n = copy_length(d);
    if (!n) return;
    // A donor that leaves with no copy is lost unless its files are whole on disk.
    const auto uncopied = [&] {
        if (disk_ && !disk_->on_disk(digests(d), model_.kv_block_tokens(), n, model_.keeps_state())) lost();
    };
    for (size_t i = 0; i < host_.size(); ++i) {
        const HostDonor& h = host_[i];
        const bool same = h.history.length == n && std::equal(h.tokens.begin(), h.tokens.end(), d.tokens.begin()) &&
                          alike(h.classes, d.classes, n) == n;
        if ((d.on_host && h.id == d.on_host) || same) {
            // The entry now stands for this donor, which is not superseded, so it takes the donor's standing too.
            host_[i].superseded = false;
            host_[i].back = host_[i].back || d.back;
            host_[i].used = std::filesystem::file_time_type::clock::now();
            host_refused_ = 0;
            if (host_[i].disk) disk_->renew(host_[i].disk, host_[i].back);
            std::rotate(host_.begin() + (std::ptrdiff_t)i, host_.begin() + (std::ptrdiff_t)i + 1, host_.end());
            return;
        }
    }
    const HostRoom plan = host_room(n, d.back);
    if (plan.why == HostRoom::Why::larger) {
        // Too large for the host tier: it is kept only where its files are already whole on disk, and otherwise lost, which is counted and said.
        uncopied();
        const bool whole = disk_ && disk_->on_disk(digests(d), model_.kv_block_tokens(), n, model_.keeps_state());
        // Whole on disk, a conversation that came back is read back to the devices by a job once room is spare, so its next turn waits for no read.
        if (whole && d.back && !stopping_) restore_later(d.tokens, n);
        if (whole) moved(Move::not_to_host);
        else gone(d.tokens, Move::not_to_host, held_below(d.tokens, d.classes, n), n, "server: a history of %zu tokens left the devices with no copy: %.1f MiB is more than the host tier holds%s\n", n, (double)model_.host_bytes(n) / (1 << 20), disk_ ? " and its files were not yet whole on disk" : "");
        return;
    }
    // What the plan releases goes first, as far as the refusal of an unreturned donor where it is refused; a copy short of room or of the host's memory releases nothing.
    if (plan.why != HostRoom::Why::memory && plan.why != HostRoom::Why::room)
        for (const auto& [bound, id] : plan.drops) {
            if (bound) {
                for (size_t i = 0; i < bounds_.size(); ++i)
                    if (bounds_[i].id == id) { drop_bound(i); break; }
            } else {
                for (size_t i = 0; i < host_.size(); ++i)
                    if (host_[i].id == id) { drop_host(i); break; }
            }
        }
    if (plan.why == HostRoom::Why::unreturned) {
        ++host_refused_;
        uncopied();
        gone(d.tokens, Move::not_to_host, held_below(d.tokens, d.classes, n), n, "server: a donor of %zu tokens was not kept in host memory (the copies there are of conversations that came back, and its own did not)\n", n);
        return;
    }
    if (plan.why == HostRoom::Why::room) {
        uncopied();
        gone(d.tokens, Move::not_to_host, held_below(d.tokens, d.classes, n), n, "server: a donor of %zu tokens was not kept in host memory (the tier's room is held by entries in use)\n", n);
        return;
    }
    if (plan.why == HostRoom::Why::memory) {
        uncopied();
        gone(d.tokens, Move::not_to_host, held_below(d.tokens, d.classes, n), n, "server: a donor of %zu tokens was not kept in host memory (the host has no memory for its slabs beside the reserve it keeps)\n", n);
        return;
    }
    // The entry is made before the copy, so nothing allocates between a copy enqueued and its entry.
    host_.emplace_back();
    HostDonor& h = host_.back();
    const Clock::time_point start = Clock::now();
    try {
        h.id = d.id ? d.id : ++donor_ids_;
        h.back = d.back;
        h.tokens.assign(d.tokens.begin(), d.tokens.begin() + (std::ptrdiff_t)n);
        h.classes = clip(d.classes, n);
        quieted()->save_host(d.seq, n, h.history, host_cap_ + parked_held());
    } catch (const std::exception& e) {
        host_.pop_back();
        gone(d.tokens, Move::not_to_host, held_below(d.tokens, d.classes, n), n, "server: a donor of %zu tokens was not kept in host memory (%s)\n", n, e.what());
        uncopied();
        return;
    }
    host_held_ += h.history.held;
    host_moved_ += h.history.bytes;
    moved(Move::to_host, "server: a donor of %zu tokens kept in host memory, %.1f MiB, its copy enqueued in %.1f ms\n", n, (double)h.history.bytes / (1 << 20), ms_since(start));
    host_refused_ = 0;
    if (disk_) {
        refresh_disk();
        if (h.disk) disk_->renew(h.disk, h.back);
        write_ahead();
    }
}

// A job for the first `n` of `tokens`, which forks what holds them and reads the rest from disk into spare room (enter), none while a restore job's donor takes the slot (restoring_).
// follow_up takes it in with the round's other new jobs, since a donor leaves the devices while jobs_ is being walked; under the lock.
inline void Scheduler::restore_later(const std::vector<uint32_t>& tokens, size_t n) {
    if (restoring_) return;
    auto j = std::make_shared<Request>(std::vector<uint32_t>(tokens.begin(), tokens.begin() + (std::ptrdiff_t)n), SampleParams{});
    j->job_ = true;
    j->whole_ = true;
    restores_.push_back(std::move(j));
}

// Whether the entry `id` is the one whose write to disk is in flight: room takes it last among those it may take, since dropping it cancels the write (docs/DISK-TIER.md, Demotion).
inline bool Scheduler::written_soon(bool boundary, uint64_t id) const { return disk_ && disk_->writing_from(id, boundary) && !disk_->write_stopped(); }

// The oldest host donor out of host memory, the one being written to disk only when it is the last; false when no other is left.
// Room passes by the copy being promoted, whose evictions make this room; under the lock.
inline bool Scheduler::drop_oldest_host() {
    for (const bool writing : {false, true})
        for (size_t i = 0; i < host_.size(); ++i)
            if (host_[i].id != promoting_ && !read_source(host_[i].id) && written_soon(false, host_[i].id) == writing) {
                drop_host(i);
                return true;
            }
    return false;
}

// Host donor i out of host memory.
// Under the lock.
inline void Scheduler::drop_host(size_t i) {
    // A superseded copy is no loss, a newer one of its conversation standing for it.
    const bool gone = !host_[i].disk && !host_[i].superseded;
    if (gone) lost();
    if (gone && disk_ && !stopping_) this->gone(host_[i].tokens, Move::host_lost, 0, host_[i].history.length, "server: cache: a copy of %zu tokens left host memory with no file\n", host_[i].history.length);
    // Without a disk tier the copy is a restore point gone all the same, given with its history so a loss is told from work never done.
    else if (gone && !stopping_) this->gone(host_[i].tokens, Move::host_dropped, 0, host_[i].history.length);
    else moved(Move::host_dropped);
    host_held_ -= host_[i].history.held;
    release_entry(false, host_[i].id, host_[i].history);
    host_.erase(host_.begin() + (std::ptrdiff_t)i);
}

// The one place a move of the cache between its places is said and counted (docs/SERVER.md, the cache's moves): `kind`, with its line, or none for a move its request's own line says.
template <class... A>
inline void Scheduler::moved(Move kind, const char* line, A... a) {
    ++moves_[(size_t)kind];
    if (!line) return;
    if constexpr (sizeof...(A) == 0) std::fputs(line, stderr);
    else std::fprintf(stderr, line, a...);
}

// The most rows below `n` of `tokens`, in whole blocks, that a donor or a host entry holds: what is left in memory of a history whose other holder goes.
inline size_t Scheduler::held_below(const std::vector<uint32_t>& tokens, const std::vector<RowClass>& classes, size_t n) {
    const size_t bt = model_.kv_block_tokens();
    for (size_t m = n / bt * bt; m; m -= bt)
        if (rows_held(tokens, classes, m)) return m;
    return 0;
}

// Rows [first, tokens) of `history` leaving memory with their last holder, or a state of it at `tokens`: moved, and first given to a test that tells whose rows they were (on_move).
template <class... A>
inline void Scheduler::gone(const std::vector<uint32_t>& history, Move kind, size_t first, size_t tokens, const char* line, A... a) {
    if (on_move) on_move(Moved{kind, first, tokens, &history});
    moved(kind, line, a...);
}

// One restore point, a history or a boundary's state a later request could have forked, released while no file holds it: counted beside a disk tier while the server runs (Stats::lost_before_written).
// Called where the entry is released (drop_host, drop_bound) or a donor leaves with no copy made (write_back); under the lock.
inline void Scheduler::lost() {
    if (disk_ && !stopping_) ++lost_before_written_;
}

// Whether a host entry released now parks its slabs, its disk write being in flight (release_entry, host_room).
inline bool Scheduler::parks(bool boundary, uint64_t id) const { return disk_ && disk_->writing_from(id, boundary); }

// A host entry's slabs back to the model, or, for the entry whose disk write is in flight, held until the write has stopped, which a cancel asks of it; the room counts free at once.
// Under the lock.

inline void Scheduler::release_entry(bool boundary, uint64_t id, infer::HostHistory& h) {
    if (parks(boundary, id)) {
        disk_->cancel_write();
        disk_parked_.push_back(std::move(h));
        h = infer::HostHistory{};
        return;
    }
    model_.release_host(h);
}

// The slabs held beside the host tier's copies, which leave its limit for them: an entry's dropped while its write stops, which count free.
inline size_t Scheduler::parked_held() const {
    size_t n = 0;
    for (const infer::HostHistory& h : disk_parked_) n += h.held;
    if (disk_) n += disk_->held();
    return n;
}

// Host entries whose files are on disk released, the oldest boundaries and then the oldest copies, until `bytes` more fit the host tier; nothing is lost, the files keeping them.
// Under the lock.
inline void Scheduler::release_written(size_t bytes) {
    for (size_t i = 0; i < bounds_.size() && host_held_ + bytes > host_cap_;) {
        if (bounds_[i].disk && bounds_[i].id != pinned_bound_) drop_bound(i);
        else ++i;
    }
    for (size_t i = 0; i < host_.size() && host_held_ + bytes > host_cap_;) {
        if (host_[i].disk && host_[i].id != promoting_ && !read_source(host_[i].id)) drop_host(i);
        else ++i;
    }
}

// Whether host entry `id` holds the rows a state being read alone from disk is for, so no room releases it until that read ends (start_read).
// Under the lock.
inline bool Scheduler::read_source(uint64_t id) const {
    return id == reading_for_ || std::any_of(disk_reads_.begin(), disk_reads_.end(), [&](const DiskRead& rd) { return rd.source == id; });
}

// The digests of a history's whole blocks (DiskIndex::digests), each stretch of its rows under the key the model gives its extent (Model::row_class), which is what makes two histories' rows the same bits.
inline std::vector<std::string> Scheduler::digests_of(const uint32_t* tokens, size_t n, const std::vector<RowClass>& classes) const {
    std::vector<ClassRun> runs;
    for (const RowClass& c : classes) runs.push_back(ClassRun{c.end, model_.row_class(c.extent)});
    if (runs.empty()) return {DiskIndex::root()};
    runs.back().end = std::numeric_limits<size_t>::max();
    return DiskIndex::digests(tokens, n, runs, model_.kv_block_tokens());
}

// An entry's own, kept with it.
template <class Entry>
inline const std::vector<std::string>& Scheduler::digests(const Entry& e, size_t n) const {
    if (e.digests.empty()) e.digests = digests_of(e.tokens.data(), std::min(e.tokens.size(), n), e.classes);
    return e.digests;
}

inline const std::vector<std::string>& Scheduler::digests(const HostDonor& h) const { return digests(h, h.history.length); }

inline const std::vector<std::string>& Scheduler::digests(const Boundary& b) const { return digests(b, b.state.length); }

inline const std::vector<std::string>& Scheduler::digests(const Donor& d) const { return digests(d, d.seq.length()); }

// Which host entries are whole on disk, looked at again after the index gained or lost a file.
// Under the lock.
inline void Scheduler::refresh_disk() {
    const size_t bt = model_.kv_block_tokens();
    for (HostDonor& h : host_) h.disk = disk_->on_disk(digests(h), bt, h.history.length, model_.keeps_state());
    for (Boundary& b : bounds_) b.disk = disk_->state_on_disk(digests(b), bt, b.state.length);
}

// After a call that may have changed the index: the entries' standing and the figures stats gives.
inline void Scheduler::sync_disk() {
    if (disk_->take_changed()) refresh_disk();
    const DiskTier::Figures was = disk_now_;
    disk_now_ = disk_->figures();
    // The tier's own outcomes since the last look, said here and never from its I/O threads: writes published or failed, and files that left the index with the tier's reason.
    for (uint64_t k = was.whole; k < disk_now_.whole; ++k) moved(Move::write_whole);
    if (disk_now_.failed > was.failed) moved(Move::write_failed, "server: cache: %llu writes to disk failed\n", (unsigned long long)(disk_now_.failed - was.failed));
    for (size_t g = 0; g < disk_now_.gone.size(); ++g)
        if (const size_t n = disk_now_.gone[g] - was.gone[g]) {
            for (size_t k = 1; k < n; ++k) moved(Move::disk_removed);
            moved(Move::disk_removed, "server: cache: %zu files left the disk (%s)\n", n, DiskTier::gone_name((DiskTier::Gone)g));
        }
}

// The disk tier's part of a round: what the store finished since the last round taken in, what a server before left adopted, files past the age limit deleted, reads started for waiting requests, and the next write started.
// Under the lock.
inline void Scheduler::disk_round(bool idle) {
    disk_wake_ = false;
    if (!disk_) return;
    settle_disk();
    if (!disk_measured_ && disk_->readable()) {
        disk_measured_ = true;
        disk_->measure(*quieted());
        quieted()->trim_host(host_cap_ + parked_held());
    }
    disk_->adopt();
    const auto now = std::chrono::steady_clock::now();
    if (const uint64_t age = disk_->max_age(); age && now >= next_expiry_) {
        // Checked every minute, or each quarter of the limit if that is shorter.
        next_expiry_ = now + std::chrono::seconds(std::max<uint64_t>(1, std::min<uint64_t>(60, age / 4)));
        disk_->expire();
    }
    sync_disk();
    stream_round();
    prefetch();
    write_ahead();
    // A donor the host tier cannot keep is written from its turn's end on, with or without --disk-cache-keep and while requests run, a piece at a time (docs/DISK-TIER.md, Demotion).
    if (disk_->can_write()) flush_one(turn_tried_, true);
    // Under --disk-cache-keep a server idle for kDiskIdle (`idle`, from run) writes what a stop would have to, one file at a time, so a stop finds little left (docs/DISK-TIER.md, Keeping entries across a restart).
    if (idle && queue_.empty() && paused_.empty() && disk_->can_write() && !flush_one(idle_tried_)) disk_->say_written();
    sync_disk();
}

// The calls the store has finished: a write's file into the index and the slabs of an entry dropped while written released, and each read that ended made a host entry or given up.
// Under the lock.
inline void Scheduler::settle_disk() {
    const DiskTier::Settled done = disk_->settle(model_);
    if (done.wrote) {
        for (infer::HostHistory& h : disk_parked_) model_.release_host(h);
        disk_parked_.clear();
    }
    for (const auto& read : done.reads) finish_read(read.first, read.second);
    // The slabs the tier's own calls took beyond the host tier go back to the host.
    if (done.wrote || !done.reads.empty()) quieted()->trim_host(host_cap_ + parked_held());
    sync_disk();
}

// At a clean exit under --disk-cache-keep, what memory holds goes to disk for the next server within a stated bound, the most valuable first (the reverse of the order room takes entries in), device donors copied to host memory first.
// A write still running at the limit is cancelled, so only whole entries stay, each keeping its last use; on the scheduler thread, without the lock.
inline void Scheduler::flush_disk() {
    std::unique_lock<std::mutex> lk(m_);
    settle_disk();
    // The bound is stated before the first write: kDiskFlush, or half as long again as what is left would take at the store's measured rate, plus five seconds for the copies off the devices.
    const uint64_t bytes = unwritten_bytes();
    const double rate = disk_->write_rate();
    const auto bound = std::max<std::chrono::steady_clock::duration>(kDiskFlush, rate > 0 ? std::chrono::seconds((long long)(1.5 * (double)bytes / rate) + 5) : kDiskFlush);
    const auto until = std::chrono::steady_clock::now() + bound;
    std::fprintf(stderr, "server: writing %.1f MiB to disk for the next server, within %lld s\n", (double)bytes / (1 << 20), (long long)std::chrono::duration_cast<std::chrono::seconds>(bound).count());
    std::vector<uint64_t> tried;
    const char* why = "";
    for (;;) {
        settle_disk();
        if (!disk_->writing()) { why = ", writing having stopped"; break; }
        if (disk_->can_write() && !flush_one(tried)) { why = ", refused room on disk or host memory"; break; }
        if (!cv_.wait_until(lk, until, [&] { return disk_wake_; })) { why = ", the bound reached"; break; }
        disk_wake_ = false;
    }
    disk_->cancel_write();
    const uint64_t left = unwritten_bytes();
    std::fprintf(stderr, "server: %zu entries kept on disk for the next server, %.1f MiB not written%s\n", disk_->index().files().size(), (double)left / (1 << 20), left ? why : "");
    // Each history left out, and why: superseded ones stand on a newer copy of their conversation, which the line before counts.
    const size_t bt = model_.kv_block_tokens();
    const auto said = [&](const std::vector<std::string>& d, size_t n, const char* reason, const char* where) {
        if (!n || !disk_->unwritten(d, bt, n, model_.keeps_state(), true)) return;
        std::fprintf(stderr, "server: a history of %zu tokens %s not kept for the next server (%s)\n", n, where, reason);
    };
    const char* flush_reason = why[0] ? why + 2 : "not written";
    for (const Donor& d : donors_)
        said(digests(d), std::min(d.seq.length(), d.tokens.size()) / bt * bt,
             d.superseded ? "superseded by a newer copy of its conversation" : !copy_length(d) ? "its checkpoint no copy can take" : flush_reason, "on the devices");
    for (const HostDonor& h : host_) said(digests(h), h.history.length, h.superseded ? "superseded by a newer copy of its conversation" : flush_reason, "in host memory");
    sync_disk();
}

// The bytes a keep flush has still to write: of every history a device or host memory holds, the segments not on disk and its state, a history two entries hold counted once.
// A device donor counts by its whole blocks, not by what a copy can take, so a history no copy can take shows as not written; under the lock.
inline uint64_t Scheduler::unwritten_bytes() const {
    const size_t bt = model_.kv_block_tokens();
    uint64_t bytes = 0;
    std::vector<std::string> seen;
    const auto add = [&](const std::vector<std::string>& d, size_t n, bool blocks) {
        if (!n || std::find(seen.begin(), seen.end(), d[n / bt]) != seen.end()) return;
        seen.push_back(d[n / bt]);
        bytes += disk_->unwritten(d, bt, n, model_.keeps_state(), blocks);
    };
    for (const Donor& d : donors_)
        if (!d.superseded) add(digests(d), std::min(d.seq.length(), d.tokens.size()) / bt * bt, true);
    for (const HostDonor& h : host_)
        if (!h.superseded) add(digests(h), h.history.length, true);
    for (const Boundary& b : bounds_) add(digests(b), b.state.length, false);
    return bytes;
}

// Starts the next file of a host entry's history that is not on disk, out of the entry's own slabs; a boundary's state once its history's blocks are there.
inline bool Scheduler::write_from(const HostDonor& h) { return started(disk_->write_next(model_, digests(h), h.history.length, h.back, h.used, &h.history, nullptr, h.id, false)); }

inline bool Scheduler::write_from(const Boundary& b) { return started(disk_->write_next(model_, digests(b), b.state.length, true, b.used, &b.state, nullptr, b.id, true)); }

inline bool Scheduler::started(DiskTier::Wrote w) {
    sync_disk();
    return w == DiskTier::Wrote::started;
}

// The next write of what memory holds, for flush_disk and for an idle server under --disk-cache-keep: true once one is in flight, false when nothing is left to write.
// Most valuable first (docs/DISK-TIER.md), a device donor's missing blocks and state copied off the devices for it and no more; a donor refused room is in `tried`, not asked again that flush or idle period; under the lock.
inline bool Scheduler::flush_one(std::vector<uint64_t>& tried, bool beyond) {
    tried.erase(std::remove_if(tried.begin(), tried.end(), [&](uint64_t id) { return std::none_of(donors_.begin(), donors_.end(), [&](const Donor& d) { return d.id == id; }); }), tried.end());
    for (const bool back : {true, false}) {
        for (size_t k = donors_.size(); k-- > 0;) {
            Donor& d = donors_[k];
            const size_t n = copy_length(d);
            if (!n || d.superseded || d.back != back || std::find(tried.begin(), tried.end(), d.id) != tried.end() || (beyond && !beyond_host(n))) continue;
            // Beyond the host tier, the copy off the devices goes a piece at a time: background work under the budget (BackgroundGrant), but whole pieces while a request waits for this donor's room (enter).
            size_t most = std::numeric_limits<size_t>::max();
            double left = BackgroundGrant::kUnbounded;
            const bool foreground = beyond && awaited_ == d.id;
            if (beyond && !foreground) {
                left = write_left_;
                const size_t bt = model_.kv_block_tokens(), blocks = BackgroundGrant::units(left, (n + bt - 1) / bt, copy_block_ms_);
                if (!blocks) return false;
                most = blocks * bt;
            }
            const Clock::time_point start = Clock::now();
            const DiskTier::Wrote w = disk_->write_next(*quieted(), digests(d), n, d.back, std::filesystem::file_time_type::clock::now(), nullptr, &d.seq, 0, false, most);
            if (started(w)) {
                if (beyond) {
                    const double ms = ms_since(start);
                    if (const size_t rows = disk_->pending_rows()) copy_block_ms_ = forget(copy_block_ms_, ms * (double)model_.kv_block_tokens() / (double)rows);
                    if (!foreground && demand_ != BackgroundGrant::Demand::none) {
                        write_spent_ms_ += ms;
                        write_spent_left_ = left;
                        write_left_ = std::max(0.0, write_left_ - ms);
                    }
                }
                return true;
            }
            if (beyond && w == DiskTier::Wrote::refused)
                std::fprintf(stderr, "server: cache: a history of %zu tokens the host tier cannot keep was not written as its turn ended (refused room on disk or host memory)\n", n);
            if (w == DiskTier::Wrote::refused || (beyond && w == DiskTier::Wrote::nothing)) tried.push_back(d.id);
        }
        // With `beyond`, only the device donors the host tier cannot keep (beyond_host), each asked until it is whole on disk or refused, then left in `tried` while it lives.
        if (beyond) continue;
        for (size_t i = host_.size(); i-- > 0;)
            if (!host_[i].disk && !host_[i].superseded && host_[i].back == back && write_from(host_[i])) return true;
        for (size_t i = bounds_.size(); back && i-- > 0;)
            if (!bounds_[i].disk && write_from(bounds_[i])) return true;
    }
    return false;
}

// Whether a waiting request has something for the round: no read or room to wait for, its wait past its bound, or its client gone; the store's finished calls wake the round otherwise.
inline bool Scheduler::ready(const std::deque<std::shared_ptr<Request>>& q) {
    const auto now = std::chrono::steady_clock::now();
    return std::any_of(q.begin(), q.end(), [&](const std::shared_ptr<Request>& r) {
        return ((!r->disk_wait_ || now >= r->disk_until_) && (!r->room_wait_ || now >= r->room_until_)) || r->cancel_.load();
    });
}

// When the loop must wake for the disk tier: the earliest bound of a request waiting for a read, the next check of the entries' age, and the start of an idle server's writes.
inline std::optional<std::chrono::steady_clock::time_point> Scheduler::next_disk_bound() const {
    std::optional<std::chrono::steady_clock::time_point> t;
    // A streamed read's copies retire with no event, so the loop looks again a millisecond on while one runs (stream_round).
    if (!streams_.empty()) t = std::chrono::steady_clock::now() + std::chrono::milliseconds(1);
    if (disk_ && disk_->max_age() && disk_now_.files && (!t || next_expiry_ < *t)) t = next_expiry_;
    for (const auto* q : {&queue_, &paused_, &jobs_})
        for (const auto& r : *q)
            if (r->disk_wait_ && r->disk_until_ != std::chrono::steady_clock::time_point::max() && (!t || r->disk_until_ < *t)) t = r->disk_until_;
    // Under keep, requests wait for the disk tier's start at most the wait's bound from the server's (reading).
    if (disk_ && disk_->keeps() && !disk_->settled()) {
        const auto bound = epoch_ + wait_bound_;
        if (!t || bound < *t) t = bound;
    }
    // A request waiting for a donor's write or for room for a restore ends its wait at its bound (waits_for_room).
    for (const auto* q : {&queue_, &paused_})
        for (const auto& r : *q)
            if (r->room_wait_ && (!t || r->room_until_ < *t)) t = r->room_until_;
    // The idle moment bounds the wait until a round has been idle, also once it is past, which a round begun before it and ended after it leaves it; after that round it bounds nothing, so the loop does not spin.
    if (idle_at_ && !idled_ && (!t || *idle_at_ < *t)) t = idle_at_;
    // And the next look at the host's memory, while there is a host tier to give slabs back from.
    if (host_cap_ && (!t || host_check_ < *t)) t = host_check_;
    return t;
}

// Whether r waits for its read; past its bound it stops waiting and computes what the read would have given.
// Under the lock.
inline bool Scheduler::reading(Request& r) {
    // Under keep a request's first admission waits for the disk tier's start, at most the wait's bound from the server's, so a turn sent as a restarted server listens reads what the server before kept (P4 in docs/CACHE-PLAN.md).
    if (disk_ && disk_->keeps() && r.classes_.empty() && !disk_->settled() && Clock::now() - epoch_ < wait_bound_) return true;
    if (!r.disk_wait_) return false;
    if (std::chrono::steady_clock::now() < r.disk_until_) return true;
    ++disk_wait_timeouts_;
    r.restore_late_.store(std::max(r.restore_late_.load(), r.disk_rows_));
    {
        std::lock_guard<std::mutex> lk(r.m_);
        ++r.restore_.timeouts;
    }
    const uint64_t id = r.disk_wait_;
    end_wait(r);
    // A streamed read past the bound of every request waiting for it is cancelled: its room and slot come back once its readers and copies have ended (stream_round).
    const auto waits = [&] {
        for (auto* q : {&queue_, &paused_, &jobs_})
            for (const auto& o : *q)
                if (o->disk_wait_ == id) return true;
        return false;
    };
    for (Streaming& s : streams_)
        if (s.id == id && !s.ended && !waits()) s.read->cancel();
    return false;
}

// A request's wait for a read ends: its time summed for the server and on the request (Request::Restore::wait_ms).
inline void Scheduler::end_wait(Request& r) {
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - r.disk_since_).count();
    disk_wait_ms_ += ms;
    r.disk_wait_ = 0;
    std::lock_guard<std::mutex> lk(r.m_);
    r.restore_.part(r.restore_.wait_ms, ms, r.first_);
}

inline void Scheduler::count_wait(Request& r) {
    std::lock_guard<std::mutex> lk(r.m_);
    ++r.restore_.waits;
}

// Read `id` finished: a history read whole becomes a host entry of a conversation that came back, its files kept and renewed, and anything else releases its slabs; the requests waiting for it go on.
// Under the lock.
inline void Scheduler::finish_read(uint64_t id, bool ok) {
    size_t i = 0;
    while (i < disk_reads_.size() && disk_reads_[i].id != id) ++i;
    // A proof's files, read by the tier: its request goes on to compare them (prove).
    if (i == disk_reads_.size()) return wake_readers(id);
    DiskRead rd = std::move(disk_reads_[i]);
    disk_reads_.erase(disk_reads_.begin() + (std::ptrdiff_t)i);
    // The read once for the server, and its latency to every request waiting for it, which may share it.
    const double read_ms = ms_since(rd.started);
    if (ok) {
        ++restore_totals_.reads;
        Stats::RestoreTotals::add(restore_totals_.read_ms, restore_totals_.read_largest_ms, read_ms);
        Stats::RestoreTotals::add(restore_totals_.slabs_ms, restore_totals_.slabs_largest_ms, rd.slabs_ms);
    }
    for (auto* q : {&queue_, &paused_, &jobs_})
        for (auto& r : *q)
            if (r->disk_wait_ == id) {
                std::lock_guard<std::mutex> lk(r->m_);
                r->restore_.read_ms += read_ms;
                r->restore_.slabs_ms += rd.slabs_ms;
                r->restore_.failed += !ok;
            }
    if (ok) {
        const size_t length = rd.history.length;
        const double mib = (double)rd.history.bytes / (1 << 20);
        ++disk_hits_;
        disk_read_bytes_ += rd.history.bytes;
        uint64_t* key = nullptr;
        if (rd.state_only) {
            // The copy holding its rows becomes the newest again, so room taken before the request is admitted goes to others first.
            for (size_t h = 0; h < host_.size(); ++h)
                if (host_[h].id == rd.source) {
                    std::rotate(host_.begin() + (std::ptrdiff_t)h, host_.begin() + (std::ptrdiff_t)h + 1, host_.end());
                    break;
                }
            bounds_.emplace_back();
            Boundary& b = bounds_.back();
            b.id = ++donor_ids_;
            b.tokens = std::move(rd.tokens);
            b.classes = std::move(rd.classes);
            b.state = std::move(rd.history);
            key = &b.disk;
        } else {
            host_.emplace_back();
            HostDonor& d = host_.back();
            d.id = ++donor_ids_;
            d.tokens = std::move(rd.tokens);
            d.classes = std::move(rd.classes);
            d.history = std::move(rd.history);
            d.back = true;
            d.from_disk = true;
            key = &d.disk;
        }
        refresh_disk();
        if (*key) disk_->renew(*key, true);
        moved(Move::read_done, "server: %s of %zu tokens read from disk, %.1f MiB\n", rd.state_only ? "a state" : "a history", length, mib);
    } else {
        moved(Move::read_failed, "server: cache: the read of %zu tokens from disk failed; its request reads its prompt\n", rd.history.length);
        host_held_ -= rd.history.held;
        model_.release_host(rd.history);
    }
    wake_readers(id);
}

// The requests waiting for read or proof `id` wait no longer.
inline void Scheduler::wake_readers(uint64_t id) {
    for (auto* q : {&queue_, &paused_, &jobs_})
        for (auto& r : *q)
            if (r->disk_wait_ == id) end_wait(*r);
}

// Proof `id` given up at the tier, and those waiting for it look again.
// Under the lock.
inline void Scheduler::end_proof(uint64_t id) {
    disk_->abandon(id, model_);
    wake_readers(id);
    sync_disk();
}

// Proves another build's files on r's path in the tier's order: r's first block compared, the path restored at its fork, its last block and state at the end compared (docs/DISK-TIER.md, Adopting another build's entries).
// The tier decides each comparison and the files' fate; r computes the rows and copies the block, and a proof that does not end equal leaves it computing its history from the start (unprove).
inline void Scheduler::prove(Request& r) {
    const size_t bt = model_.kv_block_tokens(), most = std::numeric_limits<size_t>::max();
    for (;;) {
        std::optional<DiskTier::Proving> p;
        bool room = true;
        {
            std::lock_guard<std::mutex> lk(m_);
            p = disk_->proving(r.proof_, bt);
            if (p && p->read && !p->first && p->base && r.seq_.length() < p->fork && model_.keeps_state()) room = checkpoint_room_held(0);
        }
        if (!p || !p->read || !room) return unprove(r);
        const size_t at = r.seq_.length(), stop = p->first ? p->first : p->end;
        if (!p->first && p->base && at < p->fork) {
            try { model_.reset(r.seq_); } catch (const std::exception&) {}
            try {
                r.seq_ = quieted()->restore_host(*p->base);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "server: another build's history of %zu tokens was not restored to prove it (%s)\n", p->fork, e.what());
                return unprove(r);
            }
            r.unproved_ = true;
            continue;
        }
        r.prove_at_ = stop;
        r.prove_state_ = !p->first && model_.keeps_state();
        if (at < stop) return;
        if (at > stop && !p->first) return unprove(r);
        infer::HostHistory blocks, state, proved;
        bool equal = false, compared = false;
        try {
            quieted()->save_host_blocks(r.seq_, p->first ? 0 : p->from, stop, blocks, most);
            if (r.prove_state_) quieted()->save_host(r.seq_, stop, state, most, false);
            model_.wait_host(blocks);
            if (state.owner) model_.wait_host(state);
            std::lock_guard<std::mutex> lk(m_);
            equal = disk_->compare(r.proof_, model_, blocks, state, proved);
            compared = true;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "server: a block of %zu tokens was not copied to prove another build's history (%s)\n", stop, e.what());
        }
        model_.release_host(blocks);
        model_.release_host(state);
        if (!compared || !equal) return unprove(r);
        std::lock_guard<std::mutex> lk(m_);
        if (disk_->proving(r.proof_, bt)) continue;
        // Proved: what r restored is its own from here, a prefix it reused.
        if (r.unproved_) {
            r.reused_.store(p->fork);
            ++prefix_hits_;
            prefix_tokens_ += p->fork;
        }
        if (r.boundary_kept_) asked_boundary(r);
        // The path the proof read is a history read from disk from here, a host entry where the tier has the room.
        release_written(proved.held);
        if (proved.owner && host_held_ + proved.held <= host_cap_) {
            host_.emplace_back();
            HostDonor& d = host_.back();
            d.id = ++donor_ids_;
            d.tokens.assign(r.prompt_.begin(), r.prompt_.begin() + (std::ptrdiff_t)proved.length);
            d.classes = clip(r.classes_, proved.length);
            d.back = true;
            host_held_ += proved.held;
            d.history = std::move(proved);
            refresh_disk();
            if (d.disk) disk_->renew(d.disk, true);
        }
        model_.release_host(proved);
        wake_readers(r.proof_);
        sync_disk();
        r.proof_ = r.prove_at_ = 0;
        r.prove_state_ = r.unproved_ = r.boundary_kept_ = false;
        return;
    }
}

// r's proof given up or ended unequal: the rows r restored from the unproved files go, so it computes its history from the start, and the tier lets the proof go.
inline void Scheduler::unprove(Request& r) {
    if (r.unproved_) try { model_.reset(r.seq_); } catch (const std::exception&) {}
    const uint64_t id = r.proof_;
    r.proof_ = r.prove_at_ = 0;
    r.boundary_kept_ = r.boundary_kept_ && !r.unproved_;
    r.prove_state_ = r.unproved_ = false;
    std::lock_guard<std::mutex> lk(m_);
    end_proof(id);
}

// Reads for waiting requests whose history the disk holds more whole blocks of than any history on the devices or in host memory, each request looked for once, at most kDiskReads in flight and max_seqs requests waiting.
// On a model that keeps a state the path ends at a state, the state alone being read where a history on the devices or in host memory holds its rows; a request waits at most as long as recomputing would take, under the lock.
inline void Scheduler::prefetch() {
    if (!disk_now_.files || !disk_->readable()) return;
    const size_t bt = model_.kv_block_tokens();
    size_t waiting = 0;
    for (const auto* q : {&queue_, &paused_, &jobs_})
        for (const auto& r : *q) waiting += r->disk_wait_ != 0;
    // Jobs waiting to start ask the same question as requests, after them, so a job starting again reads its history from disk rather than from row 0 (docs/CACHE-PLAN.md, gap 5).
    for (auto* q : {&paused_, &queue_, &jobs_})
        for (auto& r : *q) {
            if (r->disk_checked_ || r->disk_wait_ || r->cancel_.load() || waiting >= max_seqs_) continue;
            // A job reads only where it could start: into free blocks, so what it reads is on the devices at once and not held while it waits.
            if (r->job_) {
                const std::vector<size_t> need = pools_.blocks_for(r->prompt_.size()), free = room();
                bool spare = true;
                for (size_t k = 0; k < need.size(); ++k) spare = spare && need[k] <= free[k];
                if (!spare) continue;
            }
            // The request's history as it would compute it: its tokens but the last, under its own classes or, at its first admission, the class of its prompt.
            const size_t limit = history_tokens(*r) - 1;
            std::vector<uint32_t> tokens(limit);
            for (size_t k = 0; k < limit; ++k) tokens[k] = *token_ptr(*r, k);
            const std::vector<RowClass> first{RowClass{r->prompt_.size(), r->prompt_.size()}};
            const std::vector<RowClass>& own = r->classes_.empty() ? first : r->classes_;
            const std::vector<std::string> d = digests_of(tokens.data(), limit, own);
            DiskIndex::Path path = disk_->index().path(d, bt, limit, model_.keeps_state());
            const DiskIndex::Path whole = path;
            // A state higher than the blocks on disk reach serves where a history in memory holds its rows.
            const auto alone = model_.keeps_state() ? disk_->index().state_below(d, bt, limit) : std::nullopt;
            if (alone && alone->first > path.length && rows_held(tokens, own, alone->first)) {
                path = DiskIndex::Path{};
                path.length = alone->first;
                path.state = alone->second;
            }
            size_t shared = 0, more = 0;
            best_donor(*r, shared);
            best_host(*r, more);
            shared = std::max(shared, more);
            // A boundary counts only with a history in memory that holds its rows (best_bound); without one it does not stand for what the disk holds.
            best_bound(*r, more);
            shared = std::max(shared, more);
            // Another build's files beyond what this build holds for it: the request proves them at its first admission (prove), waiting for their read, or for the proof already on them, as for any read.
            // On a model that keeps a state the restore and the state at the path's end each take a checkpoint slot, so a server of one proves nothing.
            const bool proves = !r->job_ && r->classes_.empty() && (!model_.keeps_state() || model_.checkpoint_slots() > 1);
            const std::optional<DiskTier::Plan> plan = proves ? disk_->plan(d, bt, limit, model_.keeps_state(), shared) : std::nullopt;
            if (plan && plan->end > path.length) {
                const uint64_t id = plan->busy ? plan->busy : disk_->prove(*quieted(), *plan);
                r->disk_checked_ = !plan->busy;
                if (!id) continue;
                if (!plan->busy) r->proof_ = id;
                const auto now = std::chrono::steady_clock::now();
                r->disk_wait_ = id;
                r->disk_rows_ = plan->end;
                r->disk_since_ = now;
                count_wait(*r);
                r->disk_until_ = read_until(plan->end, now);
                ++disk_waits_;
                ++waiting;
                continue;
            }
            if (path.length <= shared) {
                r->disk_checked_ = true;
                continue;
            }
            // A state alone is read for rows in memory; the path's blocks, where they share more than memory does, are what is read should the host tier have no room for that state beside a host copy holding them.
            uint64_t source = 0;
            const bool state_only = model_.keeps_state() && rows_held(tokens, own, path.length, &source);
            // A history larger than the host tier is read straight into the devices, with no whole copy in host memory (start_read), and a stream already reading the same rows serves this request too.
            const bool streams = !state_only && beyond_host(path.length);
            uint64_t id = 0;
            for (const Streaming& s : streams_)
                if (streams && !s.ended && s.tokens.size() == path.length && std::equal(s.tokens.begin(), s.tokens.end(), tokens.begin()) && alike(s.classes, own, path.length) == path.length) id = s.id;
            size_t reads = disk_reads_.size();
            for (const Streaming& s : streams_) reads += !s.ended;
            if (!id && reads + 1 > kDiskReads) continue;
            // A stream takes its room on the devices as it starts, so one without that room now is looked at again next round, as the request's admission would be.
            if (!id && streams && !stream_room(path.length, r->job_)) continue;
            r->disk_checked_ = true;
            if (!id) id = start_read(path, state_only, tokens, own, *r, source, state_only && whole.length > shared ? &whole : nullptr);
            if (!id) continue;
            // The read started may be the shorter whole path start_read fell back to, or a stream, so the wait is for that read's rows.
            size_t rows = disk_reads_.empty() ? 0 : disk_reads_.back().tokens.size();
            for (const Streaming& s : streams_)
                if (s.id == id) rows = s.tokens.size();
            const auto now = std::chrono::steady_clock::now();
            r->disk_wait_ = id;
            r->disk_rows_ = rows;
            r->disk_since_ = now;
            count_wait(*r);
            r->disk_until_ = read_until(rows, now);
            ++disk_waits_;
            ++waiting;
        }
}

// Until when a read of `rows` rows started at `now` is waited for: the time they take to compute at the measured prompt rate, the wait's constant before a rate is measured, or DiskOptions::read_wait.
inline Scheduler::Clock::time_point Scheduler::read_until(size_t rows, Clock::time_point now) const {
    if (read_wait_) return now + std::chrono::duration_cast<Clock::duration>(*read_wait_);
    if (prompt_ms_row_ > 0) return now + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>((double)rows * prompt_ms_row_));
    return now + wait_bound_;
}

// Starts the read of a path larger than the host tier straight into a history on the devices, with no whole copy in host memory (A20, docs/DISK-TIER.md, Restore): its room taken as a promotion's is, a job's from free blocks alone.
// Its id, 0 where the room, a checkpoint slot or the history is not to be had, which is counted and said as a refused read; under the lock.
inline uint64_t Scheduler::start_stream(const DiskIndex::Path& path, const std::vector<uint32_t>& tokens, const std::vector<RowClass>& classes, const Request& r) {
    const size_t length = path.length;
    const bool spare = r.job_;
    const auto refused = [&](const char* why) {
        ++disk_refused_;
        moved(Move::read_refused, "server: a history of %zu tokens on disk was not read back (%s); its request reads its prompt\n", length, why);
        return uint64_t(0);
    };
    if (!stream_room(length, spare)) return refused("the devices have no room for it");
    // The checkpoint slot its state takes comes first, so no donor goes for its blocks where the slot is not to be had.
    if (model_.keeps_state() && !checkpoint_room_held(0)) return refused("no checkpoint slot for its state");
    const Taken t = make_room(room(), donors_.size(), npos, false, {}, 0, false, pools_.blocks_for(length), gain_of({}));
    if (!t.enough || (spare && !t.donors.empty())) return refused("the devices have no room for it");
    std::vector<size_t> gone = t.donors;
    std::sort(gone.begin(), gone.end(), std::greater<size_t>());
    for (size_t g : gone) drop_donor(g, true);
    // Whatever can throw comes before the read pins its files, so nothing after it needs undoing.
    Streaming s;
    s.tokens.assign(tokens.begin(), tokens.begin() + (std::ptrdiff_t)length);
    s.classes = clip(classes, length);
    s.restore = r.job_ && !r.of_;
    streams_.reserve(streams_.size() + 1);
    try {
        // The read's views of its slots (Backend::wrap_host) are made under the same hold as the history it fills.
        auto held = quieted();
        s.into = std::make_unique<infer::RestoreInto>(held->restore_into(length));
        s.read = disk_->stream(model_, path, *s.into, s.id);
    } catch (const std::exception& e) {
        if (s.into) quieted()->reset(s.into->seq);
        return refused(e.what());
    }
    s.started = Clock::now();
    moved(Move::read_started, "server: a history of %zu tokens read from disk straight into the devices\n", length);
    streams_.push_back(std::move(s));
    return streams_.back().id;
}

// Whether a stream of `length` tokens has its room on the devices now, a job's (`spare`) from free blocks alone; under the lock.
inline bool Scheduler::stream_room(size_t length, bool spare) const {
    const Taken t = make_room(room(), donors_.size(), npos, false, {}, 0, false, pools_.blocks_for(length), gain_of({}));
    return t.enough && (!spare || t.donors.empty());
}

// The streamed reads stepped once a round, the stages held only while full slots are sent: one whose copies have all retired becomes a donor its request forks, as a promoted one does.
// One that failed or was cancelled lets its waiters go at once and gives its room and slot back only once its readers and copies have ended, so nothing reuses blocks still being written; under the lock.
inline void Scheduler::stream_round() {
    // A stream's outcome for the server and for each request waiting for it (Request::Restore): its time from start to its last copy retired is its read, and a whole one completes a restore from disk.
    const auto record = [&](const Streaming& s, bool ok) {
        const double ms = ms_since(s.started);
        if (ok) {
            ++restore_totals_.reads;
            Stats::RestoreTotals::add(restore_totals_.read_ms, restore_totals_.read_largest_ms, ms);
        }
        for (auto* q : {&queue_, &paused_, &jobs_})
            for (auto& r : *q) {
                if (r->disk_wait_ != s.id) continue;
                if (ok && !r->job_) ++restore_totals_.from_disk;
                if (ok) r->first_pass_ = true;
                std::lock_guard<std::mutex> lk(r->m_);
                Request::Restore& x = r->restore_;
                x.read_ms += ms;
                x.failed += !ok;
                if (!ok) continue;
                ++x.restores;
                x.kind = "disk";
                x.tokens += s.into->length;
                x.bytes += s.read->bytes();
                r->first_before_ = r->first_;
            }
    };
    for (size_t i = 0; i < streams_.size();) {
        Streaming& s = streams_[i];
        // The stages are held only to send full slots; giving slots back and asking the state take no hold.
        StreamedRead::State state = s.read->poll();
        if (state == StreamedRead::State::running && s.read->ready()) {
            {
                auto held = quieted();
                s.read->send();
            }
            state = s.read->poll();
        }
        if (state == StreamedRead::State::running) {
            ++i;
            continue;
        }
        if (state == StreamedRead::State::done) {
            quieted()->restored(*s.into, s.read->last());
            record(s, true);
            restoring_ = s.restore;
            while (donors_.size() >= max_seqs_) drop_donor(0, true);
            restoring_ = false;
            Donor d;
            d.id = ++donor_ids_;
            d.seq = std::move(s.into->seq);
            d.tokens = std::move(s.tokens);
            d.classes = std::move(s.classes);
            d.back = true;
            ++disk_hits_;
            disk_read_bytes_ += s.read->bytes();
            disk_->end_stream(s.id, 0);
            moved(Move::read_done, "server: a history of %zu tokens read from disk into the devices, %.1f MiB in %.0f ms\n", d.seq.length(), (double)s.read->bytes() / (1 << 20), ms_since(s.started));
            donors_.push_back(std::move(d));
            const uint64_t id = s.id;
            streams_.erase(streams_.begin() + (std::ptrdiff_t)i);
            wake_readers(id);
            continue;
        }
        if (!s.ended) {
            s.ended = true;
            disk_->end_stream(s.id, s.read->failed_key());
            if (!s.read->why().empty()) moved(Move::read_failed, "server: cache: the read of %zu tokens from disk failed (%s); its request reads its prompt\n", s.into->length, s.read->why().c_str());
            if (!s.read->why().empty()) record(s, false);
            wake_readers(s.id);
        }
        if (!s.read->idle()) {
            ++i;
            continue;
        }
        quieted()->reset(s.into->seq);
        streams_.erase(streams_.begin() + (std::ptrdiff_t)i);
    }
}

// Whether a history on the devices or in host memory holds the first `n` rows of `tokens` computed as `classes` record, so a state alone at `n` can be forked.
// With `source`, a copy in host memory that does becomes the newest and `source` its id, so the room a read takes goes to others first and never to it (read_source).
inline bool Scheduler::rows_held(const std::vector<uint32_t>& tokens, const std::vector<RowClass>& classes, size_t n, uint64_t* source) {
    for (const Donor& d : donors_)
        if (holds(tokens, classes, n, d.tokens, d.classes, d.seq.length())) return true;
    for (size_t h = 0; h < host_.size(); ++h)
        if (holds(tokens, classes, n, host_[h].tokens, host_[h].classes, host_[h].history.length)) {
            if (source) {
                *source = host_[h].id;
                if (host_[h].disk) disk_->renew(host_[h].disk, host_[h].back);
                std::rotate(host_.begin() + (std::ptrdiff_t)h, host_.begin() + (std::ptrdiff_t)h + 1, host_.end());
            }
            return true;
        }
    return false;
}

// Starts the read of a path into slabs it takes host room for as a write-back does, a history of `tokens` computed as `classes` record, for request r; the read's id, or 0 where the room or the memory is not to be had.
// A history larger than the whole host tier is streamed to the devices instead (start_stream); under the lock.
inline uint64_t Scheduler::start_read(const DiskIndex::Path& path, bool state_only, const std::vector<uint32_t>& tokens, const std::vector<RowClass>& classes, const Request& r, uint64_t source, const DiskIndex::Path* whole) {
    const size_t length = path.length;
    const bool blocks = !state_only;
    const size_t bytes = model_.host_bytes(length, blocks);
    if (blocks && bytes > host_cap_) return start_stream(path, tokens, classes, r);
    DiskRead rd;
    rd.state_only = state_only;
    // A state alone is read for the rows host entry `source` holds, 0 for a device donor's: the room it takes is never that entry's, and where there is no other the blocks of `whole` are read with their state instead.
    rd.source = state_only ? source : 0;
    // A read that is not started says why and is counted: its request reads its prompt instead.
    const auto refused = [&](const char* why) {
        ++disk_refused_;
        moved(Move::read_refused, "server: a history of %zu tokens on disk was not read back (%s); its request reads its prompt\n", length, why);
        return uint64_t(0);
    };
    if (bytes > host_cap_) return refused("its state is larger than the host tier");
    reading_for_ = rd.source;
    while (!host_.empty() && host_.front().superseded && host_.front().id != rd.source && host_held_ + bytes > host_cap_) drop_host(0);
    release_written(bytes);
    while (host_held_ + bytes > host_cap_ && drop_oldest_bound()) {}
    while (host_held_ + bytes > host_cap_ && drop_oldest_host()) {}
    reading_for_ = 0;
    if (host_held_ + bytes > host_cap_ && rd.source && whole) return start_read(*whole, false, tokens, classes, r);
    if (host_held_ + bytes > host_cap_) return refused("the host tier has no room it may give");
    const Clock::time_point slabs = Clock::now();
    try {
        quieted()->alloc_host(length, rd.history, host_cap_ + parked_held(), blocks);
    } catch (const std::exception& x) {
        return refused(x.what());
    }
    rd.slabs_ms = ms_since(slabs);
    rd.started = Clock::now();
    rd.id = disk_->read(*quieted(), path, rd.history, state_only);
    if (!rd.id) {
        model_.release_host(rd.history);
        return refused("host memory for a part of a file was not to be had");
    }
    host_held_ += rd.history.held;
    rd.tokens.assign(tokens.begin(), tokens.begin() + (std::ptrdiff_t)length);
    rd.classes = clip(classes, length);
    const uint64_t id = rd.id;
    moved(Move::read_started);
    disk_reads_.push_back(std::move(rd));
    return id;
}

// Once host entries not on disk hold more than three quarters of the host tier, the one it would drop next is written while it stays a host entry: boundaries, oldest first, then copies whose conversations did not come back, then the oldest.
// Room taken later releases entries on disk at once (release_written), so a demotion never waits for a write and writes nothing the host tier keeps (docs/DISK-TIER.md, Demotion); under the lock, one file at a time.
inline void Scheduler::write_ahead() {
    if (!disk_->can_write()) return;
    size_t unwritten = 0;
    for (const HostDonor& h : host_)
        if (!h.disk && !h.superseded) unwritten += h.history.held;
    for (const Boundary& b : bounds_)
        if (!b.disk) unwritten += b.state.held;
    if (unwritten <= host_cap_ / 4 * 3) return;
    for (const Boundary& b : bounds_)
        if (!b.disk && write_from(b)) return;
    for (const bool back : {false, true})
        for (const HostDonor& h : host_)
            if (!h.disk && !h.superseded && h.back == back && write_from(h)) return;
}

// Host donor i back on the devices as a donor of its own, its blocks and, on a model that keeps a state, its checkpoint slot taken as a first admission takes room, evicting older device donors to host memory; the host entry stays, renewed.
// With `spare`, for a job, only into room no donor gives up; false, the devices as they were but for donors evicted, where the room or a slot is not there or the copy fails, with `room` false where the room was what lacked; under the lock.
inline bool Scheduler::promote(size_t i, bool spare, bool* room, const std::shared_ptr<Request>& r) {
    const uint64_t id = host_[i].id;
    // The device donors it evicts to host memory never take its room for theirs (release_written).
    struct Unpromote {
        uint64_t& id;
        ~Unpromote() { id = 0; }
    } unpromote{promoting_};
    promoting_ = id;
    // Its conversation came back, so the room this promotion makes does not take the entry for a donor whose conversation did not (write_back).
    host_[i].back = true;
    host_[i].used = std::filesystem::file_time_type::clock::now();
    std::rotate(host_.begin() + (std::ptrdiff_t)i, host_.begin() + (std::ptrdiff_t)i + 1, host_.end());
    const auto entry = [&]() -> HostDonor* {
        for (HostDonor& h : host_)
            if (h.id == id) return &h;
        return nullptr;
    };
    const Taken t = make_room(this->room(), donors_.size(), npos, false, {}, 0, false, pools_.blocks_for(entry()->history.length), gain_of({}));
    if (!t.enough || (spare && !t.donors.empty())) {
        if (room) *room = false;
        return false;
    }
    std::vector<size_t> gone = t.donors;
    std::sort(gone.begin(), gone.end(), std::greater<size_t>());
    for (size_t g : gone) drop_donor(g, true);
    if (model_.keeps_state() && !checkpoint_room_held(0)) {
        moved(Move::promote_failed, "server: a donor of %zu tokens was not promoted from host memory (no checkpoint slot for its state)\n", entry() ? entry()->history.length : 0);
        return false;
    }
    while (donors_.size() >= max_seqs_) drop_donor(0, true);
    HostDonor* h = entry();
    if (!h) {
        moved(Move::promote_failed, "server: a copy in host memory was not promoted (the room its promotion made took it)\n");
        return false;
    }
    Donor d;
    const Clock::time_point start = Clock::now();
    Clock::time_point held = start;
    try {
        auto quiet = quieted();
        held = Clock::now();
        d.seq = quiet->restore_host(h->history);
    } catch (const std::exception& e) {
        moved(Move::promote_failed, "server: a donor of %zu tokens was not promoted from host memory (%s)\n", h->history.length, e.what());
        return false;
    }
    // The hold's wait for the recorders and the copies' enqueue, once for the server and on the request that took them; the copies' retirement is asked once a round (copies_retired).
    const Clock::time_point enqueued = Clock::now();
    const double hold_ms = std::chrono::duration<double, std::milli>(held - start).count(), enqueue_ms = std::chrono::duration<double, std::milli>(enqueued - held).count();
    ++restore_totals_.promotions;
    Stats::RestoreTotals::add(restore_totals_.hold_ms, restore_totals_.hold_largest_ms, hold_ms);
    Stats::RestoreTotals::add(restore_totals_.enqueue_ms, restore_totals_.enqueue_largest_ms, enqueue_ms);
    copies_.push_back(Copies{h->history.tickets, enqueued, r});
    if (r) {
        if (!r->job_) ++(h->from_disk ? restore_totals_.from_disk : restore_totals_.from_host);
        r->first_pass_ = true;
        std::lock_guard<std::mutex> lk(r->m_);
        Request::Restore& x = r->restore_;
        ++x.restores;
        x.kind = h->from_disk ? "disk" : "host";
        x.tokens += h->history.length;
        x.bytes += h->history.bytes;
        x.part(x.hold_ms, hold_ms, r->first_);
        x.part(x.enqueue_ms, enqueue_ms, r->first_);
        r->first_before_ = r->first_;
    }
    d.id = ++donor_ids_;
    d.on_host = h->id;
    d.back = h->back;
    d.tokens = h->tokens;
    d.classes = h->classes;
    host_moved_ += h->history.bytes;
    ++host_hits_;
    moved(Move::promoted, "server: a donor of %zu tokens promoted from host memory, %.1f MiB, its copy enqueued in %.1f ms\n", h->history.length, (double)h->history.bytes / (1 << 20), ms_since(start));
    donors_.push_back(std::move(d));
    return true;
}

// The promotions whose copies have all retired since the last round: their time from the enqueue, once for the server and on the request that took them.
// Asked once a round without waiting, as stage_ready asks a pass (Backend::done), and only while copies are outstanding.
inline void Scheduler::copies_retired() {
    for (size_t i = 0; i < copies_.size();) {
        if (!model_.copies_retired(copies_[i].tickets)) {
            ++i;
            continue;
        }
        const double ms = ms_since(copies_[i].at);
        {
            std::lock_guard<std::mutex> lk(m_);
            ++restore_totals_.copies;
            Stats::RestoreTotals::add(restore_totals_.copies_ms, restore_totals_.copies_largest_ms, ms);
        }
        if (copies_[i].r) {
            std::lock_guard<std::mutex> lk(copies_[i].r->m_);
            ++copies_[i].r->restore_.copies;
            copies_[i].r->restore_.copies_ms += ms;
        }
        copies_.erase(copies_.begin() + (std::ptrdiff_t)i);
    }
}

// A donor's blocks back to the pool, the oldest donor's unless another is named; one the device tier evicts (`evicted`) is copied to host memory first.
// Under the lock.
inline void Scheduler::drop_donor(size_t i, bool evicted) {
    Donor& d = donors_[i];
    if (evicted) write_back(d);
    try { model_.reset(d.seq); } catch (const std::exception&) {}
    donors_.erase(donors_.begin() + (std::ptrdiff_t)i);
}

// Job j's donor `kept` supersedes the other donors of its conversation, in the device tier and the host tier alike: the one it forked, the request's it reads the reply of and every one whose tokens its own begin with.
// Each goes to the front of its tier, oldest first among them, so it is the first to go when room is needed, and a superseded donor the devices evict is not copied to host memory; under the lock.
inline void Scheduler::supersede(const Request& j, uint64_t kept) {
    for (Donor& d : donors_)
        if (d.id != kept && supersedes(j, d.id, d.tokens)) d.superseded = true;
    for (HostDonor& h : host_)
        if (h.id != kept && supersedes(j, h.id, h.tokens)) {
            h.superseded = true;
        }
    std::stable_partition(donors_.begin(), donors_.end(), [](const Donor& d) { return d.superseded; });
    std::stable_partition(host_.begin(), host_.end(), [](const HostDonor& h) { return h.superseded; });
}

// Whether job j's donor supersedes the donor or host donor `id` holding `tokens` (supersede).
inline bool Scheduler::supersedes(const Request& j, uint64_t id, const std::vector<uint32_t>& tokens) {
    return (id && (id == j.source_ || (j.of_ && id == j.of_->parked_))) ||
           (tokens.size() <= j.prompt_.size() && std::equal(tokens.begin(), tokens.end(), j.prompt_.begin()));
}

} // namespace server
