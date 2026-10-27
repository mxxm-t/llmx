#pragma once
// The cache, behind one owner the scheduler calls (docs/CACHE-PLAN.md, section 4): its holders, their room and ranking, parking, the jobs, the tiers and the disk tier's side, and the restores.
// Included by server/scheduler.hpp after Request, which names Cache its friend.

namespace server {

// The tokens of r's history, prompt then generated: what a resume holds its cache to.
inline size_t history_tokens(const Request& r) { return r.prompt_.size() + r.gen_.size(); }

// The active requests a client waits on, jobs left out.
inline size_t requests(const std::vector<std::shared_ptr<Request>>& active) {
    return (size_t)std::count_if(active.begin(), active.end(), [](const std::shared_ptr<Request>& r) { return !r->job_; });
}

// Where history token `at` of r lies, prompt or generated; a stretch never spans the two, since the prompt's own ends with it.
inline const uint32_t* token_ptr(const Request& r, size_t at) {
    return at < r.prompt_.size() ? r.prompt_.data() + at : r.gen_.data() + (at - r.prompt_.size());
}

// A request's history, its prompt then what it generated, which a donor keeps as tokens.
inline std::vector<uint32_t> history(const Request& r) {
    std::vector<uint32_t> h = r.prompt_;
    h.insert(h.end(), r.gen_.begin(), r.gen_.end());
    return h;
}

inline double ms_since(std::chrono::steady_clock::time_point t) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(); }

// Each read and promotion once, where Request::Restore attributes them to requests: counts, and each part's sum and largest (docs/SERVER.md, What a restore cost).
struct RestoreTotals {
    size_t reads = 0, promotions = 0, copies = 0, from_host = 0, from_disk = 0, timeouts = 0;
    double read_ms = 0, read_largest_ms = 0, slabs_ms = 0, slabs_largest_ms = 0, hold_ms = 0, hold_largest_ms = 0, enqueue_ms = 0, enqueue_largest_ms = 0, copies_ms = 0, copies_largest_ms = 0;
    static void add(double& sum, double& largest, double ms) {
        sum += ms;
        largest = std::max(largest, ms);
    }
};

class Cache {
public:
    using Clock = std::chrono::steady_clock;
    // The model and the lock it shares with the scheduler, and the settings it keeps.
    Cache(infer::Model& model, std::mutex& m, Pools pools, size_t max_seqs, size_t host_cap, size_t limit, Clock::duration wait_bound, std::optional<std::chrono::milliseconds> read_wait,
          std::optional<double> recompute_row_ms, Clock::time_point epoch, std::function<void(bool)> hold, std::function<void()> unhold)
        : hold_(std::move(hold)), unhold_(std::move(unhold)), model_(model), m_(m), pools_(std::move(pools)), max_seqs_(max_seqs), host_cap_(host_cap), limit_(limit), wait_bound_(wait_bound), read_wait_(read_wait), recompute_row_ms_(recompute_row_ms),
          epoch_(epoch) {}
    // The scheduler's requests waiting for admission, queued and paused, which the calls that search them are given; under the lock.
    struct Lines {
        std::deque<std::shared_ptr<Request>>& queue;
        std::deque<std::shared_ptr<Request>>& paused;
    };
    // The disk tier, where the options give it bytes, its completions waking the scheduler through `wake`.
    void open_disk(const DiskOptions& disk, std::function<void()> wake) {
        if (!disk.bytes) return;
        if (!host_cap_) throw std::logic_error("server: a disk tier keeps what the host tier drops, and there is no host tier");
        disk_ = std::make_unique<DiskTier>(disk, model_, [this, wake] {
            {
                std::lock_guard<std::mutex> lk(m_);
                disk_wake_ = true;
            }
            wake();
        });
    }
    // Whether the disk tier keeps what memory holds for the next server (--disk-cache-keep).
    bool keeps() const { return disk_ && disk_->keeps(); }
    // The cache's half of the scheduler's stats, under the lock.
    template <class Into>
    void fill(Into& s) const {
        s.donors = donors_.size();
        s.prefix_hits = prefix_hits_;
        s.prefix_tokens = prefix_tokens_;
        s.taken_back = taken_back_;
        s.reserved = reserved_now_;
        s.donor_blocks = donor_blocks_now_;
        s.reprefills = reprefills_;
        s.jobs = jobs_now_.load();
        s.job_rows_left = job_rows_left_.load();
        s.room_waits = room_waits_;
        s.room_wait_timeouts = room_wait_timeouts_;
        s.room_wait_ms = room_wait_ms_;
        s.room_wait_longest_ms = room_wait_longest_ms_;
        s.reprefill_rows = reprefill_rows_;
        s.reprefill_cancels = reprefill_cancels_;
        s.reprefill_handed = reprefill_handed_;
        s.host_donors = host_.size();
        s.host_bytes = host_held_;
        s.host_hits = host_hits_;
        s.host_allocated = host_alive_;
        s.host_yielded = host_yielded_;
        s.host_bytes_moved = host_moved_;
        s.boundaries = bounds_.size();
        s.boundary_hits = bound_hits_;
        s.disk_entries = disk_now_.files;
        s.disk_states = disk_now_.states;
        s.disk_bytes = disk_now_.bytes;
        s.disk_bytes_written = disk_now_.written;
        s.disk_errors = disk_ ? disk_->errors() : 0;
        s.disk_refused_dirs = disk_ ? disk_->refused_dirs() : 0;
        s.lost_before_written = lost_before_written_;
        s.uncopyable = uncopyable_;
        for (size_t k = 0; k < s.moves.size(); ++k) s.moves[k] = moves_[k].load();
        s.reprefill_released = reprefill_released_.load();
        s.disk_refused = disk_refused_;
        s.disk_capped = disk_now_.capped;
        s.disk_unproved = disk_now_.unproved;
        s.disk_proved = disk_now_.proved;
        s.disk_disproved = disk_now_.disproved;
        s.disk_writing = disk_ && disk_->writing();
        s.disk_hits = disk_hits_;
        s.disk_waits = disk_waits_;
        s.disk_wait_timeouts = disk_wait_timeouts_;
        s.disk_bytes_read = disk_read_bytes_;
        s.disk_wait_ms = disk_wait_ms_;
        s.restore = restore_totals_;
        s.restore.timeouts = disk_wait_timeouts_;
        s.disk_in_flight = disk_ ? disk_->calls() : 0;
        s.disk_ready = disk_ && disk_->readable();
        s.disk_limit = disk_ ? disk_->cap() : 0;
    }
    // Whether the cache has work for the round, a seat free for a job or not (`seat`), and whether the disk tier woke it; under the lock.
    bool due(bool seat) const { return disk_wake_ || !follows_.empty() || !restores_.empty() || (seat && ready(jobs_)); }
    bool woken() const { return disk_wake_; }
    // The idle moment under --disk-cache-keep, from the scheduler's own condition, `busy`: kDiskIdle from when the server last had nothing to do, the round waking for it; under the lock.
    void idle_moment(bool busy) {
        if (!disk_ || !disk_->keeps() || busy) {
            idle_at_.reset();
            idle_tried_.clear();   // each idle period tries every donor again, one whose copy was refused or dropped before its file landed too
        }
        else if (!idle_at_) {
            idle_at_ = std::chrono::steady_clock::now() + disk_->idle();
            idled_ = false;
        }
    }
    // With nothing to do the host tier's slabs are taken ahead, a step a round with no sleep between, so a copy made later allocates none (grow_host); whether a step was taken.
    bool grow_ahead() {
        if (!host_growing_ || !jobs_.empty() || !follows_.empty()) return false;
        host_growing_ = grow_host();
        return true;
    }
    // The ids a conversation's next turn begins with, for a job to read (Scheduler::follow); under the lock.
    void follow(const std::shared_ptr<Request>& r, std::vector<uint32_t> ids, bool whole) { follows_.push_back(Follow{r, std::move(ids), whole}); }
    // Rows a job's pass read, under the lock.
    void job_rows_read(size_t n) { reprefill_rows_ += n; }
    // The scheduler's measured prompt rate, a pass's milliseconds a prompt row, as it measures it.
    void prompt_measured(double ms_row) { prompt_ms_row_ = ms_row; }
    // The scheduler stops (Scheduler::stop); under the lock.
    void stopping() { stopping_ = true; }
    // The donors make_room chose for a growing request (Scheduler::take), given up as evicted; it takes the lock.
    void take(const Taken& t) {
        std::vector<size_t> gone = t.donors;
        std::sort(gone.begin(), gone.end(), std::greater<size_t>());
        std::lock_guard<std::mutex> lk(m_);
        for (size_t i : gone) drop_donor(i, true);
    }
    // The donor `id` a paused request's history became, given up as evicted where make_room took it too; it takes the lock.
    void evict(uint64_t id) {
        std::lock_guard<std::mutex> lk(m_);
        for (size_t i = 0; i < donors_.size(); ++i)
            if (donors_[i].id == id) return drop_donor(i, true);
    }
    // A waiting request whose client left: its own donor goes with it when it holds less than a block, which no fork can share, and so does its proof; under the lock.
    void leave(const Lines& lines, const Request& r) {
        const size_t d = own_donor(r);
        if (d < donors_.size() && donors_[d].tokens.size() < model_.kv_block_tokens()) drop_donor(d);
        if (r.proof_) end_proof(lines, r.proof_);
    }
    // The cache's part of the server's stop, once every active request and job has been parked or released (Scheduler::run).
    void stop(const std::vector<std::shared_ptr<Request>>& active, const Lines& lines, std::condition_variable& cv) {
        // With --disk-cache-keep what memory holds goes to disk first; then the tier goes, its write in flight cancelled, so no write reads a slab released below; its thread calls back under the lock.
        if (disk_ && disk_->keeps()) flush_disk(lines, cv);
        // Streamed reads stop and are waited for before the tier goes, since their readers read its files and their copies write the devices.
        for (Streaming& s : streams_) s.read->cancel();
        for (Streaming& s : streams_) {
            while (!s.read->idle()) std::this_thread::yield();
            model_.reset(s.into->seq);
        }
        streams_.clear();
        std::unique_ptr<DiskTier> disk;
        {
            std::lock_guard<std::mutex> lk(m_);
            disk = std::move(disk_);
        }
        // The store goes first, so nothing reads a slab that is released after it.
        if (disk) {
            disk->close();
            disk->release(model_);
        }
        disk.reset();
        std::lock_guard<std::mutex> lk(m_);
        for (infer::HostHistory& h : disk_parked_) model_.release_host(h);
        disk_parked_.clear();
        for (DiskRead& rd : disk_reads_) {
            host_held_ -= rd.history.held;
            model_.release_host(rd.history);
        }
        disk_reads_.clear();
        jobs_.clear();
        follows_.clear();
        while (!donors_.empty()) drop_donor();
        while (!host_.empty()) drop_host(0);
        while (!bounds_.empty()) drop_bound(0);
        note_room(active);
    }
    static constexpr size_t npos = std::numeric_limits<size_t>::max();
    // A loss as `gone` reports it: the kind, the rows, and the history they were of, valid during the call.
    struct Moved {
        Move kind;
        size_t first, tokens;
        const std::vector<uint32_t>* history;
    };
    // Called on the scheduler thread, under its lock at some sites, with each loss of rows or of a state, so a test matches it to the history it was of; set before run.
    // A handler takes nothing of the scheduler's: it copies what it needs and returns.
    std::function<void(const Moved&)> on_move;
    // Admits a queued or paused request if make_room finds room without pausing anyone, reserving its history and max_tokens, or uncapped a growth step: its own donor taken back, a fork of what `served` found, or a fresh sequence.
    // A donor it takes back or shares every full block of goes first, one it forks otherwise last, and a forked donor make_room takes is consumed, the shared blocks counted once; under the lock.
    bool enter(const std::shared_ptr<Request>& r, std::vector<std::shared_ptr<Request>>& active) {
        std::vector<size_t> need = pools_.blocks_for(kGrowth.entry(history_tokens(*r), r->params_.until_limit, (size_t)r->params_.max_tokens, r->gen_.size()));
        size_t shared = 0;
        size_t d = own_donor(*r);
        const bool take = d < donors_.size();
        const Request* from = nullptr;
        // A message boundary is pinned while room is made, so nothing that room drops is the one chosen, and renewed once the request is admitted.
        struct Unpin {
            uint64_t& id;
            ~Unpin() { id = 0; }
        } unpin{pinned_bound_};
        if (!take) {
            const auto ask = asked(*r);
            const std::vector<uint32_t>& mine = ask.first;
            const std::vector<RowClass>& own = ask.second;
            // A job whose best source is in a pass waits for the round that pass has retired in: admitted now it would read what it could have forked, the whole prompt of the request it follows.
            const size_t held = served(mine, own, r.get(), &active).tokens;
            for (const auto& a : active) {
                const bool source = a->job_ ? a != r : a == r->of_;
                if (r->job_ && source && a->seq_.in_flight() && !a->classes_.empty() && shareable(*r, a->job_ ? a->prompt_ : history(*a), a->classes_, a->seq_) > held) return false;
            }
            // The holder serving most is used, and one that fails is passed over and the question asked again, once a holder (docs/CACHE-PLAN.md, section 4).
            std::vector<uint64_t> skip;
            for (;;) {
                const Served s = served(mine, own, r.get(), &active, skip);
                // A host copy goes back to the devices as a donor of its own and the question is asked again; a request finding no room waits, at most kWaitBound, rather than computing rows a copy holds (P2), a job never.
                const auto promoted = [&](size_t h, size_t tokens, const char* what) -> std::optional<bool> {
                    bool room = true;
                    const uint64_t id = host_[h].id;
                    if (promote(active, h, r->job_, &room, r)) return true;
                    if (!room && !r->job_ && waits_for_room(*r, tokens, what)) return std::nullopt;
                    skip.push_back(id);
                    return false;
                };
                if (s.kind == Served::Kind::host) {
                    if (!promoted(s.index, s.tokens, "a copy in host memory")) return false;
                    continue;
                }
                if (s.kind == Served::Kind::bound) {
                    // A boundary takes its blocks from a donor holding its rows, a host copy holding them promoted first, and a checkpoint slot of its own (Model::fork with a state).
                    const uint64_t b = bounds_[s.index].id;
                    pinned_bound_ = b;
                    const Boundary& bound = bounds_[s.index];
                    size_t k = donors_.size();
                    for (size_t i = 0; i < donors_.size() && k == donors_.size(); ++i)
                        if (std::find(skip.begin(), skip.end(), donors_[i].id) == skip.end() && holds(bound, donors_[i].tokens, donors_[i].classes, donors_[i].seq.length())) k = i;
                    if (k == donors_.size()) {
                        for (size_t h = 0; h < host_.size(); ++h)
                            if (std::find(skip.begin(), skip.end(), host_[h].id) == skip.end() && holds(bound, host_[h].tokens, host_[h].classes, host_[h].history.length)) {
                                if (!promoted(h, s.tokens, "a copy in host memory under a message boundary")) return false;
                                break;
                            }
                        continue;
                    }
                    const uint64_t id = donors_[k].id;
                    k = donors_.size();
                    if (checkpoint_room_held(0, id))
                        for (size_t i = 0; i < donors_.size(); ++i)
                            if (donors_[i].id == id) k = i;
                    if (k == donors_.size()) {
                        std::fprintf(stderr, "server: cache: a boundary of %zu tokens found no checkpoint slot; the next restore point is asked\n", s.tokens);
                        pinned_bound_ = 0;
                        skip.push_back(b);
                        continue;
                    }
                    d = k;
                    shared = s.tokens;
                    break;
                }
                pinned_bound_ = 0;
                // A running job, as far as it has read, is a source too, and a job's own request while it runs, where no pass holds them.
                d = s.kind == Served::Kind::donor ? s.index : donors_.size();
                if (s.kind == Served::Kind::running) from = active[s.index].get();
                shared = s.tokens;
                break;
            }
        }
        const bool keep = !from && (take || shared);
        // A job's own source goes first: the job holds what it forks, and takes no other donor's room (below).
        const bool keep_first = take || (keep && (r->job_ || donors_[d].tokens.size() - shared < model_.kv_block_tokens()));
        // A fork takes the blocks it shares from its source, and a donor taken back is the request's own, so neither is room the pool must give.
        std::vector<size_t> fresh = need;
        const std::vector<size_t> shares = take ? model_.kv_held(donors_[d].seq) : pools_.blocks_for(shared);
        for (size_t p = 0; p < fresh.size(); ++p) fresh[p] -= std::min(fresh[p], shares[p]);
        const Taken t = make_room(room(active), ranked(), keep ? d : npos, keep_first, {}, 0, false, fresh, gain_of({}, keep ? d : npos, take ? 0 : shared, take));
        if (!t.enough) return false;
        // A request whose room takes a donor the host tier cannot keep waits for that donor's files to be whole on disk, its write then whole pieces (question 5 of docs/CACHE-PLAN.md).
        // Past the wait's bound it takes the room, and the rows are said as lost.
        if (const size_t g = r->job_ ? donors_.size() : awaited_write(t, keep ? d : npos); g < donors_.size()) {
            const Donor& v = donors_[g];
            const size_t n = copy_length(v);
            // While the donor's own write is in flight: what is not yet on disk, its files' bytes less what that file has written, and the rate, the write's own once measured, else the store's.
            const std::vector<std::string> dv = digests(v);
            const auto [wrote, seconds] = disk_->writing_of(dv, model_.kv_block_tokens());
            const double remaining = std::max(0.0, (double)disk_->unwritten(dv, model_.kv_block_tokens(), n, model_.keeps_state(), true) - (double)wrote);
            const double own = in_flight_rate(wrote, seconds), rate = seconds <= 0 ? 0 : own > 0 ? own : disk_->write_rate();
            if (waits_for_room(*r, n, "a donor's write to disk", remaining, rate)) {
                // The disk round makes the awaited donor's write whole pieces once, not every round.
                if (awaited_ != v.id) disk_wake_ = true;
                awaited_ = v.id;
                return false;
            }
        }
        if (r->room_wait_) {
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - r->room_since_).count();
            ++room_waits_;
            room_wait_ms_ += ms;
            room_wait_longest_ms_ = std::max(room_wait_longest_ms_, ms);
            r->room_wait_ = false;
            awaited_ = 0;
        }
        // A job takes only spare room: no donor but the one it forks gives blocks up for it (docs/CACHE-PLAN.md, question 2), and what it would have read is the next turn's new work.
        if (r->job_ && std::any_of(t.donors.begin(), t.donors.end(), [&](size_t g) { return !keep || g != d; })) {
            if (!r->no_room_) std::fprintf(stderr, "server: cache: a job of %zu ids waits for spare room\n", r->prompt_.size());
            r->no_room_ = true;
            return false;
        }
        std::vector<size_t> gone = t.donors;
        std::sort(gone.begin(), gone.end(), std::greater<size_t>());
        bool consume = false;
        for (size_t i : gone) {
            if (keep && i == d) { consume = true; continue; }
            drop_donor(i, true);
            if (i < d) --d;
        }
        Boundary* bound = pinned_bound_ ? find_bound(pinned_bound_) : nullptr;
        admit(*r, d, shared, take, from, bound ? &bound->state : nullptr);
        if (bound) bound->used = std::filesystem::file_time_type::clock::now();
        // A cut's uncut history (step 9 of docs/CACHE-PLAN.md): every holder of the first row it dropped is superseded, so it ranks first to go; the turn says the re-read it pays (P5).
        if (!r->cut_.empty()) {
            const size_t again = history_tokens(*r) - shared;
            if (prompt_ms_row_ > 0) std::fprintf(stderr, "server: cache: a cut turn reads %zu rows again beside %zu reused, about %.1f s at the measured prompt rate\n", again, shared, prompt_ms_row_ * (double)again / 1000);
            else std::fprintf(stderr, "server: cache: a cut turn reads %zu rows again beside %zu reused\n", again, shared);
            const auto uncut = [&](const std::vector<uint32_t>& t) { return t.size() >= r->cut_.size() && std::equal(r->cut_.begin(), r->cut_.end(), t.begin()); };
            for (Donor& o : donors_) o.superseded = o.superseded || uncut(o.tokens);
            for (HostDonor& o : host_) o.superseded = o.superseded || uncut(o.tokens);
            for (Boundary& o : bounds_) o.superseded = o.superseded || uncut(o.tokens);
            r->cut_.clear();
        }
        if (consume && !take) {
            moved(Move::consumed, "server: cache: a donor of %zu tokens consumed by the %s that forked %zu of it\n", donors_[d].seq.length(), r->job_ ? "job" : "request", shared);
            // The fork holds the donor's rows up to where it forked; what the donor could restore past that goes with it unless another holder has it.
            const size_t point = copy_length(donors_[d]);
            const std::vector<uint32_t> tokens = donors_[d].tokens;
            const std::vector<RowClass> classes = donors_[d].classes;
            drop_donor(d);
            const size_t left = point > shared ? std::max(shared, held_below(tokens, classes, point)) : point;
            if (point > left) this->gone(tokens, Move::consumed_lost, left, point, "server: cache: rows %zu to %zu of a history left memory with the donor a fork consumed\n", left, point);
        }
        if (shared) moved(Move::forked);
        if (r->job_) moved(Move::job_started, "server: cache: a job started, rows %zu to %zu to read\n", shared, r->prompt_.size());
        else stale(r->prompt_, nullptr, "a request admitted reads them");
        r->need_ = std::move(need);
        if (!r->admission_) r->admission_ = ++admissions_;
        r->edge_steps_ = 1;
        r->stepped_ = Clock::now();
        active.insert(std::upper_bound(active.begin(), active.end(), r, [](const auto& a, const auto& b) { return a->admission_ < b->admission_; }), r);
        return true;
    }
    void host_pressure();
    // Defined below, out of the class.
    bool checkpoint_room(size_t keeps, uint64_t spare = 0);
    // A holder's place in the one ranking of what stays (rank): its class, 0 where its going loses nothing, 1 for a conversation that did not come back and 2 for one that did, then its last use.
    struct Rank {
        int cls = 0;
        std::filesystem::file_time_type used{};
        bool operator<(const Rank& o) const { return cls != o.cls ? cls < o.cls : used < o.used; }
    };
    void asked_boundary(Request& r);
    std::vector<size_t> ranked() const;
    // What write_back would do with a copy of `n` tokens of a donor that came back (`back`) or not: keep it after releasing `drops` ((a boundary, id), in order), or why not.
    struct HostRoom {
        enum class Why { kept, larger, outranked, room, memory } why = Why::kept;
        std::vector<std::pair<bool, uint64_t>> drops;
    };
    HostRoom host_room(size_t n, const Rank& mine) const;
    std::vector<size_t> room(const std::vector<std::shared_ptr<Request>>& active) const;
    std::vector<size_t> promised(const Request& r) const;
    Gain gain_of(const std::vector<std::shared_ptr<Request>>& active, size_t keep = npos, size_t kept_len = 0, bool take = false, const std::vector<const infer::Sequence*>& also = {}) const;
    void note_room(const std::vector<std::shared_ptr<Request>>& active);
    template <class... A>
    void moved(Move kind, const char* line = nullptr, A... a);
    void disk_round(const std::vector<std::shared_ptr<Request>>& active, const Lines& lines);
    // Defined below, out of the class.
    static bool ready(const std::deque<std::shared_ptr<Request>>& q);
    std::optional<std::chrono::steady_clock::time_point> next_disk_bound(const Lines& lines);
    bool reading(const Lines& lines, Request& r);
    void prove(const Lines& lines, Request& r);
    void prefetch(const std::vector<std::shared_ptr<Request>>& active, const Lines& lines);
    void copies_retired();
    // A request leaves the active set, its history kept as a donor when it holds at least `least` tokens (a full block unless a paused request's own asks for less); the donor's id, or 0 with `why` saying why.
    // A donor keeps only the blocks it holds reserved, at most max_seqs donors exist, and on a model that keeps a recurrent state a donor ends at its checkpoint, with none it keeps no donor and recomputes from its start.
    uint64_t park(std::vector<std::shared_ptr<Request>>& active, const Lines& lines, size_t i, const std::vector<uint32_t>& h, size_t least = 0, const char** why = nullptr) {
        auto r = active[i];
        active.erase(active.begin() + (std::ptrdiff_t)i);
        drop_encode(*r);
        if (r->proof_) unprove(lines, *r);
        size_t held = r->seq_.length();
        if (model_.keeps_state() && held && !(least && quieted()->keep(r->seq_))) {
            const std::optional<size_t> kept = model_.checkpoint(r->seq_);
            held = kept ? quieted()->retract(r->seq_, *kept) : 0;
        }
        uint64_t id = 0;
        // A paused request's own donor (`least`) is a conversation that comes back, so the host tier does not refuse it first.
        const bool back = least || (r->job_ ? !r->of_ || r->of_->reused() > 0 : r->reused() > 0);
        std::unique_lock<std::mutex> lk(m_, std::defer_lock);
        if (held && held >= (least ? least : model_.kv_block_tokens())) {
            lk.lock();
            // A job's donor supersedes its own conversation's older donors, so it never takes an unrelated conversation's place while those stay.
            if (r->job_ && donors_.size() >= max_seqs_)
                for (Donor& o : donors_) o.superseded = o.superseded || supersedes(*r, o.id, o.tokens);
        }
        if (lk.owns_lock()) {
            Donor d;
            d.id = id = ++donor_ids_;
            d.back = back;
            d.tokens.assign(h.begin(), h.begin() + (std::ptrdiff_t)std::min(h.size(), held));
            d.classes = clip(r->classes_, held);
            d.seq = std::move(r->seq_);
            r->need_.clear();
            // The copy its last pause left in host memory (paused_as_) is superseded, the first to go, and so is a donor promoted from it and left on the devices, which would stand for it again when evicted.
            // Where a finished request's donor ends at its prompt's checkpoint the copy is the longer one, its rows past the prompt decode rows no later prompt forks.
            for (Donor& o : donors_) o.superseded = o.superseded || (r->paused_as_ && o.on_host == r->paused_as_);
            donors_.push_back(std::move(d));
            moved(Move::parked);
            // A donor holding a block that no copy can take (copy_length 0) is counted and said; by construction only a request paused inside a block, usable for its own take-back alone, makes one.
            if (held >= model_.kv_block_tokens() && !copy_length(donors_.back())) {
                ++uncopyable_;
                std::fprintf(stderr, "server: cache: a donor of %zu tokens whose checkpoint no copy can take: it stays on the devices only\n", held);
            }
            for (HostDonor& h : host_) h.superseded = h.superseded || (r->paused_as_ && h.id == r->paused_as_);
            // The donor count then gives up the lowest-ranked donor, the newcomer among them and a write in flight last (A19 in docs/CACHE-PLAN.md), each leaving as any evicted donor does.
            while (donors_.size() > max_seqs_) {
                const size_t v = ranked().front();
                if (donors_[v].id == id) {
                    if (why) *why = "every donor ranks above it";
                    id = 0;
                }
                restoring_ = r->job_ && !r->of_;
                drop_donor(v, true);
                restoring_ = false;
            }
        } else {
            if (why) *why = "too short to keep";
            release(*r, active);
        }
        if (lk.owns_lock()) {
            note_room(active);
            lk.unlock();
        }
        r->seq_ = infer::Sequence{};
        return id;
    }
    // Whether r could be admitted now beside the requests and jobs running, by admission's own test: a free seat, and the blocks enter reserves for it from free blocks and donors make_room may give up.
    // Under the lock.
    bool admits(const Request& r, size_t seats, const std::vector<std::shared_ptr<Request>>& active) const {
        if (seats >= max_seqs_) return false;
        const std::vector<size_t> need = pools_.blocks_for(kGrowth.entry(history_tokens(r), r.params_.until_limit, (size_t)r.params_.max_tokens, r.gen_.size()));
        return make_room(room(active), ranked(), npos, false, {}, 0, false, need, gain_of({})).enough;
    }
    // Where a job would give way, a waiting request that can fork more of its rows than of any history kept takes them over: the job's history becomes a donor with its blocks (park), which admission forks and consumes by the donor rules.
    // The job ends where the request's prompt holds all its ids, the request being the turn it read ahead for, and otherwise waits to start again from that donor; a job in a pass is passed by, as it is by yield_jobs, until its pass has retired.
    void hand_jobs(std::vector<std::shared_ptr<Request>>& active, const Lines& lines) {
        for (size_t i = 0; i < active.size();) {
            auto j = active[i];
            bool wanted = false, covered = false;
            if (j->job_ && !j->seq_.in_flight() && !j->classes_.empty()) {
                std::lock_guard<std::mutex> lk(m_);
                // Queued requests alone: a paused one has its own history and takes its own donor back.
                for (const auto& r : lines.queue) {
                    if (wanted || r->cancel_.load() || !r->classes_.empty()) continue;
                    const auto ask = asked(*r);
                    const std::vector<uint32_t>& mine = ask.first;
                    const std::vector<RowClass>& own = ask.second;
                    if (shareable(*r, j->prompt_, j->classes_, j->seq_) <= served(mine, own, r.get(), nullptr).tokens) continue;
                    wanted = true;
                    covered = r->prompt_.size() >= j->prompt_.size() && std::equal(j->prompt_.begin(), j->prompt_.end(), r->prompt_.begin());
                }
            }
            if (!wanted) {
                ++i;
                continue;
            }
            park_job(active, lines, i, "was handed to the request it read for");
            std::lock_guard<std::mutex> lk(m_);
            ++reprefill_handed_;
            moved(Move::job_handed);
            if (covered) j->finished_ = true;
            else {
                // Back in the line, it is looked at again (prefetch), its history having moved.
                j->disk_checked_ = j->looked_ = false;
                jobs_.push_front(j);
            }
        }
    }
    // Each job not in flight gives way (park_job) and waits to start again from its donor once nothing else waits; whether any did.
    // One with at most one block left reads it first where `waiter` would wait for a donor's write in any case (A21), a spare with no clock: one block at most, ended by the write, which the next round no longer names.
    bool yield_jobs(std::vector<std::shared_ptr<Request>>& active, const Lines& lines, const Request* waiter = nullptr) {
        const bool spare = waiter && waits_for_write(*waiter, active);
        const size_t bt = model_.kv_block_tokens();
        bool any = false;
        for (size_t i = 0; i < active.size();) {
            auto j = active[i];
            if (!j->job_ || j->seq_.in_flight() || (spare && j->whole_ && j->prompt_.size() <= j->seq_.length() + bt)) {
                ++i;
                continue;
            }
            {
                std::lock_guard<std::mutex> lk(m_);
                moved(Move::job_not_handed, "server: cache: a job of %zu rows was not handed over (%s)\n", j->seq_.length(), not_handed(lines, *j));
            }
            park_job(active, lines, i, "gave way");
            any = true;
            std::lock_guard<std::mutex> lk(m_);
            moved(Move::job_gave_way);
            j->disk_checked_ = j->looked_ = false;
            jobs_.push_front(j);
            ++reprefill_cancels_;
        }
        return any;
    }
    // The ids given since the last round, each to its request's job: a new job, a job's ids grown as the reply is written or made whole, or a job that has read ids the reply's next turn no longer begins with started again; none drops the job.
    // Then, while no request waits and a seat is free, the waiting jobs oldest first are admitted as a request is, through enter, so each forks the history that shares most with it and takes room as a first admission does.
    void follow_up(std::vector<std::shared_ptr<Request>>& active, const Lines& lines) {
        std::vector<Follow> given;
        {
            std::lock_guard<std::mutex> lk(m_);
            given.swap(follows_);
        }
        std::vector<Follow> later;
        {
            // The jobs that read back a conversation that left the devices (restore_later): one waiting of the same start goes, as for a new job below.
            std::lock_guard<std::mutex> lk(m_);
            for (auto& j : restores_) {
                stale(j->prompt_, nullptr, "a newer job of the same conversation replaces it");
                if (jobs_.size() >= max_seqs_) jobs_.pop_front();
                jobs_.push_back(std::move(j));
            }
            restores_.clear();
        }
        for (Follow& f : given) {
            if (f.whole && !f.ids.empty()) trim(f.of->parked_, f.ids);
            const std::pair<size_t, size_t> at = job_of(active, f.of.get());
            std::shared_ptr<Request> j = at.first != npos ? active[at.first] : at.second != npos ? jobs_[at.second] : nullptr;
            if (j && j->seq_.in_flight()) {
                later.push_back(std::move(f));
                continue;
            }
            // Ids given once the reply ended stay; any given before them is older.
            if (j && j->whole_ && !f.whole) continue;
            // What the job's cache holds must still begin the ids; otherwise it starts again.
            const size_t read = j ? j->seq_.length() : 0;
            const bool holds = j && !f.ids.empty() && f.ids.size() >= read && std::equal(j->prompt_.begin(), j->prompt_.begin() + (std::ptrdiff_t)read, f.ids.begin());
            // A running job keeps going only where the blocks its longer ids take past its reservation are free, reserved before it reads them; otherwise it gives its room back and waits to start again, as a first admission would.
            bool keeps = holds && at.first != npos;
            std::vector<size_t> more;
            if (keeps) {
                const std::vector<size_t> need = pools_.blocks_for(f.ids.size()), free = room(active);
                more.resize(need.size());
                for (size_t p = 0; p < need.size(); ++p) {
                    more[p] = need[p] > j->need_[p] ? need[p] - j->need_[p] : 0;
                    keeps = keeps && more[p] <= free[p];
                }
            }
            if (j && at.first != npos && !keeps) {
                park_job(active, lines, at.first, holds ? "outgrew its room as its ids grew" : "stopped, its ids changed");
                std::lock_guard<std::mutex> lk(m_);
                moved(Move::job_gave_way);
            }
            std::lock_guard<std::mutex> lk(m_);
            if (j && (at.second != npos || !keeps)) {
                const auto w = std::find(jobs_.begin(), jobs_.end(), j);
                if (w != jobs_.end()) jobs_.erase(w);
            }
            if (f.ids.empty()) continue;
            if (keeps) {
                add(j->need_, more);
                j->prompt_ = std::move(f.ids);
                j->whole_ = f.whole;
                j->disk_checked_ = j->looked_ = false;
                continue;
            }
            server::SampleParams params;
            params.max_tokens = 0;
            auto n = std::make_shared<Request>(std::move(f.ids), params);
            n->job_ = true;
            n->whole_ = f.whole;
            n->of_ = f.of;
            // The conversation's images, which its ids hold, without their rows: the request's encoder gives them again where the job reads one.
            for (infer::MediaRun m : f.of->media_) {
                m.encoded.reset();
                n->media_.push_back(std::move(m));
            }
            n->encode_ = f.of->encode_;
            // A job per conversation: one waiting that the new one's ids extend, or that reads for the same request, is that conversation's older read, and goes; at most max_seqs wait, the oldest dropped.
            stale(n->prompt_, n->of_.get(), "a newer job of the same conversation replaces it");
            if (jobs_.size() >= max_seqs_) jobs_.pop_front();
            jobs_.push_back(n);
        }
        std::lock_guard<std::mutex> lk(m_);
        follows_.insert(follows_.begin(), std::make_move_iterator(later.begin()), std::make_move_iterator(later.end()));
        const bool stalled = std::any_of(active.begin(), active.end(), [](const std::shared_ptr<Request>& r) { return r->stalled_; });
        // The waiting jobs in order: one reading its history from disk, or refused room while requests run, is passed over, so a younger one that needs neither starts (F2DEV's audit, 2026-10-22).
        for (size_t i = 0; !stalled && lines.queue.empty() && lines.paused.empty() && active.size() < max_seqs_ && i < jobs_.size();) {
            const std::shared_ptr<Request> j = jobs_[i];
            // A job whose history is being read from disk starts once the read has ended or its bound passed (reading).
            if (reading(lines, *j)) {
                ++i;
                continue;
            }
            if (!enter(j, active)) {
                // With no request running, nothing will make room it does not find now.
                if (!requests(active)) {
                    jobs_.erase(jobs_.begin() + (std::ptrdiff_t)i);
                    break;
                }
                ++i;
                continue;
            }
            jobs_.erase(jobs_.begin() + (std::ptrdiff_t)i);
        }
    }
    // A finished request's donor `id` keeps only the token prefix its next turn's whole ids share with it, in whole blocks: no turn forks the rows past it, a reply's reasoning among them (F2DEV and XDEV, 2026-10-20 08:29).
    // A state model's donor ends at its checkpoint, which no retract moves, so it stays on the devices whole and is copied and written only where that checkpoint lies within the prefix.
    void trim(uint64_t id, const std::vector<uint32_t>& ids) {
        std::lock_guard<std::mutex> lk(m_);
        for (size_t d = 0; id && d < donors_.size(); ++d) {
            Donor& x = donors_[d];
            if (x.id != id) continue;
            size_t same = 0;
            while (same < x.tokens.size() && same < ids.size() && x.tokens[same] == ids[same]) ++same;
            const size_t bt = model_.kv_block_tokens(), keep = same / bt * bt;
            if (keep >= x.seq.length() / bt * bt) return;
            // A file of this donor in flight past the prefix stops, so none of the rows let go is left under the history's old identity; the prefix is written again (F2DEV, 2026-10-22).
            if (disk_ && disk_->writes_past(digests(x), bt, keep)) disk_->cancel_write();
            x.keep = keep;
            x.digests.clear();
            if (model_.keeps_state()) return;
            if (!keep) return drop_donor(d);
            quieted()->retract(x.seq, keep);
            x.tokens.resize(keep);
            while (!x.classes.empty() && x.classes.back().end > keep && (x.classes.size() == 1 || x.classes[x.classes.size() - 2].end >= keep)) x.classes.pop_back();
            if (!x.classes.empty()) x.classes.back().end = std::min(x.classes.back().end, keep);
            return;
        }
    }
    // The jobs running and waiting, and the ids they have yet to read, for the server's figures.
    void count_jobs(const std::vector<std::shared_ptr<Request>>& active) {
        size_t n = 0, left = 0;
        for (const auto& r : active)
            if (r->job_) ++n, left += r->prompt_.size() - std::min(r->prompt_.size(), r->seq_.length());
        std::lock_guard<std::mutex> lk(m_);
        for (const auto& j : jobs_) ++n, left += j->prompt_.size();
        jobs_now_.store(n);
        job_rows_left_.store(left);
    }
    // On a model that keeps a state, each job between passes keeps its state where it has read to a whole block (Model::keep), so a follow-up turn that arrives before it completes forks what it has read.
    // Each job that has read the whole of its ids becomes a donor beside the one it forked, which a regenerated reply still forks at its earlier checkpoint while room allows (supersede).
    void complete_jobs(std::vector<std::shared_ptr<Request>>& active, const Lines& lines) {
        for (size_t i = 0; i < active.size();) {
            auto j = active[i];
            const size_t len = j->seq_.length();
            if (j->job_ && model_.keeps_state() && !j->seq_.in_flight() && len && len % model_.kv_block_tokens() == 0 &&
                model_.checkpoint(j->seq_) != std::optional<size_t>(len) && !quieted()->keep(j->seq_) && checkpoint_room(0, j->source_))
                quieted()->keep(j->seq_);
            if (!j->job_ || !j->whole_ || j->seq_.in_flight() || len < j->prompt_.size()) {
                ++i;
                continue;
            }
            // A job that forked every one of its ids from a donor still there read nothing: that donor holds them, so the job leaves no second one and takes no slot, and the donor stands for the job's.
            bool held;
            {
                std::lock_guard<std::mutex> lk(m_);
                held = j->source_ && j->reused() == j->prompt_.size() && std::any_of(donors_.begin(), donors_.end(), [&](const Donor& d) { return d.id == j->source_; });
            }
            if (held) {
                active.erase(active.begin() + (std::ptrdiff_t)i);
                release(*j, active);
                j->seq_ = infer::Sequence{};
                j->finished_ = true;
                std::lock_guard<std::mutex> lk(m_);
                ++reprefills_;
                moved(Move::job_done);
                disk_wake_ = true;
                // It is the conversation's newest holder now, whatever an earlier job's longer ids made of it.
                for (Donor& d : donors_)
                    if (d.id == j->source_) {
                        d.superseded = false;
                        keep_boundary(d);
                    }
                supersede(*j, j->source_);
                continue;
            }
            // On a model that keeps a state its live state becomes its checkpoint, in a slot an older donor gives up or, where none can, the donor it forked.
            if (model_.keeps_state() && !quieted()->keep(j->seq_) && !(checkpoint_room(0, j->source_) && quieted()->keep(j->seq_))) {
                {
                    std::lock_guard<std::mutex> lk(m_);
                    for (size_t d = 0; d < donors_.size(); ++d)
                        if (j->source_ && donors_[d].id == j->source_) { drop_donor(d, true); break; }
                }
                if (!quieted()->keep(j->seq_)) {
                    // With no slot for its end's state it is parked at the checkpoint it kept between passes, so its conversation's whole blocks keep a holder.
                    const char* why = "";
                    const uint64_t id = park(active, lines, i, j->prompt_, 0, &why);
                    j->finished_ = true;
                    if (id) std::fprintf(stderr, "server: cache: a job of %zu rows found no slot for its end's state and was parked at its checkpoint\n", len);
                    else std::fprintf(stderr, "server: cache: a job of %zu rows found no slot for its end's state and kept nothing, %s\n", len, why);
                    continue;
                }
            }
            const char* why = "";
            const uint64_t id = park(active, lines, i, j->prompt_, 0, &why);
            j->finished_ = true;
            std::fprintf(stderr, "server: cache: a job completed, its %zu rows %s%s\n", len, id ? "kept as a donor" : "kept nowhere, ", id ? "" : why);
            std::lock_guard<std::mutex> lk(m_);
            if (!id) continue;
            ++reprefills_;
            moved(Move::job_done);
            disk_wake_ = true;
            if (!donors_.empty() && donors_.back().id == id) keep_boundary(donors_.back());
            supersede(*j, id);
        }
    }
    // reset waits for the last pass that touched the sequence, so its blocks return to the pool only once the device is done with them.
    void release(Request& r, const std::vector<std::shared_ptr<Request>>& active) {
        try { model_.reset(r.seq_); } catch (const std::exception&) {}
        std::lock_guard<std::mutex> lk(m_);
        r.need_.clear();
        note_room(active);
    }
    // How a turn-end write beyond the host tier may go this round, told by the round before disk_round (Scheduler::write_turn): whole, a piece's time, a piece that takes background work's turn, or none but the awaited donor's.
    enum class WriteTurn { whole, piece, due, none };
    void write_turn(WriteTurn t) { turn_ = t; }
    // Whether such a write waited for its turn in the last disk_round, and whether one took the turn since asked, which the scheduler counts; under the lock.
    bool write_held() const { return held_; }
    bool took_write_turn() {
        const bool took = took_;
        took_ = false;
        return took;
    }
    // The position r's slices stop at, its next image's first row while that image waits for its rows, or npos.
    static size_t image_stop(const Request& r) { return r.media_next_ < r.media_.size() ? r.media_[r.media_next_].first : npos; }
    // One piece of r's waiting image's encode under the hold, a pass in flight possibly recording on its device, and more while `more` says so; true if a piece ran, the last giving the sequence its images.
    // None runs while another request's image encodes, but a job's encode gives way to a request's, since a job takes turns only while no prompt waits.
    bool encode_piece(Request& r, const std::function<bool()>& more) {
        if (encoding_for_ && encoding_for_ != &r) {
            if (r.job_ || !encoding_for_->job_) return false;
            drop_encode(*encoding_for_);
        }
        std::shared_ptr<const std::vector<float>> rows;
        {
            const Held held = quieted();
            if (!encoding_for_) {
                encoding_ = r.encode_(r.media_next_);
                encoding_for_ = &r;
                encoded_pieces_ = 0;
                encode_began_ = std::chrono::steady_clock::now();
            }
            do {
                rows = encoding_.step();
                ++encoded_pieces_;
            } while (!rows && more());
        }
        if (!rows) return true;
        drop_encode(r);
        r.media_[r.media_next_].encoded = std::move(rows);
        give_ready_media(r);
        return true;
    }
    // Ends r's encode at a piece boundary, as r ends or its sequence is replaced; the encoder is free for the next image.
    void drop_encode(const Request& r) {
        if (encoding_for_ != &r) return;
        encoding_ = ImageSteps{};
        encoding_for_ = nullptr;
    }
    // The pieces of r's encode run and in all and when it began, 0 of 0 where the encoder is not r's.
    struct EncodeProgress {
        size_t done = 0, total = 0;
        std::chrono::steady_clock::time_point began{};
    };
    EncodeProgress encode_progress(const Request& r) const { return encoding_for_ == &r ? EncodeProgress{encoded_pieces_, encoding_.pieces, encode_began_} : EncodeProgress{}; }
private:
    // Defined below, out of the class.
    void drop_donor(size_t i = 0, bool evicted = false);
    static constexpr Growth kGrowth{};
    size_t token_limit() const { return limit_; }
    // The model for work on a stage's devices that is not a pass's stage, held for as long as the value lives (Scheduler::quieted, whose hold and release the scheduler hands in at construction).
    std::function<void(bool)> hold_;
    std::function<void()> unhold_;
    struct Held {
        const Cache& c;
        explicit Held(const Cache& cache, bool generated = false) : c(cache) { c.hold_(generated); }
        Held(const Held&) = delete;
        Held& operator=(const Held&) = delete;
        ~Held() { c.unhold_(); }
        infer::Model* operator->() const { return &c.model_; }
        infer::Model& operator*() const { return c.model_; }
    };
    Held quieted(bool generated = false) const { return Held(*this, generated); }
    // The stats' own types the moved code names.
    struct Stats {
        using RestoreTotals = server::RestoreTotals;
    };
    // How many of the first `n` positions two records computed alike, their extents of one class.
    size_t alike(const std::vector<RowClass>& a, const std::vector<RowClass>& b, size_t n) const {
        size_t at = 0;
        for (size_t i = 0, j = 0; at < n && i < a.size() && j < b.size();) {
            if (a[i].image != b[j].image || (a[i].extent != b[j].extent && model_.row_class(a[i].extent) != model_.row_class(b[j].extent))) break;
            at = std::min({a[i].end, b[j].end, n});
            if (a[i].end <= at) ++i;
            if (b[j].end <= at) ++j;
        }
        return std::min(at, n);
    }
    // Defined below, out of the class.
    bool grow_host();
    // A finished or paused request kept for its cache: the tokens its history holds, how they were computed, the sequence holding them and the blocks it has reserved.
    // Shared full blocks are immutable, so a fork of it is safe while it lives; its partial last block, which no fork shares, goes on only in the request that paused, taking it back whole.
    struct Donor {
        uint64_t id = 0;   // which paused request may take it back (Request::donor_)
        std::vector<uint32_t> tokens;
        std::vector<RowClass> classes;
        infer::Sequence seq;
        uint64_t on_host = 0;         // the host donor holding the same history, which it was promoted from
        bool superseded = false;      // a job's donor holds what its conversation's next turn needs, so this one is kept only while room allows, as a regenerate's
        bool back = false;            // its conversation came back: the request it holds, or the one a job read the reply of, forked a history a tier kept
        size_t keep = std::numeric_limits<size_t>::max();   // what its next turn's ids share with it, past which nothing is copied or written (trim)
        mutable std::vector<std::string> digests;   // as HostDonor::digests, of what its cache holds
        std::filesystem::file_time_type used = std::filesystem::file_time_type::clock::now();   // its last use: parked, forked or promoted
    };
    // Defined below, out of the class.
    size_t own_donor(const Request& r) const;
    bool checkpoint_room_held(size_t keeps, uint64_t spare = 0);
    // What memory serves a history (served): the longest restore point a holder gives, the holder's kind and its index in donors_, host_, bounds_ or the running requests.
    struct Served {
        enum class Kind { none, donor, host, bound, running };
        Kind kind = Kind::none;
        size_t index = 0, tokens = 0;
    };
    Served served(const std::vector<uint32_t>& tokens, const std::vector<RowClass>& classes, const Request* self, const std::vector<std::shared_ptr<Request>>* running, const std::vector<uint64_t>& skip = {}) const;
    std::pair<std::vector<uint32_t>, std::vector<RowClass>> asked(const Request& r) const;
    size_t shareable(const Request& r, const std::vector<uint32_t>& t, const std::vector<RowClass>& classes, const infer::Sequence& seq) const;
    size_t shareable(const Request& r, const std::vector<uint32_t>& t, const std::vector<RowClass>& classes, size_t held, std::optional<size_t> kept) const;
    // A donor the device tier evicted, its history in host memory (Model::save_host): the tokens and row classes it holds, which a request matches as it matches a device donor.
    struct HostDonor {
        uint64_t id = 0;
        std::vector<uint32_t> tokens;
        std::vector<RowClass> classes;
        infer::HostHistory history;
        bool superseded = false;   // as Donor::superseded
        bool back = false;         // as Donor::back
        uint64_t disk = 0;         // the file that makes the same history whole on disk (DiskTier::on_disk), 0 while part of it is not there
        mutable std::vector<std::string> digests;   // its whole blocks' digests, once asked for (digests_of)
        bool from_disk = false;    // read back from disk, so a promotion of it completes a restore from disk
        std::filesystem::file_time_type used = std::filesystem::file_time_type::clock::now();   // its last use: made, renewed or promoted, which its file keeps
    };
    // A message boundary (docs/SPECULATIVE.md, section 2, Host tier): a job's donor's state alone (Model::save_host without blocks), at its checkpoint where the next user message starts, with the tokens and row classes below it.
    // A request that edits or regenerates that message forks it with the blocks of a history holding the same rows, a donor or a host donor of the same conversation (Model::fork with a state).
    struct Boundary {
        uint64_t id = 0;
        std::vector<uint32_t> tokens;
        std::vector<RowClass> classes;
        infer::HostHistory state;
        uint64_t disk = 0;         // as HostDonor::disk
        mutable std::vector<std::string> digests;   // as HostDonor::digests
        std::filesystem::file_time_type used = std::filesystem::file_time_type::clock::now();   // as HostDonor::used
        bool back = false;         // as HostDonor::back, the conversation's whose state it is
        bool superseded = false;   // as HostDonor::superseded, a cut having orphaned it
    };
    // Defined below, out of the class.
    void keep_boundary(Donor& d);
    void keep_boundary(infer::Sequence& seq, const std::vector<uint32_t>& tokens, const std::vector<RowClass>& classes, bool back);
    Boundary* find_bound(uint64_t id);
    void drop_bound(size_t i);
    Rank rank(const Donor& d) const;
    Rank rank(const HostDonor& h) const;
    Rank rank(const Boundary& b) const;
    bool writing(const Donor& d) const;
    std::vector<std::pair<bool, uint64_t>> host_victims(const Rank& above, bool* outranked = nullptr) const;
    bool host_room_for(size_t bytes, const Rank& above);
    bool holds(const Boundary& b, const std::vector<uint32_t>& t, const std::vector<RowClass>& classes, size_t held) const;
    bool holds(const std::vector<uint32_t>& bt, const std::vector<RowClass>& bc, size_t n, const std::vector<uint32_t>& t, const std::vector<RowClass>& classes,
               size_t held) const;
    size_t copy_length(const Donor& d) const;
    void write_back(Donor& d);
    bool written_soon(bool boundary, uint64_t id) const;
    void drop_host(size_t i);
    size_t held_below(const std::vector<uint32_t>& tokens, const std::vector<RowClass>& classes, size_t n, uint64_t other = 0) const;
    // Why job j, about to give way, was not handed to a waiting request (hand_jobs), in that function's own terms; under the lock.
    const char* not_handed(const Lines& lines, const Request& j) const {
        bool first = false, shares = false;
        for (const auto& r : lines.queue) {
            if (r->cancel_.load() || !r->classes_.empty()) continue;
            first = true;
            shares = shares || shareable(*r, j.prompt_, j.classes_, j.seq_) > 0;
        }
        return !first ? "no queued request at its first admission" : !shares ? "no queued request shares a state or block it kept" : "a history kept shares as much with the request";
    }
    // Defined below, out of the class.
    template <class... A>
    void gone(const std::vector<uint32_t>& history, Move kind, size_t first, size_t tokens, const char* line = nullptr, A... a);
    void lost();
    bool parks(bool boundary, uint64_t id) const;
    void release_entry(bool boundary, uint64_t id, infer::HostHistory& h);
    size_t parked_held() const;
    bool read_source(uint64_t id) const;
    std::vector<std::string> digests_of(const uint32_t* tokens, size_t n, const std::vector<RowClass>& classes) const;
    template <class Entry>
    const std::vector<std::string>& digests(const Entry& e, size_t n) const;
    const std::vector<std::string>& digests(const HostDonor& h) const;
    const std::vector<std::string>& digests(const Boundary& b) const;
    const std::vector<std::string>& digests(const Donor& d) const;
    void refresh_disk();
    void sync_disk();
    void settle_disk(const Lines& lines);
    void flush_disk(const Lines& lines, std::condition_variable& cv);
    uint64_t unwritten_bytes() const;
    bool write_from(const HostDonor& h);
    bool write_from(const Boundary& b);
    bool started(DiskTier::Wrote w);
    bool flush_one(std::vector<uint64_t>& tried, bool beyond = false);
    void restore_later(const std::vector<uint32_t>& tokens, size_t n);
    // Whether request r waits this round for `what`, a restore or a save of `n` tokens its admission needs, said once and counted; under the lock.
    // The bound is the scheduler's constant, raised each round for a save of `remaining` bytes (0 or more) being written at `rate` bytes a second to the write's time (write_wait_ms), never lowered.
    bool waits_for_room(Request& r, size_t n, const char* what, double remaining = -1, double rate = 0) {
        const auto now = std::chrono::steady_clock::now();
        if (!r.room_wait_) {
            r.room_wait_ = true;
            r.room_since_ = now;
            r.room_until_ = now + wait_bound_;
            std::fprintf(stderr, "server: cache: a request waits for %s of %zu tokens before it is admitted\n", what, n);
        }
        if (remaining >= 0 && rate > 0) {
            const double ms = write_wait_ms(remaining, rate, std::chrono::duration<double, std::milli>(now - r.room_since_).count(), recompute_row_ms_.value_or(prompt_ms_row_) * (double)n);
            r.room_until_ = std::max(r.room_until_, r.room_since_ + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(ms)));
        }
        if (now < r.room_until_) return true;
        if (!r.room_timed_out_) ++room_wait_timeouts_;
        r.room_timed_out_ = true;
        return false;
    }
    // Whether a history of `n` tokens is one the host tier cannot keep, its copy larger than the whole tier, so the disk is its only save (write_back, disk_round).
    bool beyond_host(size_t n) const { return model_.host_bytes(n) > host_cap_; }
    // The most reads from disk in flight (docs/DISK-TIER.md, Restore).
    static constexpr size_t kDiskReads = 2;
    void count_wait(Request& r);
    void end_wait(Request& r);
    void finish_read(const Lines& lines, uint64_t id, bool ok);
    void wake_readers(const Lines& lines, uint64_t id);
    void end_proof(const Lines& lines, uint64_t id);
    void unprove(const Lines& lines, Request& r);
    uint64_t start_stream(const std::vector<std::shared_ptr<Request>>& active, const DiskIndex::Path& path, const std::vector<uint32_t>& tokens, const std::vector<RowClass>& classes, const Request& r);
    void stream_round(const Lines& lines);
    bool stream_room(const std::vector<std::shared_ptr<Request>>& active, size_t length, bool spare) const;
    Clock::time_point read_until(size_t rows, Clock::time_point now) const;
    uint64_t start_read(const std::vector<std::shared_ptr<Request>>& active, const DiskIndex::Path& path, bool state_only, const std::vector<uint32_t>& tokens, const std::vector<RowClass>& classes, const Request& r, uint64_t source = 0, const DiskIndex::Path* whole = nullptr);
    // Defined below, out of the class.
    void write_ahead();
    bool promote(const std::vector<std::shared_ptr<Request>>& active, size_t i, bool spare = false, bool* room = nullptr, const std::shared_ptr<Request>& r = nullptr);
    // A history for an admitted request: its own donor `d` taken back whole (`take`), a fork at `shared` tokens of donor `d` or of the running request `from`, or a fresh sequence.
    // A first admission records its rows: a forked prefix as its source recorded it, the rest of the prompt at the prompt's extent, then the generated tokens at extent 1 (a job's all at its prompt's extent); under the lock.
    void admit(Request& r, size_t d, size_t shared, bool take = false, const Request* from = nullptr, infer::HostHistory* state = nullptr) {
        {
            std::lock_guard<std::mutex> lk(r.m_);
            if (r.admitted_ == Request::Clock::time_point{}) r.admitted_ = Request::Clock::now();
        }
        const bool first = r.classes_.empty();
        r.donor_ = 0;
        if (take) {
            r.seq_ = std::move(donors_[d].seq);
            donors_.erase(donors_.begin() + (std::ptrdiff_t)d);
            ++r.taken_back_;
            ++taken_back_;
        } else {
            if (shared && state) {
                // A fork with a state whose copy fails reads the history from the start: the room admission made holds the whole history either way.
                try {
                    r.seq_ = quieted()->fork(donors_[d].seq, shared, *state);
                    ++bound_hits_;
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "server: a message boundary of %zu tokens was not forked (%s)\n", shared, e.what());
                    shared = 0;
                    r.seq_ = model_.make_sequence();
                }
            } else {
                r.seq_ = shared ? model_.fork(from ? from->seq_ : donors_[d].seq, shared) : model_.make_sequence();
            }
            if (shared && !from) donors_[d].used = std::filesystem::file_time_type::clock::now();
        }
        give_media(r);
        if (!first) return;
        if (shared) {
            r.reused_.store(shared);
            r.classes_ = clip(from ? from->classes_ : donors_[d].classes, shared);
        }
        const size_t p = r.prompt_.size();
        if (r.job_) {
            r.source_ = shared && !from ? donors_[d].id : 0;
            // A job reads every id it has at its prompt's extent, its last stretch open-ended, an image's rows under the image's digest.
            prompt_classes(r, shared, r.classes_);
            if (!r.classes_.empty() && r.classes_.back().end == p && r.classes_.back().image == std::array<uint8_t, 32>{} && r.classes_.back().extent == p)
                r.classes_.back().end = std::numeric_limits<size_t>::max();
            else
                r.classes_.push_back(RowClass{std::numeric_limits<size_t>::max(), p});
            r.reached_ = shared;
            return;
        }
        if (shared) {
            ++prefix_hits_;
            prefix_tokens_ += shared;
        }
        prompt_classes(r, shared, r.classes_);
        r.classes_.push_back(RowClass{std::numeric_limits<size_t>::max(), 1});
        r.reached_ = shared;
        r.rng_.seed(r.params_.seed);
        // One checkpoint where a follow-up turn would fork it: whole blocks within the stable prefix, never the last prompt token, past what the request forked.
        const size_t bt = model_.kv_block_tokens(), at = std::min(r.stable_, p - 1) / bt * bt;
        if (model_.checkpoint_slots() && at > shared) r.keep_at_ = at;
        // And a message boundary where its last user message starts, whole blocks, past what it forked, so an edit or a regenerate of that message leaves the boundary the next one forks.
        const size_t message = std::min(r.message_, p - 1) / bt * bt;
        if (model_.checkpoint_slots() && host_cap_ && message > shared) r.boundary_at_ = message;
    }
    // The stretches of a prompt read at its extent from `from` on, cut at each of its images' first and end rows, the image's rows carrying its digest, appended to `out`.
    static void prompt_classes(const Request& r, size_t from, std::vector<RowClass>& out) {
        const size_t p = r.prompt_.size();
        for (const infer::MediaRun& m : r.media_) {
            const size_t end = std::min(m.first + m.rows, p);
            if (end <= from) continue;
            if (m.first > from) out.push_back(RowClass{m.first, p});
            out.push_back(RowClass{end, p, m.digest});
            from = end;
        }
        if (from < p) out.push_back(RowClass{p, p});
    }
    // How a request not yet admitted would compute its prompt, which admission matches against what the tiers hold.
    static std::vector<RowClass> first_classes(const Request& r) {
        std::vector<RowClass> out;
        prompt_classes(r, 0, out);
        return out;
    }
    // The request's images given to a new sequence of it, a fork's or a restore's included, as far as they can be: those below its length without rows, and those with rows.
    // The first whose rows it lacks waits for its encode, which the request's slice turns run (encode_piece).
    void give_media(Request& r) {
        drop_encode(r);
        r.media_next_ = 0;
        give_ready_media(r);
    }
    // Gives r's sequence its images from media_next_ on until one it reads needs rows it lacks.
    void give_ready_media(Request& r) {
        for (; r.media_next_ < r.media_.size(); ++r.media_next_) {
            infer::MediaRun run = r.media_[r.media_next_];
            if (run.first + run.rows > r.seq_.length() && !run.encoded) return;
            model_.add_media(r.seq_, std::move(run));
            r.media_[r.media_next_].encoded.reset();
        }
    }
    // The one image encoding: its steps and the request it is for.
    ImageSteps encoding_;
    const Request* encoding_for_ = nullptr;
    size_t encoded_pieces_ = 0;
    std::chrono::steady_clock::time_point encode_began_{};
    // The ids a conversation's next turn begins with, given for request `of` (follow).
    struct Follow {
        std::shared_ptr<Request> of;
        std::vector<uint32_t> ids;
        bool whole;
    };
    // The job of request `of`, active or waiting, as an index into `active` or else into jobs_; npos for none.
    std::pair<size_t, size_t> job_of(const std::vector<std::shared_ptr<Request>>& active, const Request* of) const {
        for (size_t i = 0; i < active.size(); ++i)
            if (active[i]->job_ && active[i]->of_.get() == of) return {i, npos};
        for (size_t i = 0; i < jobs_.size(); ++i)
            if (jobs_[i]->of_.get() == of) return {npos, i};
        return {npos, npos};
    }
    // The donor of plan `t` whose write to disk a request would wait for (enter): the first it takes, but `skip`, the one it forks, whose rows the host tier cannot keep and whose files are not yet whole on disk; donors_.size() where none.
    // Under the lock.
    size_t awaited_write(const Taken& t, size_t skip) {
        if (!disk_ || !disk_->writing()) return donors_.size();
        for (size_t g : t.donors) {
            if (g == skip) continue;
            const Donor& v = donors_[g];
            const size_t n = copy_length(v);
            if (n && !v.superseded && beyond_host(n) && !disk_->on_disk(digests(v), model_.kv_block_tokens(), n, model_.keeps_state())) return g;
        }
        return donors_.size();
    }
    // Whether request r, refused beside the running jobs, would wait for a donor's write once they had gone: its room planned as `admits` plans it, the jobs' sequences counted as freed.
    // With no forked donor in that plan it may name a write `enter` would not wait for, which costs at most one block of one job, or miss one, which leaves the job giving way as before.
    bool waits_for_write(const Request& r, const std::vector<std::shared_ptr<Request>>& active) {
        if (r.job_) return false;
        std::lock_guard<std::mutex> lk(m_);
        std::vector<const infer::Sequence*> jobs;
        std::vector<size_t> free = room(active);
        for (const auto& a : active)
            if (a->job_) {
                jobs.push_back(&a->seq_);
                add(free, promised(*a));
            }
        if (jobs.empty()) return false;
        add(free, model_.freed(jobs).blocks);
        const std::vector<size_t> need = pools_.blocks_for(kGrowth.entry(history_tokens(r), r.params_.until_limit, (size_t)r.params_.max_tokens, r.gen_.size()));
        const Taken t = make_room(free, ranked(), npos, false, {}, 0, false, need, gain_of({}, npos, 0, false, jobs));
        return t.enough && awaited_write(t, npos) < donors_.size();
    }
    // Job i leaves the active set by the one path every job takes: its rows become a donor (park), which make_room gives up or copies out as any donor's; the donor's id, 0 where none was kept.
    // Only rows park cannot keep, below a whole block or past a model's checkpoint, are released, said as lost where no other holder has them; `what` says why in the line.
    uint64_t park_job(std::vector<std::shared_ptr<Request>>& active, const Lines& lines, size_t i, const char* what) {
        auto j = active[i];
        const size_t held = j->seq_.length();
        const std::optional<size_t> kept = model_.keeps_state() ? model_.checkpoint(j->seq_) : std::nullopt;
        size_t usable = 0, elsewhere = 0;
        {
            std::lock_guard<std::mutex> lk(m_);
            const size_t bt = model_.kv_block_tokens();
            usable = model_.keeps_state() ? kept.value_or(0) : held / bt * bt;
            elsewhere = held_below(j->prompt_, j->classes_, usable);
        }
        const uint64_t donor = park(active, lines, i, j->prompt_);
        j->classes_.clear();
        j->source_ = 0;
        std::lock_guard<std::mutex> lk(m_);
        if (donor) {
            std::fprintf(stderr, "server: cache: a job %s, its %zu rows kept as a donor, its state kept at %zu\n", what, usable, kept.value_or(0));
        } else {
            reprefill_released_ += held;
            std::fprintf(stderr, "server: cache: a job %s, %zu rows released, its state kept at %zu, %zu of them held elsewhere\n", what, held, kept.value_or(0), elsewhere);
            if (usable > elsewhere) gone(j->prompt_, Move::job_lost, elsewhere, usable, "server: cache: rows %zu to %zu of a history left memory with the job that alone held them\n", elsewhere, usable);
        }
        return donor;
    }
    // Drops the waiting jobs of the conversation `ids` continues: those whose whole ids it begins with (a newer job's, or an admitted request's prompt), and those reading for the same request `of`; under the lock.
    // Conversations that only share a start, as under one system prompt, keep their jobs.
    void stale(const std::vector<uint32_t>& ids, const Request* of, const char* why) {
        for (auto it = jobs_.begin(); it != jobs_.end();) {
            const std::vector<uint32_t>& p = (*it)->prompt_;
            const bool prefix = p.size() <= ids.size() && std::equal(p.begin(), p.end(), ids.begin());
            if (prefix || (of && (*it)->of_.get() == of)) {
                std::fprintf(stderr, "server: cache: a waiting job of %zu ids dropped, %s\n", p.size(), why);
                it = jobs_.erase(it);
            } else {
                ++it;
            }
        }
    }
    // Defined below, out of the class.
    void supersede(const Request& j, uint64_t kept);
    static bool supersedes(const Request& j, uint64_t id, const std::vector<uint32_t>& tokens);
    bool host_growing_ = true;            // its slabs are not all taken yet (grow_host), under the lock
    // How often the host's memory is looked at (host_pressure): pressure builds over seconds and the look is one small file read, so idle slabs outstay it by a second at most.
    static constexpr std::chrono::seconds kHostCheck{1};
    Clock::time_point host_check_{};      // the next look, under the lock
    // How long after idle slabs went back the pool takes none ahead: a host short again a second later means something else is taking memory, and slabs taken in between only go back again.
    static constexpr std::chrono::seconds kHostQuiet{30};
    Clock::time_point host_quiet_{};      // when idle growth may start again after the last give-back, under the lock
    size_t host_run_bytes_ = 0;           // the bytes given back since a run of give-backs began, said once it has ended
    Clock::time_point host_run_since_{};
    size_t host_alive_ = 0, host_yielded_ = 0;   // the pool's slabs alive at the last look and the bytes given back since the start, under the lock
    static constexpr size_t kHostGrow = 4;   // the fresh slabs an idle round takes, more only where the shortest copy needs more; a slab took about 21 ms on an MI50 host (docs/STATUS.md)
    std::deque<Donor> donors_;
    std::deque<HostDonor> host_;                  // under the lock, oldest first
    std::deque<Boundary> bounds_;                 // under the lock, oldest first, in host_held_
    uint64_t pinned_bound_ = 0;                   // the boundary a request being admitted forks (enter), which nothing drops meanwhile
    uint64_t promoting_ = 0;                      // the host entry being promoted, which no room it makes releases
    uint64_t reading_for_ = 0;                    // the host entry whose rows a state about to be read alone is for, while that read's room is made (read_source)
    size_t bound_hits_ = 0;                       // under the lock
    size_t host_held_ = 0, host_hits_ = 0, host_moved_ = 0;   // under the lock
    std::vector<Follow> follows_;                 // under the lock, the ids given since the last round
    std::vector<std::shared_ptr<Request>> restores_;   // under the lock, jobs restore_later made since follow_up last took them in
    bool restoring_ = false;   // under the lock: a donor a restore job read takes its slot, so the donor it evicts gets no restore job (restore_later)
    std::deque<std::shared_ptr<Request>> jobs_;   // under the lock, jobs waiting for a seat, oldest first
    size_t reprefills_ = 0, reprefill_rows_ = 0, reprefill_cancels_ = 0, reprefill_handed_ = 0;   // under the lock
    std::atomic<size_t> jobs_now_{0}, job_rows_left_{0};   // Stats::jobs and Stats::job_rows_left, as the round last counted them (count_jobs)
    size_t prefix_hits_ = 0, prefix_tokens_ = 0;   // under the lock
    uint64_t donor_ids_ = 0;    // under the lock
    std::vector<size_t> reserved_now_, donor_blocks_now_;            // under the lock: Stats::reserved and donor_blocks as note_room last found them
    // The disk tier (docs/DISK-TIER.md), under the lock: the slabs of a host entry dropped while written, and the tier's figures as stats gives them, kept past the tier itself.
    std::vector<infer::HostHistory> disk_parked_;
    DiskTier::Figures disk_now_;
    bool disk_measured_ = false;
    bool disk_wake_ = false;
    // Reads from disk in flight, each into slabs counted in host_held_, and what they gave.
    std::vector<uint64_t> idle_tried_;   // the donors an idle server has copied towards disk (flush_one), under m_
    struct DiskRead {
        uint64_t id = 0;
        infer::HostHistory history;
        bool state_only = false;   // a state alone, for blocks a history on the devices or in host memory holds
        uint64_t source = 0;       // the host entry holding those blocks, which no room releases while the read runs, 0 for a history on the devices
        std::vector<uint32_t> tokens;
        std::vector<RowClass> classes;
        Clock::time_point started{};   // the store's call made, from which its read-ready latency runs (Request::Restore::read_ms)
        double slabs_ms = 0;           // its slabs taken and zero-filled (alloc_host)
    };
    std::deque<DiskRead> disk_reads_;
    // A history read from disk straight into the devices (start_stream): the read, the history it fills, its tokens, classes, blocks reserved and start.
    // `ended` once its outcome is taken; it is kept until its readers and copies have ended.
    struct Streaming {
        uint64_t id = 0;
        std::unique_ptr<infer::RestoreInto> into;
        std::unique_ptr<StreamedRead> read;   // after `into`, which it fills, so it goes first
        std::vector<uint32_t> tokens;
        std::vector<RowClass> classes;
        Clock::time_point started;
        bool ended = false;
        bool restore = false;   // read for a job restore_later made
    };
    std::vector<Streaming> streams_;   // under the lock
    size_t lost_before_written_ = 0;   // Stats::lost_before_written, under the lock
    size_t uncopyable_ = 0;            // Stats::uncopyable, under the lock
    std::array<std::atomic<size_t>, (size_t)Move::count> moves_{};   // Stats::moves
    std::atomic<size_t> reprefill_released_{0};   // Stats::reprefill_released
    size_t disk_refused_ = 0;   // Stats::disk_refused, under the lock
    size_t disk_hits_ = 0, disk_waits_ = 0, disk_wait_timeouts_ = 0;
    uint64_t disk_read_bytes_ = 0;
    double disk_wait_ms_ = 0;
    Stats::RestoreTotals restore_totals_;   // under the lock
    // A promotion's copies not yet seen retired, asked once a round (copies_retired): their tickets, their enqueue and the request that took them, if any.
    struct Copies {
        std::vector<backend::Ticket> tickets;
        Clock::time_point at;
        std::shared_ptr<Request> r;
    };
    std::vector<Copies> copies_;   // the scheduler thread's
    double copy_block_ms_ = 0;         // under the lock, a forgetting mean of the milliseconds a block of a donor's copy off the devices for its write took (flush_one)
    uint64_t awaited_ = 0;             // under the lock, the donor whose write a request waits for (enter), written in whole pieces
    std::vector<uint64_t> turn_tried_; // under the lock, the donors the host tier cannot keep that are whole on disk or were refused their write (disk_round)
    size_t room_waits_ = 0, room_wait_timeouts_ = 0;   // under the lock, admissions that waited for a donor's write or a restore's room, and those whose wait reached kWaitBound
    double room_wait_ms_ = 0, room_wait_longest_ms_ = 0;
    std::chrono::steady_clock::time_point next_expiry_;   // under the lock, the next check of the entries' age
    std::unique_ptr<DiskTier> disk_;   // after the lock and the condition, which its threads' callbacks take
    infer::Model& model_;
    std::mutex& m_;
    const Pools pools_;                   // the model's cache pools, which never change size
    const size_t max_seqs_;               // the requests and jobs active at once at most
    const size_t host_cap_;               // the host tier's bytes at most
    const size_t limit_;                  // the tokens a request may hold (request_limit)
    const Clock::duration wait_bound_;    // the bound of a wait for a restore or a save while nothing better is measured: kWaitBound, or DiskOptions::wait in tests
    const std::optional<std::chrono::milliseconds> read_wait_;   // DiskOptions::read_wait
    const std::optional<double> recompute_row_ms_;   // DiskOptions::recompute_row_ms
    const Clock::time_point epoch_;       // the server's start, which the scheduler's clocks count from
    double prompt_ms_row_ = 0;            // the scheduler thread's, a pass's milliseconds a prompt row as the scheduler measured it (Scheduler::measure_prompt); 0 until a pass has
    WriteTurn turn_ = WriteTurn::whole;   // under the lock, how a turn-end write may go (write_turn)
    bool held_ = false, took_ = false;    // under the lock, a write waited for its turn in the last disk_round, and one took the turn (write_held, took_write_turn)
    size_t taken_back_ = 0;               // under the lock, resumes that took their donor back
    uint64_t admissions_ = 0;             // the scheduler thread's
    bool stopping_ = false;               // under the lock, the scheduler stops (Scheduler::stop)
    std::optional<std::chrono::steady_clock::time_point> idle_at_;   // when a server with nothing to do since may write ahead under keep (Scheduler::run), under the lock
    bool idled_ = false;                  // a round has run idle since idle_at_ was set, under the lock
};


// Takes fresh slabs of the host tier while the server has nothing to do, kHostGrow a round or the shortest copy's where that needs more, and gives them back at once to stand idle in the model's pool; whether more are to come.
// A fresh slab is zero-filled as it is taken, which a read from disk otherwise spends on this thread before it starts (docs/STATUS.md); the model counts the fresh ones device by device (Model::host_fresh), and this runs under the lock.
inline bool Cache::grow_host() {
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
inline void Cache::host_pressure() {
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
inline size_t Cache::own_donor(const Request& r) const {
    for (size_t d = 0; r.donor_ && d < donors_.size(); ++d)
        if (donors_[d].id == r.donor_) return d;
    return donors_.size();
}

// The one count of the device's room (docs/CACHE-PLAN.md, section 4): what is free in each pool now, the model's free blocks less the room the running requests promised and have not taken.
// A request has promised its reservation (`need_`) less the blocks its sequence holds, a pass's prepared blocks among them, so one in flight counts at what its pass took and its step is not subtracted twice.
inline std::vector<size_t> Cache::room(const std::vector<std::shared_ptr<Request>>& active) const {
    std::vector<size_t> free = model_.kv_free();
    for (const auto& r : active) {
        const std::vector<size_t> p = promised(*r);
        for (size_t s = 0; s < free.size(); ++s) free[s] -= std::min(free[s], p[s]);
    }
    return free;
}

inline std::vector<size_t> Cache::promised(const Request& r) const {
    std::vector<size_t> p(pools_.blocks.size(), 0);
    if (r.need_.empty()) return p;
    const std::vector<size_t> held = model_.kv_held(r.seq_);
    for (size_t s = 0; s < p.size(); ++s) p[s] = r.need_[s] > held[s] ? r.need_[s] - held[s] : 0;
    return p;
}

// make_room's gain over donors_ and `active`: what the model frees of every victim together, asked of the whole set and never victim by victim (Model::freed), with each paused request's promise.
// The donor `keep` a request forks keeps the `kept_len` tokens it shares, one it takes back (`take`) frees nothing, its blocks being the request's, and `also`, sequences going in any case, count for what they free beside the victims.
inline Gain Cache::gain_of(const std::vector<std::shared_ptr<Request>>& active, size_t keep, size_t kept_len, bool take, const std::vector<const infer::Sequence*>& also) const {
    const std::vector<size_t> base = also.empty() ? std::vector<size_t>() : model_.freed(also).blocks;
    return [this, &active, keep, kept_len, take, also, base](const Taken& t) {
        std::vector<const infer::Sequence*> set = also;
        std::vector<size_t> g(pools_.blocks.size(), 0);
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
inline void Cache::note_room(const std::vector<std::shared_ptr<Request>>& active) {
    const std::vector<size_t> free = room(active);
    reserved_now_.assign(free.size(), 0);
    for (size_t s = 0; s < free.size(); ++s) reserved_now_[s] = pools_.blocks[s] - free[s];
    std::vector<const infer::Sequence*> held;
    for (const Donor& d : donors_) held.push_back(&d.seq);
    for (const Streaming& st : streams_) held.push_back(&st.into->seq);
    donor_blocks_now_ = model_.freed(held).blocks;
}

// A free checkpoint slot for a keep beside `keeps` others the pass takes, the oldest donors that hold one giving theirs where none is free (make_room, checkpoint slots being one more pool).
// The donor `spare` a job forked keeps its slot, which its fork may still read, and a checkpoint is never forced, so false leaves it out.
inline bool Cache::checkpoint_room(size_t keeps, uint64_t spare) {
    if (model_.checkpoints_free() > keeps) return true;
    std::lock_guard<std::mutex> lk(m_);
    return checkpoint_room_held(keeps, spare);
}

// checkpoint_room under the lock.
inline bool Cache::checkpoint_room_held(size_t keeps, uint64_t spare) {
    if (model_.checkpoints_free() > keeps) return true;
    const size_t free = model_.checkpoints_free();
    // The donors in rank order; a donor frees its slot only where no fork still reads it, which the model counts (Model::freed).
    std::vector<size_t> at, order;
    for (size_t d : ranked())
        if (!spare || donors_[d].id != spare) {
            order.push_back(at.size());
            at.push_back(d);
        }
    const Taken t = make_room({0}, order, npos, false, {}, 0, false, {keeps - free + 1}, [&](const Taken& p) {
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


// How much of r's history a fork of `seq`, holding the history `t` computed as `classes` record, can give it: whole blocks of the same tokens over rows computed as r's were or, at its first admission, would be.
// A job's rows are at the class every longer prompt takes, it is within what `seq` holds, and on a model that keeps a state only as far as the checkpoint.
inline size_t Cache::shareable(const Request& r, const std::vector<uint32_t>& t, const std::vector<RowClass>& classes, const infer::Sequence& seq) const {
    return shareable(r, t, classes, seq.length(), model_.checkpoint(seq));
}

// The same for a history of `held` tokens whose state, on a model that keeps one, is kept at `kept`.
inline size_t Cache::shareable(const Request& r, const std::vector<uint32_t>& t, const std::vector<RowClass>& classes, size_t held, std::optional<size_t> kept) const {
    const size_t bt = model_.kv_block_tokens(), h = history_tokens(r);
    const std::vector<RowClass> first = first_classes(r);
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
inline void Cache::keep_boundary(Donor& d) { keep_boundary(d.seq, d.tokens, d.classes, d.back); }

// The boundary request r asked for, its pass retired: kept, or said not kept with what keep_boundary found; under the lock.
inline void Cache::asked_boundary(Request& r) {
    const std::optional<size_t> kept = model_.keeps_state() ? model_.checkpoint(r.seq_) : std::nullopt;
    const size_t before = bounds_.size(), bt = model_.kv_block_tokens();
    const uint64_t was = moves_[(size_t)Move::boundary_kept].load() + moves_[(size_t)Move::boundary_not_kept].load();
    keep_boundary(r.seq_, r.prompt_, r.classes_, r.reused() > 0);
    if (!host_cap_ || !model_.keeps_state() || bounds_.size() > before || moves_[(size_t)Move::boundary_kept].load() + moves_[(size_t)Move::boundary_not_kept].load() != was) return;
    const char* why = !kept ? "its history has no state kept once its pass retired" : !*kept || *kept % bt ? "the state kept is not at a whole block" : *kept > r.prompt_.size() ? "the state kept is past its prompt" : nullptr;
    if (why) moved(Move::boundary_not_kept, "server: cache: the message boundary a request asked for was not kept (%s, at %zu)\n", why, kept.value_or(0));
}

// The checkpoint of `seq`, holding `tokens` computed as `classes` record, kept as a boundary, the copy enqueued behind the passes that wrote it, within the room the host tier and the host's free memory leave; one already kept is renewed.
// Its room is the ranking's (host_room_for), the conversation's boundaries taking its use, so the state survives the history's fate; under the lock.
inline void Cache::keep_boundary(infer::Sequence& seq, const std::vector<uint32_t>& tokens, const std::vector<RowClass>& classes, bool back) {
    if (!host_cap_ || !model_.keeps_state()) return;
    const std::optional<size_t> kept = model_.checkpoint(seq);
    const size_t bt = model_.kv_block_tokens();
    if (!kept || !*kept || *kept % bt || *kept > tokens.size()) return;
    const size_t n = *kept;
    const auto now = std::filesystem::file_time_type::clock::now();
    for (Boundary& b : bounds_) {
        const auto& t = b.tokens;
        if (t.size() > n || !std::equal(t.begin(), t.end(), tokens.begin())) continue;
        // The conversation's boundaries take its use, so the ranking keeps them with it (rank).
        b.used = now;
        b.back = b.back || back;
        if (t.size() == n && alike(b.classes, classes, n) == n) {
            if (b.disk) disk_->renew(b.disk, true);
            return;
        }
    }
    const size_t bytes = model_.host_bytes(n, false);
    if (!host_room_for(bytes, Rank{back ? 2 : 1, now})) {
        moved(Move::boundary_not_kept, "server: a message boundary at %zu tokens was not kept in host memory (the host tier has no room it may give)\n", n);
        return;
    }
    // The entry is made before the copy, so nothing allocates between a copy enqueued and its entry: a throw before the copy, or from it, which releases what it took, leaves no entry.
    bounds_.emplace_back();
    Boundary& b = bounds_.back();
    try {
        b.id = ++donor_ids_;
        b.back = back;
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
inline Cache::Boundary* Cache::find_bound(uint64_t id) {
    for (Boundary& b : bounds_)
        if (b.id == id) return &b;
    return nullptr;
}

// The one ranking of what stays (docs/CACHE-PLAN.md, section 4): class 0 where a holder's going loses nothing (superseded, cut off, a boundary without rows, not the last holder), 1 a conversation that did not come back, 2 one that did.
// Within a class the least recently used goes first; every tier's victims are asked of it.
inline Cache::Rank Cache::rank(const Donor& d) const {
    const size_t n = copy_length(d);
    const bool kept = n && ((disk_ && disk_->on_disk(digests(d), model_.kv_block_tokens(), n, model_.keeps_state())) || held_below(d.tokens, d.classes, n, d.id) >= n);
    return {d.superseded || kept ? 0 : d.back ? 2 : 1, d.used};
}

inline Cache::Rank Cache::rank(const HostDonor& h) const {
    const size_t n = h.history.length;
    return {h.superseded || h.disk || (n && held_below(h.tokens, h.classes, n, h.id) >= n) ? 0 : h.back ? 2 : 1, h.used};
}

inline Cache::Rank Cache::rank(const Boundary& b) const {
    const bool rows = std::any_of(donors_.begin(), donors_.end(), [&](const Donor& d) { return holds(b, d.tokens, d.classes, d.seq.length()); }) ||
                      std::any_of(host_.begin(), host_.end(), [&](const HostDonor& h) { return holds(b, h.tokens, h.classes, h.history.length); });
    return {b.superseded || b.disk || !rows ? 0 : b.back ? 2 : 1, b.used};
}

// Whether donor d's write to disk is in flight, which pins it: the donor count and room take it last (A19).
inline bool Cache::writing(const Donor& d) const { return disk_ && disk_->writing_of(digests(d), model_.kv_block_tokens()).second > 0; }

// donors_ in rank order, least worth first, a donor whose write is in flight last; make_room takes its victims in this order.
inline std::vector<size_t> Cache::ranked() const {
    std::vector<std::pair<std::pair<bool, Rank>, size_t>> keys;
    for (size_t d = 0; d < donors_.size(); ++d) keys.push_back({{writing(donors_[d]), rank(donors_[d])}, d});
    std::stable_sort(keys.begin(), keys.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<size_t> order;
    for (const auto& k : keys) order.push_back(k.second);
    return order;
}

// The host tier's entries, boundaries and copies, in the order room releases them: ranked below `above`, least worth first, a write in flight last, never one pinned (a boundary being admitted, a copy being promoted, a read's source).
// With `outranked`, whether an entry was passed over for ranking at or above `above`.
inline std::vector<std::pair<bool, uint64_t>> Cache::host_victims(const Rank& above, bool* outranked) const {
    std::vector<std::pair<std::pair<bool, Rank>, std::pair<bool, uint64_t>>> keys;
    const auto add = [&](bool bound, uint64_t id, const Rank& r) {
        if (r < above) keys.push_back({{written_soon(bound, id), r}, {bound, id}});
        else if (outranked) *outranked = true;
    };
    for (const Boundary& b : bounds_)
        if (b.id != pinned_bound_) add(true, b.id, rank(b));
    for (const HostDonor& h : host_)
        if (h.id != promoting_ && !read_source(h.id)) add(false, h.id, rank(h));
    std::stable_sort(keys.begin(), keys.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<std::pair<bool, uint64_t>> out;
    for (const auto& k : keys) out.push_back(k.second);
    return out;
}

// Host room for `bytes` more, entries released one at a time in host_victims' order below `above`, ranked again after each, as a holder may become the last; whether the bytes fit. Under the lock.
inline bool Cache::host_room_for(size_t bytes, const Rank& above) {
    while (host_held_ + bytes > host_cap_) {
        const auto victims = host_victims(above);
        if (victims.empty()) break;
        const auto [bound, id] = victims.front();
        if (bound) {
            for (size_t i = 0; i < bounds_.size(); ++i)
                if (bounds_[i].id == id) { drop_bound(i); break; }
        } else {
            for (size_t i = 0; i < host_.size(); ++i)
                if (host_[i].id == id) { drop_host(i); break; }
        }
    }
    return host_held_ + bytes <= host_cap_;
}

// Boundary i out of host memory.
// Under the lock.
inline void Cache::drop_bound(size_t i) {
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

// The one question for reuse (docs/CACHE-PLAN.md, section 4): the longest restore point memory holds of `tokens` computed as `classes` record and its holder, passing over `skip`, changing nothing; without `self` the rows held.
// Shares are whole blocks over rows computed alike, the newest donor where several share as much, a host copy only past every donor, a boundary only with its rows held, and with `running` a running source (enter) only past all of them.
inline Cache::Served Cache::served(const std::vector<uint32_t>& tokens, const std::vector<RowClass>& classes, const Request* self, const std::vector<std::shared_ptr<Request>>* running, const std::vector<uint64_t>& skip) const {
    const size_t bt = model_.kv_block_tokens();
    const auto passed = [&](uint64_t id) { return std::find(skip.begin(), skip.end(), id) != skip.end(); };
    const auto share = [&](const std::vector<uint32_t>& t, const std::vector<RowClass>& c, size_t held, std::optional<size_t> kept) {
        size_t n = 0;
        const size_t limit = std::min({tokens.size(), t.size(), held});
        while (n < limit && t[n] == tokens[n]) ++n;
        n = alike(classes, c, n) / bt * bt;
        if (self && model_.keeps_state()) n = kept && *kept <= n && *kept % bt == 0 ? *kept : 0;
        return n;
    };
    Served best;
    for (size_t d = 0; d < donors_.size(); ++d) {
        if (passed(donors_[d].id)) continue;
        const size_t n = share(donors_[d].tokens, donors_[d].classes, donors_[d].seq.length(), model_.checkpoint(donors_[d].seq));
        if (n && n >= best.tokens) best = {Served::Kind::donor, d, n};
    }
    for (size_t h = 0; h < host_.size(); ++h) {
        if (passed(host_[h].id)) continue;
        const size_t len = host_[h].history.length;
        const size_t n = share(host_[h].tokens, host_[h].classes, len, model_.keeps_state() ? std::optional<size_t>(len) : std::nullopt);
        if (n > best.tokens) best = {Served::Kind::host, h, n};
    }
    if (!self) return best;
    // A boundary is a state alone, so a longer one without its rows does not hide a shorter one with them.
    for (size_t i = 0; i < bounds_.size(); ++i) {
        const Boundary& b = bounds_[i];
        if (passed(b.id)) continue;
        const size_t n = share(b.tokens, b.classes, b.state.length, std::optional<size_t>(b.state.length));
        if (n <= best.tokens) continue;
        const bool rows = std::any_of(donors_.begin(), donors_.end(), [&](const Donor& d) { return !passed(d.id) && holds(b, d.tokens, d.classes, d.seq.length()); }) ||
                          std::any_of(host_.begin(), host_.end(), [&](const HostDonor& h) { return !passed(h.id) && holds(b, h.tokens, h.classes, h.history.length); });
        if (rows) best = {Served::Kind::bound, i, n};
    }
    for (size_t i = 0; running && i < running->size(); ++i) {
        const Request& a = *(*running)[i];
        const bool source = a.job_ ? &a != self : self->job_ && &a == self->of_.get();
        if (!source || a.seq_.in_flight() || a.classes_.empty()) continue;
        const size_t n = share(a.job_ ? a.prompt_ : history(a), a.classes_, a.seq_.length(), model_.checkpoint(a.seq_));
        if (n > best.tokens) best = {Served::Kind::running, i, n};
    }
    return best;
}

// What request r asks served about: its history but the last token, which the pass giving its logits computes (a job, wanting none, all of it), under its own row classes or, at its first admission, its prompt's.
inline std::pair<std::vector<uint32_t>, std::vector<RowClass>> Cache::asked(const Request& r) const {
    const size_t h = history_tokens(r), limit = r.job_ ? h : h - 1;
    std::vector<uint32_t> tokens(limit);
    for (size_t k = 0; k < limit; ++k) tokens[k] = *token_ptr(r, k);
    return {std::move(tokens), r.classes_.empty() ? first_classes(r) : r.classes_};
}

// Whether a history of tokens `t` computed as `classes` record, `held` of them on hand, holds boundary b's rows: its tokens and row classes below b's position.
inline bool Cache::holds(const Boundary& b, const std::vector<uint32_t>& t, const std::vector<RowClass>& classes, size_t held) const {
    return holds(b.tokens, b.classes, b.state.length, t, classes, held);
}

// The same for the rows below `n` of tokens `bt` computed as `bc` record.
inline bool Cache::holds(const std::vector<uint32_t>& bt, const std::vector<RowClass>& bc, size_t n, const std::vector<uint32_t>& t, const std::vector<RowClass>& classes,
           size_t held) const {
    return held >= n && t.size() >= n && bt.size() >= n && std::equal(bt.begin(), bt.begin() + (std::ptrdiff_t)n, t.begin()) && alike(bc, classes, n) == n;
}


// Donor d's history copied to host memory as it leaves the devices (enqueued, not waited for), its whole blocks up to its checkpoint on a model that keeps a state, where the tier's cap and the host's free memory allow.
// Superseded host donors and then the oldest go first, but for a donor whose conversation did not come back (below); under the lock.
inline size_t Cache::copy_length(const Donor& d) const {
    const size_t bt = model_.kv_block_tokens();
    size_t n = std::min({d.seq.length(), d.tokens.size(), d.keep}) / bt * bt;
    if (model_.keeps_state()) {
        const std::optional<size_t> kept = model_.checkpoint(d.seq);
        n = kept && *kept <= n && *kept % bt == 0 ? *kept : 0;
    }
    return n;
}

// What write_back does with a copy of `n` tokens ranked `mine`, the one owner of whether a copy is kept: the room it takes, in the ranking's order below its own, and host memory for the slabs it takes anew.
inline Cache::HostRoom Cache::host_room(size_t n, const Rank& mine) const {
    HostRoom plan;
    const size_t bytes = model_.host_bytes(n);
    if (bytes > host_cap_) {
        plan.why = HostRoom::Why::larger;
        return plan;
    }
    size_t held = host_held_, parked = parked_held();
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
    const auto short_of = [&] { return held + bytes > host_cap_; };
    bool outranked = false;
    for (const auto& [bound, id] : host_victims(mine, &outranked)) {
        if (!short_of()) break;
        if (bound) {
            for (const Boundary& b : bounds_)
                if (b.id == id) release(b.state, true, id);
        } else {
            for (const HostDonor& h : host_)
                if (h.id == id) release(h.history, false, id);
        }
    }
    // Still short once every entry ranked below it is gone, the copy is refused for the entries that outrank it, or for those pinned (a boundary being admitted, the copy being promoted, a read's source).
    if (short_of()) {
        plan.why = outranked ? HostRoom::Why::outranked : HostRoom::Why::room;
        return plan;
    }
    const infer::Model::HostFit fit = model_.host_copy_fits(n, host_cap_ + parked, freed);
    if (fit == infer::Model::HostFit::limit) plan.why = HostRoom::Why::room;
    else if (fit == infer::Model::HostFit::memory) plan.why = HostRoom::Why::memory;
    return plan;
}

inline void Cache::write_back(Donor& d) {
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
            if (host_[i].disk) disk_->renew(host_[i].disk, host_[i].back);
            std::rotate(host_.begin() + (std::ptrdiff_t)i, host_.begin() + (std::ptrdiff_t)i + 1, host_.end());
            return;
        }
    }
    const HostRoom plan = host_room(n, Rank{d.superseded ? 0 : d.back ? 2 : 1, d.used});
    if (plan.why == HostRoom::Why::larger) {
        // Too large for the host tier: it is kept only where its files are already whole on disk, and otherwise lost, which is counted and said.
        uncopied();
        const bool whole = disk_ && disk_->on_disk(digests(d), model_.kv_block_tokens(), n, model_.keeps_state());
        // Whole on disk, a conversation that came back is read back to the devices by a job once room is spare, so its next turn waits for no read.
        if (whole && d.back && !stopping_) restore_later(d.tokens, n);
        if (whole) moved(Move::not_to_host);
        else gone(d.tokens, Move::not_to_host, held_below(d.tokens, d.classes, n, d.id), n, "server: a history of %zu tokens left the devices with no copy: %.1f MiB is more than the host tier holds%s\n", n, (double)model_.host_bytes(n) / (1 << 20), disk_ ? " and its files were not yet whole on disk" : "");
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
    if (plan.why == HostRoom::Why::outranked) {
        uncopied();
        gone(d.tokens, Move::not_to_host, held_below(d.tokens, d.classes, n, d.id), n, "server: a donor of %zu tokens was not kept in host memory (the entries there rank above it)\n", n);
        return;
    }
    if (plan.why == HostRoom::Why::room) {
        uncopied();
        gone(d.tokens, Move::not_to_host, held_below(d.tokens, d.classes, n, d.id), n, "server: a donor of %zu tokens was not kept in host memory (the tier's room is held by entries in use)\n", n);
        return;
    }
    if (plan.why == HostRoom::Why::memory) {
        uncopied();
        gone(d.tokens, Move::not_to_host, held_below(d.tokens, d.classes, n, d.id), n, "server: a donor of %zu tokens was not kept in host memory (the host has no memory for its slabs beside the reserve it keeps)\n", n);
        return;
    }
    // The entry is made before the copy, so nothing allocates between a copy enqueued and its entry.
    host_.emplace_back();
    HostDonor& h = host_.back();
    const Clock::time_point start = Clock::now();
    try {
        h.id = d.id ? d.id : ++donor_ids_;
        h.back = d.back;
        // The copy takes the donor's last use, not its own making, so the ranking keeps its place.
        h.used = d.used;
        h.tokens.assign(d.tokens.begin(), d.tokens.begin() + (std::ptrdiff_t)n);
        h.classes = clip(d.classes, n);
        quieted()->save_host(d.seq, n, h.history, host_cap_ + parked_held());
    } catch (const std::exception& e) {
        host_.pop_back();
        gone(d.tokens, Move::not_to_host, held_below(d.tokens, d.classes, n, d.id), n, "server: a donor of %zu tokens was not kept in host memory (%s)\n", n, e.what());
        uncopied();
        return;
    }
    host_held_ += h.history.held;
    host_moved_ += h.history.bytes;
    moved(Move::to_host, "server: a donor of %zu tokens kept in host memory, %.1f MiB, its copy enqueued in %.1f ms\n", n, (double)h.history.bytes / (1 << 20), ms_since(start));
    if (disk_) {
        refresh_disk();
        if (h.disk) disk_->renew(h.disk, h.back);
        write_ahead();
    }
}

// A job for the first `n` of `tokens`, which forks what holds them and reads the rest from disk into spare room (enter), none while a restore job's donor takes the slot (restoring_).
// follow_up takes it in with the round's other new jobs, since a donor leaves the devices while jobs_ is being walked; under the lock.
inline void Cache::restore_later(const std::vector<uint32_t>& tokens, size_t n) {
    if (restoring_) return;
    auto j = std::make_shared<Request>(std::vector<uint32_t>(tokens.begin(), tokens.begin() + (std::ptrdiff_t)n), SampleParams{});
    j->job_ = true;
    j->whole_ = true;
    restores_.push_back(std::move(j));
}

// Whether the entry `id` is the one whose write to disk is in flight: room takes it last among those it may take, since dropping it cancels the write (docs/DISK-TIER.md, Demotion).
inline bool Cache::written_soon(bool boundary, uint64_t id) const { return disk_ && disk_->writing_from(id, boundary) && !disk_->write_stopped(); }


// Host donor i out of host memory.
// Under the lock.
inline void Cache::drop_host(size_t i) {
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
inline void Cache::moved(Move kind, const char* line, A... a) {
    ++moves_[(size_t)kind];
    if (!line) return;
    if constexpr (sizeof...(A) == 0) std::fputs(line, stderr);
    else std::fprintf(stderr, line, a...);
}

// The most rows below `n` of `tokens`, in whole blocks, that a donor or a host entry holds but `other`, the holder going: what is left in memory of a history whose holder goes.
inline size_t Cache::held_below(const std::vector<uint32_t>& tokens, const std::vector<RowClass>& classes, size_t n, uint64_t other) const {
    const size_t bt = model_.kv_block_tokens();
    return served(std::vector<uint32_t>(tokens.begin(), tokens.begin() + (std::ptrdiff_t)std::min(n / bt * bt, tokens.size())), classes, nullptr, nullptr, other ? std::vector<uint64_t>{other} : std::vector<uint64_t>{}).tokens;
}

// Rows [first, tokens) of `history` leaving memory with their last holder, or a state of it at `tokens`: moved, and first given to a test that tells whose rows they were (on_move).
template <class... A>
inline void Cache::gone(const std::vector<uint32_t>& history, Move kind, size_t first, size_t tokens, const char* line, A... a) {
    if (on_move) on_move(Moved{kind, first, tokens, &history});
    moved(kind, line, a...);
}

// One restore point, a history or a boundary's state a later request could have forked, released while no file holds it: counted beside a disk tier while the server runs (Stats::lost_before_written).
// Called where the entry is released (drop_host, drop_bound) or a donor leaves with no copy made (write_back); under the lock.
inline void Cache::lost() {
    if (disk_ && !stopping_) ++lost_before_written_;
}

// Whether a host entry released now parks its slabs, its disk write being in flight (release_entry, host_room).
inline bool Cache::parks(bool boundary, uint64_t id) const { return disk_ && disk_->writing_from(id, boundary); }

// A host entry's slabs back to the model, or, for the entry whose disk write is in flight, held until the write has stopped, which a cancel asks of it; the room counts free at once.
// Under the lock.

inline void Cache::release_entry(bool boundary, uint64_t id, infer::HostHistory& h) {
    if (parks(boundary, id)) {
        disk_->cancel_write();
        disk_parked_.push_back(std::move(h));
        h = infer::HostHistory{};
        return;
    }
    model_.release_host(h);
}

// The slabs held beside the host tier's copies, which leave its limit for them: an entry's dropped while its write stops, which count free.
inline size_t Cache::parked_held() const {
    size_t n = 0;
    for (const infer::HostHistory& h : disk_parked_) n += h.held;
    if (disk_) n += disk_->held();
    return n;
}


// Whether host entry `id` holds the rows a state being read alone from disk is for, so no room releases it until that read ends (start_read).
// Under the lock.
inline bool Cache::read_source(uint64_t id) const {
    return id == reading_for_ || std::any_of(disk_reads_.begin(), disk_reads_.end(), [&](const DiskRead& rd) { return rd.source == id; });
}

// The digests of a history's whole blocks (DiskIndex::digests), each stretch of its rows under the key the model gives its extent (Model::row_class), which is what makes two histories' rows the same bits.
inline std::vector<std::string> Cache::digests_of(const uint32_t* tokens, size_t n, const std::vector<RowClass>& classes) const {
    std::vector<ClassRun> runs;
    for (const RowClass& c : classes) runs.push_back(ClassRun{c.end, model_.row_class(c.extent), c.image});
    if (runs.empty()) return {DiskIndex::root()};
    runs.back().end = std::numeric_limits<size_t>::max();
    return DiskIndex::digests(tokens, n, runs, model_.kv_block_tokens());
}

// An entry's own, kept with it.
template <class Entry>
inline const std::vector<std::string>& Cache::digests(const Entry& e, size_t n) const {
    if (e.digests.empty()) e.digests = digests_of(e.tokens.data(), std::min(e.tokens.size(), n), e.classes);
    return e.digests;
}

inline const std::vector<std::string>& Cache::digests(const HostDonor& h) const { return digests(h, h.history.length); }

inline const std::vector<std::string>& Cache::digests(const Boundary& b) const { return digests(b, b.state.length); }

inline const std::vector<std::string>& Cache::digests(const Donor& d) const { return digests(d, d.seq.length()); }

// Which host entries are whole on disk, looked at again after the index gained or lost a file.
// Under the lock.
inline void Cache::refresh_disk() {
    const size_t bt = model_.kv_block_tokens();
    for (HostDonor& h : host_) h.disk = disk_->on_disk(digests(h), bt, h.history.length, model_.keeps_state());
    for (Boundary& b : bounds_) b.disk = disk_->state_on_disk(digests(b), bt, b.state.length);
}

// After a call that may have changed the index: the entries' standing and the figures stats gives.
inline void Cache::sync_disk() {
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
inline void Cache::disk_round(const std::vector<std::shared_ptr<Request>>& active, const Lines& lines) {
    const bool idle = idle_at_ && std::chrono::steady_clock::now() >= *idle_at_;
    idled_ = idled_ || idle;
    disk_wake_ = false;
    if (!disk_) return;
    settle_disk(lines);
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
    stream_round(lines);
    prefetch(active, lines);
    write_ahead();
    // A donor the host tier cannot keep is written from its turn's end on, with or without --disk-cache-keep and while requests run, a piece at a time (docs/DISK-TIER.md, Demotion).
    if (disk_->can_write()) flush_one(turn_tried_, true);
    // Under --disk-cache-keep a server idle for kDiskIdle (`idle`, from run) writes what a stop would have to, one file at a time, so a stop finds little left (docs/DISK-TIER.md, Keeping entries across a restart).
    if (idle && lines.queue.empty() && lines.paused.empty() && disk_->can_write() && !flush_one(idle_tried_)) disk_->say_written();
    sync_disk();
}

// The calls the store has finished: a write's file into the index and the slabs of an entry dropped while written released, and each read that ended made a host entry or given up.
// Under the lock.
inline void Cache::settle_disk(const Lines& lines) {
    const DiskTier::Settled done = disk_->settle(model_);
    if (done.wrote) {
        for (infer::HostHistory& h : disk_parked_) model_.release_host(h);
        disk_parked_.clear();
    }
    for (const auto& read : done.reads) finish_read(lines, read.first, read.second);
    // The slabs the tier's own calls took beyond the host tier go back to the host.
    if (done.wrote || !done.reads.empty()) quieted()->trim_host(host_cap_ + parked_held());
    sync_disk();
}

// At a clean exit under --disk-cache-keep, what memory holds goes to disk for the next server within a stated bound, the most valuable first (the reverse of the order room takes entries in), device donors copied to host memory first.
// A write still running at the limit is cancelled, so only whole entries stay, each keeping its last use; on the scheduler thread, without the lock.
inline void Cache::flush_disk(const Lines& lines, std::condition_variable& cv) {
    std::unique_lock<std::mutex> lk(m_);
    settle_disk(lines);
    // A store not made yet, its model's digest still being read, takes nothing, so the stop does not wait for it.
    const bool made = disk_->readable();
    // The bound is stated before the first write: kDiskFlush, or half as long again as what is left would take at the store's measured rate, plus five seconds for the copies off the devices.
    const uint64_t bytes = unwritten_bytes();
    const double rate = disk_->write_rate();
    const auto bound = std::max<std::chrono::steady_clock::duration>(kDiskFlush, rate > 0 ? std::chrono::seconds((long long)(1.5 * (double)bytes / rate) + 5) : kDiskFlush);
    const auto until = std::chrono::steady_clock::now() + bound;
    if (made) std::fprintf(stderr, "server: writing %.1f MiB to disk for the next server, within %lld s\n", (double)bytes / (1 << 20), (long long)std::chrono::duration_cast<std::chrono::seconds>(bound).count());
    std::vector<uint64_t> tried;
    const char* why = made ? "" : ", the disk cache not made yet";
    while (made) {
        settle_disk(lines);
        if (!disk_->writing()) { why = ", writing having stopped"; break; }
        if (disk_->can_write() && !flush_one(tried)) { why = ", refused room on disk or host memory"; break; }
        if (!cv.wait_until(lk, until, [&] { return disk_wake_; })) { why = ", the bound reached"; break; }
        disk_wake_ = false;
    }
    disk_->cancel_write();
    const uint64_t left = unwritten_bytes();
    std::fprintf(stderr, "server: %zu entries kept on disk for the next server, %.1f MiB not written%s\n", disk_->index().files().size(), (double)left / (1 << 20), left || !made ? why : "");
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
inline uint64_t Cache::unwritten_bytes() const {
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
inline bool Cache::write_from(const HostDonor& h) { return started(disk_->write_next(model_, digests(h), h.history.length, h.back, h.used, &h.history, nullptr, h.id, false)); }

inline bool Cache::write_from(const Boundary& b) { return started(disk_->write_next(model_, digests(b), b.state.length, true, b.used, &b.state, nullptr, b.id, true)); }

inline bool Cache::started(DiskTier::Wrote w) {
    sync_disk();
    if (w == DiskTier::Wrote::started) moved(Move::write_started);
    return w == DiskTier::Wrote::started;
}

// The next write of what memory holds, for flush_disk and for an idle server under --disk-cache-keep: true once one is in flight, false when nothing is left to write.
// Most valuable first (docs/DISK-TIER.md), a device donor's missing blocks and state copied off the devices for it and no more; a donor refused room is in `tried`, not asked again that flush or idle period; under the lock.
inline bool Cache::flush_one(std::vector<uint64_t>& tried, bool beyond) {
    if (beyond) held_ = false;
    tried.erase(std::remove_if(tried.begin(), tried.end(), [&](uint64_t id) { return std::none_of(donors_.begin(), donors_.end(), [&](const Donor& d) { return d.id == id; }); }), tried.end());
    // The donor a request waits for first, so no other donor's segments go before it.
    for (const bool first : {true, false})
    for (const bool back : {true, false}) {
        for (size_t k = donors_.size(); k-- > 0;) {
            Donor& d = donors_[k];
            const size_t n = copy_length(d);
            if (!n || d.superseded || d.back != back || std::find(tried.begin(), tried.end(), d.id) != tried.end() || (beyond && !beyond_host(n))) continue;
            if (first != (beyond && awaited_ == d.id)) continue;
            // Beyond the host tier the copy off the devices goes as the round's turn says (write_turn): segments of a piece's time at the measured cost a block, one block before any is measured, beside replies only at background work's turn.
            // It goes whole while nothing runs or waits, and for the donor a request waits for (enter), which is asked first.
            size_t most = std::numeric_limits<size_t>::max();
            const bool foreground = beyond && awaited_ == d.id;
            if (beyond && !foreground && turn_ == WriteTurn::none) {
                held_ = true;
                continue;
            }
            if (beyond && !foreground && turn_ != WriteTurn::whole) most = (copy_block_ms_ > 0 ? std::max<size_t>(1, (size_t)(kPieceMs / copy_block_ms_)) : 1) * model_.kv_block_tokens();
            const Clock::time_point start = Clock::now();
            const DiskTier::Wrote w = disk_->write_next(*quieted(), digests(d), n, d.back, std::filesystem::file_time_type::clock::now(), nullptr, &d.seq, 0, false, most);
            if (started(w)) {
                if (beyond) {
                    const double ms = ms_since(start);
                    if (const size_t rows = disk_->pending_rows()) copy_block_ms_ = forget(copy_block_ms_, ms * (double)model_.kv_block_tokens() / (double)rows);
                    // A segment beside replies took background work's turn, which no later round's write takes again until the scheduler gives another.
                    if (!foreground && turn_ == WriteTurn::due) {
                        took_ = true;
                        turn_ = WriteTurn::none;
                    }
                }
                return true;
            }
            if (beyond && w == DiskTier::Wrote::refused)
                std::fprintf(stderr, "server: cache: a history of %zu tokens the host tier cannot keep was not written as its turn ended (refused room on disk or host memory)\n", n);
            if (w == DiskTier::Wrote::refused || (beyond && w == DiskTier::Wrote::nothing)) tried.push_back(d.id);
        }
        // With `beyond`, only the device donors the host tier cannot keep (beyond_host), each asked until it is whole on disk or refused, then left in `tried` while it lives.
        if (beyond || first) continue;
        for (size_t i = host_.size(); i-- > 0;)
            if (!host_[i].disk && !host_[i].superseded && host_[i].back == back && write_from(host_[i])) return true;
        for (size_t i = bounds_.size(); back && i-- > 0;)
            if (!bounds_[i].disk && write_from(bounds_[i])) return true;
    }
    return false;
}

// Whether a waiting request has something for the round: no read or room to wait for, its wait past its bound, or its client gone; the store's finished calls wake the round otherwise.
inline bool Cache::ready(const std::deque<std::shared_ptr<Request>>& q) {
    const auto now = std::chrono::steady_clock::now();
    return std::any_of(q.begin(), q.end(), [&](const std::shared_ptr<Request>& r) {
        return ((!r->disk_wait_ || now >= r->disk_until_) && (!r->room_wait_ || now >= r->room_until_)) || r->cancel_.load();
    });
}

// When the loop must wake for the disk tier: the earliest bound of a request waiting for a read, the next check of the entries' age, and the start of an idle server's writes.
inline std::optional<std::chrono::steady_clock::time_point> Cache::next_disk_bound(const Lines& lines) {
    std::optional<std::chrono::steady_clock::time_point> t;
    // A streamed read's copies retire with no event, so the loop looks again a millisecond on while one runs (stream_round).
    if (!streams_.empty()) t = std::chrono::steady_clock::now() + std::chrono::milliseconds(1);
    if (disk_ && disk_->max_age() && disk_now_.files && (!t || next_expiry_ < *t)) t = next_expiry_;
    for (const auto* q : {&lines.queue, &lines.paused, &jobs_})
        for (const auto& r : *q)
            if (r->disk_wait_ && r->disk_until_ != std::chrono::steady_clock::time_point::max() && (!t || r->disk_until_ < *t)) t = r->disk_until_;
    // Under keep, requests wait for the disk tier's start at most the wait's bound from the server's (reading).
    if (disk_ && disk_->keeps() && !disk_->settled()) {
        const auto bound = epoch_ + wait_bound_;
        if (!t || bound < *t) t = bound;
    }
    // A request waiting for a donor's write or for room for a restore ends its wait at its bound (waits_for_room).
    for (const auto* q : {&lines.queue, &lines.paused})
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
inline bool Cache::reading(const Lines& lines, Request& r) {
    // Under keep a request's first admission waits for the disk tier's start, at most the wait's bound from the server's, so a turn sent as a restarted server listens reads what the server before kept (P4 in docs/CACHE-PLAN.md).
    if (disk_ && disk_->keeps() && r.classes_.empty() && !disk_->settled() && Clock::now() - epoch_ < wait_bound_) return true;
    // A job starts once prefetch has looked at it, so where the disk holds more than memory and its room is there it reads those rows rather than computing them (gap 5).
    if (r.job_ && !r.looked_ && !r.disk_wait_ && disk_ && disk_now_.files && disk_->readable()) return true;
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
        for (auto* q : {&lines.queue, &lines.paused, &jobs_})
            for (const auto& o : *q)
                if (o->disk_wait_ == id) return true;
        return false;
    };
    for (Streaming& s : streams_)
        if (s.id == id && !s.ended && !waits()) s.read->cancel();
    return false;
}

// A request's wait for a read ends: its time summed for the server and on the request (Request::Restore::wait_ms).
inline void Cache::end_wait(Request& r) {
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - r.disk_since_).count();
    disk_wait_ms_ += ms;
    r.disk_wait_ = 0;
    std::lock_guard<std::mutex> lk(r.m_);
    r.restore_.part(r.restore_.wait_ms, ms, r.first_);
}

inline void Cache::count_wait(Request& r) {
    std::lock_guard<std::mutex> lk(r.m_);
    ++r.restore_.waits;
}

// Read `id` finished: a history read whole becomes a host entry of a conversation that came back, its files kept and renewed, and anything else releases its slabs; the requests waiting for it go on.
// Under the lock.
inline void Cache::finish_read(const Lines& lines, uint64_t id, bool ok) {
    size_t i = 0;
    while (i < disk_reads_.size() && disk_reads_[i].id != id) ++i;
    // A proof's files, read by the tier: its request goes on to compare them (prove).
    if (i == disk_reads_.size()) return wake_readers(lines, id);
    DiskRead rd = std::move(disk_reads_[i]);
    disk_reads_.erase(disk_reads_.begin() + (std::ptrdiff_t)i);
    // The read once for the server, and its latency to every request waiting for it, which may share it.
    const double read_ms = ms_since(rd.started);
    if (ok) {
        ++restore_totals_.reads;
        Stats::RestoreTotals::add(restore_totals_.read_ms, restore_totals_.read_largest_ms, read_ms);
        Stats::RestoreTotals::add(restore_totals_.slabs_ms, restore_totals_.slabs_largest_ms, rd.slabs_ms);
    }
    for (auto* q : {&lines.queue, &lines.paused, &jobs_})
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
    wake_readers(lines, id);
}

// The requests waiting for read or proof `id` wait no longer.
inline void Cache::wake_readers(const Lines& lines, uint64_t id) {
    for (auto* q : {&lines.queue, &lines.paused, &jobs_})
        for (auto& r : *q)
            if (r->disk_wait_ == id) end_wait(*r);
}

// Proof `id` given up at the tier, and those waiting for it look again.
// Under the lock.
inline void Cache::end_proof(const Lines& lines, uint64_t id) {
    disk_->abandon(id, model_);
    wake_readers(lines, id);
    sync_disk();
}

// Proves another build's files on r's path in the tier's order: r's first block compared, the path restored at its fork, its last block and state at the end compared (docs/DISK-TIER.md, Adopting another build's entries).
// The tier decides each comparison and the files' fate; r computes the rows and copies the block, and a proof that does not end equal leaves it computing its history from the start (unprove).
inline void Cache::prove(const Lines& lines, Request& r) {
    if (!disk_) return;
    const size_t bt = model_.kv_block_tokens(), most = std::numeric_limits<size_t>::max();
    for (;;) {
        std::optional<DiskTier::Proving> p;
        bool room = true;
        {
            std::lock_guard<std::mutex> lk(m_);
            p = disk_->proving(r.proof_, bt);
            if (p && p->read && !p->first && p->base && r.seq_.length() < p->fork && model_.keeps_state()) room = checkpoint_room_held(0);
        }
        if (!p || !p->read || !room) return unprove(lines, r);
        const size_t at = r.seq_.length(), stop = p->first ? p->first : p->end;
        if (!p->first && p->base && at < p->fork) {
            try { model_.reset(r.seq_); } catch (const std::exception&) {}
            try {
                r.seq_ = quieted()->restore_host(*p->base);
                give_media(r);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "server: another build's history of %zu tokens was not restored to prove it (%s)\n", p->fork, e.what());
                return unprove(lines, r);
            }
            r.unproved_ = true;
            continue;
        }
        r.prove_at_ = stop;
        r.prove_state_ = !p->first && model_.keeps_state();
        if (at < stop) return;
        if (at > stop && !p->first) return unprove(lines, r);
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
        if (!compared || !equal) return unprove(lines, r);
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
        // Only what loses nothing makes its room (rank's class 0).
        host_room_for(proved.held, Rank{1, {}});
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
        wake_readers(lines, r.proof_);
        sync_disk();
        r.proof_ = r.prove_at_ = 0;
        r.prove_state_ = r.unproved_ = false;
        r.boundary_kept_ = 0;
        return;
    }
}

// r's proof given up or ended unequal: the rows r restored from the unproved files go, so it computes its history from the start, and the tier lets the proof go.
inline void Cache::unprove(const Lines& lines, Request& r) {
    if (r.unproved_) try { model_.reset(r.seq_); } catch (const std::exception&) {}
    const uint64_t id = r.proof_;
    r.proof_ = r.prove_at_ = 0;
    if (r.unproved_) r.boundary_kept_ = 0;
    r.prove_state_ = r.unproved_ = false;
    std::lock_guard<std::mutex> lk(m_);
    end_proof(lines, id);
}

// Reads for waiting requests whose history the disk holds more whole blocks of than any history on the devices or in host memory, each request looked for once, at most kDiskReads in flight and max_seqs requests waiting.
// On a model that keeps a state the path ends at a state, the state alone being read where a history on the devices or in host memory holds its rows; a request waits at most as long as recomputing would take, under the lock.
inline void Cache::prefetch(const std::vector<std::shared_ptr<Request>>& active, const Lines& lines) {
    if (!disk_ || !disk_now_.files || !disk_->readable()) return;
    const size_t bt = model_.kv_block_tokens();
    size_t waiting = 0;
    for (const auto* q : {&lines.queue, &lines.paused, &jobs_})
        for (const auto& r : *q) waiting += r->disk_wait_ != 0;
    // Jobs waiting to start ask the same question as requests, after them, so a job starting again reads its history from disk rather than from row 0 (docs/CACHE-PLAN.md, gap 5).
    for (auto* q : {&lines.paused, &lines.queue, &jobs_})
        for (auto& r : *q) {
            if (r->disk_checked_ || r->disk_wait_ || r->cancel_.load() || waiting >= max_seqs_) continue;
            r->looked_ = true;
            // What memory serves the request, the running requests and jobs among it (N3).
            const auto ask = asked(*r);
            const std::vector<uint32_t>& mine = ask.first;
            const std::vector<RowClass>& own = ask.second;
            const Served held = served(mine, own, r.get(), &active);
            const size_t shared = held.tokens;
            // A job reads only where it could start, by enter's room for it: its need less what it forks, from free blocks and the donor it forks alone, so what it reads is on the devices at once.
            if (r->job_) {
                std::vector<size_t> fresh = pools_.blocks_for(r->prompt_.size());
                const std::vector<size_t> shares = pools_.blocks_for(shared);
                for (size_t k = 0; k < fresh.size(); ++k) fresh[k] -= std::min(fresh[k], shares[k]);
                const size_t keep = held.kind == Served::Kind::donor ? held.index : npos;
                const Taken t = make_room(room(active), ranked(), keep, true, {}, 0, false, fresh, gain_of({}, keep, keep == npos ? 0 : shared));
                if (!t.enough || std::any_of(t.donors.begin(), t.donors.end(), [&](size_t g) { return g != keep; })) continue;
            }
            // The request's history as it would compute it: its tokens but the last, a job's all of them, which it reads and keeps whole.
            const std::vector<uint32_t>& tokens = mine;
            const size_t limit = tokens.size();
            // Whether a donor or a host copy holds the first `n` rows, and which host copy where no donor does.
            const auto rows = [&](size_t n) { return served(std::vector<uint32_t>(tokens.begin(), tokens.begin() + (std::ptrdiff_t)n), own, nullptr, nullptr); };
            const std::vector<std::string> d = digests_of(tokens.data(), limit, own);
            DiskIndex::Path path = disk_->index().path(d, bt, limit, model_.keeps_state());
            const DiskIndex::Path whole = path;
            // A state higher than the blocks on disk reach serves where a history in memory holds its rows.
            const auto alone = model_.keeps_state() ? disk_->index().state_below(d, bt, limit) : std::nullopt;
            if (alone && alone->first > path.length && rows(alone->first).tokens >= alone->first) {
                path = DiskIndex::Path{};
                path.length = alone->first;
                path.state = alone->second;
            }
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
            bool state_only = false;
            if (model_.keeps_state()) {
                const Served held = rows(path.length);
                state_only = held.tokens >= path.length;
                // The host copy holding those rows is renewed and becomes the newest, so no room the read takes releases it (read_source).
                if (state_only && held.kind == Served::Kind::host) {
                    HostDonor& h = host_[held.index];
                    source = h.id;
                    if (h.disk) disk_->renew(h.disk, h.back);
                    std::rotate(host_.begin() + (std::ptrdiff_t)held.index, host_.begin() + (std::ptrdiff_t)held.index + 1, host_.end());
                }
            }
            // A history larger than the host tier is read straight into the devices, with no whole copy in host memory (start_read), and a stream already reading the same rows serves this request too.
            const bool streams = !state_only && beyond_host(path.length);
            // A read already reading the same rows serves this request too, a stream or a read into host memory alike (gap 6).
            const auto same = [&](const std::vector<uint32_t>& t, const std::vector<RowClass>& c) {
                return t.size() == path.length && std::equal(t.begin(), t.end(), tokens.begin()) && alike(c, own, path.length) == path.length;
            };
            uint64_t id = 0;
            for (const Streaming& s : streams_)
                if (streams && !s.ended && same(s.tokens, s.classes)) id = s.id;
            for (const DiskRead& rd : disk_reads_)
                if (!streams && rd.state_only == state_only && same(rd.tokens, rd.classes)) id = rd.id;
            size_t reads = disk_reads_.size();
            for (const Streaming& s : streams_) reads += !s.ended;
            if (!id && reads + 1 > kDiskReads) continue;
            // A stream takes its room on the devices as it starts, so one without that room now is looked at again next round, as the request's admission would be.
            if (!id && streams && !stream_room(active, path.length, r->job_)) continue;
            r->disk_checked_ = true;
            if (!id) id = start_read(active, path, state_only, tokens, own, *r, source, state_only && whole.length > shared ? &whole : nullptr);
            if (!id) continue;
            // The read started may be the shorter whole path start_read fell back to, or a stream, or a read another request started, so the wait is for that read's rows.
            size_t length = 0;
            for (const DiskRead& rd : disk_reads_)
                if (rd.id == id) length = rd.tokens.size();
            for (const Streaming& s : streams_)
                if (s.id == id) length = s.tokens.size();
            const auto now = std::chrono::steady_clock::now();
            r->disk_wait_ = id;
            r->disk_rows_ = length;
            r->disk_since_ = now;
            count_wait(*r);
            r->disk_until_ = read_until(length, now);
            ++disk_waits_;
            ++waiting;
        }
}

// Until when a read of `rows` rows started at `now` is waited for: the time they take to compute at the measured prompt rate, the wait's constant before a rate is measured, or DiskOptions::read_wait.
inline Cache::Clock::time_point Cache::read_until(size_t rows, Clock::time_point now) const {
    if (read_wait_) return now + std::chrono::duration_cast<Clock::duration>(*read_wait_);
    if (prompt_ms_row_ > 0) return now + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>((double)rows * prompt_ms_row_));
    return now + wait_bound_;
}

// Starts the read of a path larger than the host tier straight into a history on the devices, with no whole copy in host memory (A20, docs/DISK-TIER.md, Restore): its room taken as a promotion's is, a job's from free blocks alone.
// Its id, 0 where the room, a checkpoint slot or the history is not to be had, which is counted and said as a refused read; under the lock.
inline uint64_t Cache::start_stream(const std::vector<std::shared_ptr<Request>>& active, const DiskIndex::Path& path, const std::vector<uint32_t>& tokens, const std::vector<RowClass>& classes, const Request& r) {
    const size_t length = path.length;
    const bool spare = r.job_;
    const auto refused = [&](const char* why) {
        ++disk_refused_;
        moved(Move::read_refused, "server: a history of %zu tokens on disk was not read back (%s); its request reads its prompt\n", length, why);
        return uint64_t(0);
    };
    if (!stream_room(active, length, spare)) return refused("the devices have no room for it");
    // The checkpoint slot its state takes comes first, so no donor goes for its blocks where the slot is not to be had.
    if (model_.keeps_state() && !checkpoint_room_held(0)) return refused("no checkpoint slot for its state");
    const Taken t = make_room(room(active), ranked(), npos, false, {}, 0, false, pools_.blocks_for(length), gain_of({}));
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
inline bool Cache::stream_room(const std::vector<std::shared_ptr<Request>>& active, size_t length, bool spare) const {
    const Taken t = make_room(room(active), ranked(), npos, false, {}, 0, false, pools_.blocks_for(length), gain_of({}));
    return t.enough && (!spare || t.donors.empty());
}

// The streamed reads stepped once a round, the stages held only while full slots are sent: one whose copies have all retired becomes a donor its request forks, as a promoted one does.
// One that failed or was cancelled lets its waiters go at once and gives its room and slot back only once its readers and copies have ended, so nothing reuses blocks still being written; under the lock.
inline void Cache::stream_round(const Lines& lines) {
    // A stream's outcome for the server and for each request waiting for it (Request::Restore): its time from start to its last copy retired is its read, and a whole one completes a restore from disk.
    const auto record = [&](const Streaming& s, bool ok) {
        const double ms = ms_since(s.started);
        if (ok) {
            ++restore_totals_.reads;
            Stats::RestoreTotals::add(restore_totals_.read_ms, restore_totals_.read_largest_ms, ms);
        }
        for (auto* q : {&lines.queue, &lines.paused, &jobs_})
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
            while (donors_.size() >= max_seqs_) drop_donor(ranked().front(), true);
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
            wake_readers(lines, id);
            continue;
        }
        if (!s.ended) {
            s.ended = true;
            disk_->end_stream(s.id, s.read->failed_key());
            if (!s.read->why().empty()) moved(Move::read_failed, "server: cache: the read of %zu tokens from disk failed (%s); its request reads its prompt\n", s.into->length, s.read->why().c_str());
            if (!s.read->why().empty()) record(s, false);
            wake_readers(lines, s.id);
        }
        if (!s.read->idle()) {
            ++i;
            continue;
        }
        quieted()->reset(s.into->seq);
        streams_.erase(streams_.begin() + (std::ptrdiff_t)i);
    }
}

// Starts the read of a path into slabs it takes host room for as a write-back does, a history of `tokens` computed as `classes` record, for request r; the read's id, or 0 where the room or the memory is not to be had.
// A history larger than the whole host tier is streamed to the devices instead (start_stream); under the lock.
inline uint64_t Cache::start_read(const std::vector<std::shared_ptr<Request>>& active, const DiskIndex::Path& path, bool state_only, const std::vector<uint32_t>& tokens, const std::vector<RowClass>& classes, const Request& r, uint64_t source, const DiskIndex::Path* whole) {
    const size_t length = path.length;
    const bool blocks = !state_only;
    const size_t bytes = model_.host_bytes(length, blocks);
    if (blocks && bytes > host_cap_) return start_stream(active, path, tokens, classes, r);
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
    // A read is for a conversation that came back, so it ranks as one, newest of all.
    host_room_for(bytes, Rank{2, std::filesystem::file_time_type::clock::now()});
    reading_for_ = 0;
    if (host_held_ + bytes > host_cap_ && rd.source && whole) return start_read(active, *whole, false, tokens, classes, r);
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

// Once host entries not on disk hold more than three quarters of the host tier, the one the ranking would drop next is written while it stays a host entry.
// Room taken later releases entries on disk first (rank's class 0), so a demotion never waits for a write and writes nothing the host tier keeps (docs/DISK-TIER.md, Demotion); under the lock, one file at a time.
inline void Cache::write_ahead() {
    if (!disk_->can_write()) return;
    size_t unwritten = 0;
    for (const HostDonor& h : host_)
        if (!h.disk && !h.superseded) unwritten += h.history.held;
    for (const Boundary& b : bounds_)
        if (!b.disk) unwritten += b.state.held;
    if (unwritten <= host_cap_ / 4 * 3) return;
    for (const auto& [bound, id] : host_victims(Rank{3, {}})) {
        if (bound) {
            for (const Boundary& b : bounds_)
                if (b.id == id && !b.disk && !b.superseded && write_from(b)) return;
        } else {
            for (const HostDonor& h : host_)
                if (h.id == id && !h.disk && !h.superseded && write_from(h)) return;
        }
    }
}

// Host donor i back on the devices as a donor of its own, its blocks and, on a model that keeps a state, its checkpoint slot taken as a first admission takes room, evicting older device donors to host memory; the host entry stays, renewed.
// With `spare`, for a job, only into room no donor gives up; false, the devices as they were but for donors evicted, where the room or a slot is not there or the copy fails, with `room` false where the room was what lacked; under the lock.
inline bool Cache::promote(const std::vector<std::shared_ptr<Request>>& active, size_t i, bool spare, bool* room, const std::shared_ptr<Request>& r) {
    const uint64_t id = host_[i].id;
    // The device donors it evicts to host memory never take its room for theirs (promoting_ pins it, host_victims).
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
    const Taken t = make_room(this->room(active), ranked(), npos, false, {}, 0, false, pools_.blocks_for(entry()->history.length), gain_of({}));
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
    while (donors_.size() >= max_seqs_) drop_donor(ranked().front(), true);
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
inline void Cache::copies_retired() {
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
inline void Cache::drop_donor(size_t i, bool evicted) {
    Donor& d = donors_[i];
    if (evicted) write_back(d);
    try { model_.reset(d.seq); } catch (const std::exception&) {}
    donors_.erase(donors_.begin() + (std::ptrdiff_t)i);
}

// Job j's donor `kept` supersedes the other donors of its conversation, in the device tier and the host tier alike: the one it forked, the request's it reads the reply of and every one whose tokens its own begin with.
// Each goes to the front of its tier, oldest first among them, so it is the first to go when room is needed, and a superseded donor the devices evict is not copied to host memory; under the lock.
inline void Cache::supersede(const Request& j, uint64_t kept) {
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
inline bool Cache::supersedes(const Request& j, uint64_t id, const std::vector<uint32_t>& tokens) {
    return (id && (id == j.source_ || (j.of_ && id == j.of_->parked_))) ||
           (tokens.size() <= j.prompt_.size() && std::equal(tokens.begin(), tokens.end(), j.prompt_.begin()));
}

} // namespace server
