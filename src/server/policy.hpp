#pragma once
// The scheduler's policy core (docs/SERVER.md, the round): pool blocks, reservation growth, who gives up blocks, which stages a round records, decode entries per pass and logits rows, as free functions over plain data.
// The scheduler calls them with its requests and passes, and the server-passes CTest with a simulated executor's.
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "backends/backend.hpp"

namespace server {

// The cache pools as the ledger counts them: each pool's blocks and the tokens one of its blocks holds.
struct Pools {
    std::vector<size_t> blocks, block_tokens;
    // What `tokens` positions take in each pool, in that pool's own blocks, never more than the pool holds.
    std::vector<size_t> blocks_for(size_t tokens) const {
        std::vector<size_t> b(blocks.size());
        for (size_t s = 0; s < b.size(); ++s) b[s] = std::min(backend::blocks_for(tokens, block_tokens[s]), blocks[s]);
        return b;
    }
};

// The marks a server of `max_seqs` requests needs for drafts on devices whose decode kernels hold `columns`: a request drafting takes a column for its last pick and one a draft.
// So at most half the columns' requests draft in a pass, one at least (docs/SERVER.md, Drafts).
inline size_t draft_marks(size_t max_seqs, size_t columns) {
    return std::min(max_seqs, std::max<size_t>(1, columns / 2));
}

// The ledger's arithmetic over blocks per pool.
inline void add(std::vector<size_t>& to, const std::vector<size_t>& b) {
    if (to.size() < b.size()) to.resize(b.size(), 0);
    for (size_t s = 0; s < b.size(); ++s) to[s] += b[s];
}
inline void sub(std::vector<size_t>& from, const std::vector<size_t>& b) {
    for (size_t s = 0; s < b.size(); ++s) from[s] -= b[s];
}
inline std::vector<size_t> less(std::vector<size_t> from, const std::vector<size_t>& b) {
    sub(from, b);
    return from;
}

// A stretch of a history computed one way: the rows before `end`, from the stretch before it on, took this extent, whose class chooses a device's kernels (Model::row_class).
// A prompt takes its whole length, a generated token extent 1, and a forked prefix keeps the stretches its donor recorded.
struct RowClass {
    size_t end, extent;
};

// The stretch holding position `at`.
inline const RowClass& class_at(const std::vector<RowClass>& rows, size_t at) {
    size_t i = 0;
    while (rows[i].end <= at) ++i;
    return rows[i];
}
// The stretches of the first `n` positions.
inline std::vector<RowClass> clip(const std::vector<RowClass>& rows, size_t n) {
    std::vector<RowClass> out;
    for (size_t i = 0; i < rows.size() && (out.empty() || out.back().end < n); ++i) {
        out.push_back(rows[i]);
        out.back().end = std::min(out.back().end, n);
    }
    return out;
}

// How a request reserves room: a capped one its history and what it may still generate, an uncapped one `tokens` past its history, and again `tokens` past its next position whenever that position would pass what it holds.
// Admission and growth read the same `tokens`, so the first uncapped request a growth plan pauses always frees a whole step, and a plan never pauses more than one.
struct Growth {
    size_t tokens = 256;
    // The tokens a request reserves as it is admitted, first or on resuming.
    size_t entry(size_t history, bool uncapped, size_t max_tokens, size_t generated) const {
        return history + (uncapped ? tokens : max_tokens - generated);
    }
    // The blocks per pool an uncapped decoding request's growth step adds before a pass, whose cache holds `held` positions under the `need` blocks it has reserved; empty when no step falls due.
    std::vector<size_t> step(const Pools& pools, bool uncapped, bool decoding, size_t held, const std::vector<size_t>& need) const {
        if (!uncapped || !decoding) return {};
        const std::vector<size_t> next = pools.blocks_for(held + 1), to = pools.blocks_for(held + 1 + tokens);
        const auto has = [&](size_t s) { return s < need.size() ? need[s] : 0; };
        bool due = false;
        for (size_t s = 0; s < next.size(); ++s) due = due || next[s] > has(s);
        if (!due) return {};
        std::vector<size_t> more(to.size());
        for (size_t s = 0; s < more.size(); ++s) more[s] = to[s] > has(s) ? to[s] - has(s) : 0;
        return more;
    }
    // Turns at the pool's edge (docs/SERVER.md, turns at the pool's edge): the growth steps a request holds the last room for, so many that a change of turn, `swap_ms`, costs kTurnShare of the turn.
    // A tenth is a constant from the measured turns (at histories of 14k tokens 5 percent doubles a turn and 20 percent shortens none); never under a step, and one where either time is not yet measured.
    static constexpr double kTurnShare = 0.10;
    static size_t turn(double swap_ms, double step_ms) {
        if (!(swap_ms > 0) || !(step_ms > 0)) return 1;
        return (size_t)std::max(1.0, std::ceil(std::min(swap_ms / (kTurnShare * step_ms), 1e6)));
    }
// The time of a change of turn that `turn` takes: the smaller of the one just timed and the one before it, and the one timed where none was before.
// So one slow change does not lengthen the next turn and two in a row do.
    static double change(double before, double timed) { return before > 0 ? std::min(before, timed) : timed; }
    // Whether a request whose step falls due gives its room up instead: it has held the edge for `taken` steps while an uncapped request is paused (`waiting`) that would then fit, and its own history can be kept whole meanwhile (`kept`).
    static bool yields(size_t taken, size_t turn, bool waiting, bool kept) { return waiting && kept && taken >= turn; }
};

// A running request as make_room sees it: its first admission, whether it may be paused (an uncapped one), the blocks its history would keep as a donor once paused, per pool, and whether a pass in flight holds it.
struct Holder {
    uint64_t admission;
    bool uncapped;
    std::vector<size_t> kept;
    bool flying = false;
};

// What make_room takes, in the order it takes it: donors by index, then running requests to pause by index, each with whether the donor its history becomes goes too.
// With `wait` a request to pause is in flight, so the caller takes nothing and plans again at its next room.
struct Taken {
    bool enough = false;
    bool wait = false;
    std::vector<size_t> donors;
    std::vector<std::pair<size_t, bool>> paused;
};

// What dropping the victims of a plan together frees, per pool: asked of the whole set of victims so far, never of one victim alone, since two forks of one prefix free it only together (Model::freed).
using Gain = std::function<std::vector<size_t>(const Taken&)>;

// Who gives up blocks for whom (docs/SERVER.md, room by first admission): for `need` blocks a pool beside `free`, donors go first, oldest first, the one the request forks (`keep`) last or, with `keep_first`, first.
// Only growth (`preempt`) then pauses uncapped requests admitted after `after`, latest first, each giving its headroom and, if still short, its donor; nothing is taken unless the whole is enough.
inline Taken make_room(const std::vector<size_t>& free, size_t donors, size_t keep, bool keep_first, const std::vector<Holder>& active, uint64_t after, bool preempt,
                       const std::vector<size_t>& need, const Gain& gain) {
    Taken t;
    const auto gained = [&] { return t.donors.empty() && t.paused.empty() ? std::vector<size_t>(need.size()) : gain(t); };
    const auto fits = [&] {
        const std::vector<size_t> g = gained();
        for (size_t s = 0; s < need.size(); ++s)
            if (need[s] > (s < free.size() ? free[s] : 0) + (s < g.size() ? g[s] : 0)) return false;
        return true;
    };
    t.enough = fits();
    std::vector<size_t> order;
    if (keep < donors && keep_first) order.push_back(keep);
    for (size_t d = 0; d < donors; ++d)
        if (d != keep) order.push_back(d);
    if (keep < donors && !keep_first) order.push_back(keep);
    for (size_t i = 0; i < order.size() && !t.enough; ++i) {
        t.donors.push_back(order[i]);
        t.enough = fits();
    }
    if (preempt && !t.enough) {
        std::vector<size_t> later;
        for (size_t i = 0; i < active.size(); ++i)
            if (active[i].uncapped && active[i].admission > after) later.push_back(i);
        std::sort(later.begin(), later.end(), [&](size_t a, size_t b) { return active[a].admission > active[b].admission; });
        for (size_t i = 0; i < later.size() && !t.enough; ++i) {
            const Holder& h = active[later[i]];
            t.paused.push_back({later[i], false});
            t.wait = t.wait || h.flying;
            if ((t.enough = fits()) || std::all_of(h.kept.begin(), h.kept.end(), [](size_t b) { return b == 0; })) continue;
            t.paused.back().second = true;
            t.enough = fits();
        }
    }
    if (!t.enough) return Taken{};
    // A donor that frees nothing beside the other victims goes back, so one whose blocks a fork keeps is not taken for nothing; the order above stands for every other.
    const auto same = [](const std::vector<size_t>& a, const std::vector<size_t>& b) {
        for (size_t s = 0; s < std::max(a.size(), b.size()); ++s)
            if ((s < a.size() ? a[s] : 0) != (s < b.size() ? b[s] : 0)) return false;
        return true;
    };
    for (size_t i = t.donors.size(); i-- > 0;) {
        const std::vector<size_t> with = gained();
        const size_t d = t.donors[i];
        t.donors.erase(t.donors.begin() + (std::ptrdiff_t)i);
        if (!same(gained(), with)) t.donors.insert(t.donors.begin() + (std::ptrdiff_t)i, d);
    }
    return t;
}

// The host tier's default size (Scheduler, --host-cache-bytes): `seqs` histories of `history` bytes, within half of `free`, the host memory free once the model is loaded; none where that is unknown.
inline size_t host_cache_default(size_t history, size_t seqs, std::optional<size_t> free) {
    if (!free) return 0;
    const size_t cap = *free / 2;
    return history && seqs > cap / history ? cap : std::min(cap, history * seqs);
}

// The most tokens a history may have for a request to yield the pool's edge (Scheduler::yields keeps a history whole only where `bytes_of(its length)` is within the host tier's `cap`): whole blocks of `bt` up to `limit`, 0 where not one fits.
template <class Bytes>
size_t turn_tokens(size_t cap, size_t bt, size_t limit, Bytes bytes_of) {
    size_t lo = 0, hi = bt ? limit / bt : 0;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo + 1) / 2;
        if (bytes_of(mid * bt) <= cap) lo = mid;
        else hi = mid - 1;
    }
    return lo * bt;
}

// The longest history an idle step takes slabs for (Scheduler::grow_host): whole blocks of `bt` up to `limit`, its copy within `room` bytes and allocating at most `fresh` slabs anew, which `fresh_of` counts device by device.
// Where none is, the shortest history if `least`, the step having taken nothing yet, so a model whose shortest copy needs more slabs than a step takes still grows.
template <class Bytes, class Fresh>
size_t grow_tokens(size_t room, size_t fresh, size_t bt, size_t limit, bool least, Bytes bytes_of, Fresh fresh_of) {
    const size_t n = turn_tokens(room, bt, limit, [&](size_t t) { return fresh_of(t) > fresh ? std::numeric_limits<size_t>::max() : bytes_of(t); });
    return n || !least || !bt || bt > limit || bytes_of(bt) > room ? n : bt;
}

// The start line for a host tier of `cap` bytes that does not hold the longest history, which takes `need` bytes; `tokens` is what turn_tokens gives, 0 where no history fits and so none takes turns.
inline std::string short_tier_line(size_t cap, size_t need, size_t tokens) {
    const std::string sizes = "host memory for evicted prefixes holds " + std::to_string(cap >> 20) + " MiB and the longest history a request may have takes " + std::to_string((need + ((size_t)1 << 20) - 1) >> 20) + " MiB (--host-cache-bytes)";
    if (tokens) return "uncapped requests that fill the pool together take turns only while a request's history is within " + std::to_string(tokens) + " tokens, a longer one keeping the edge until it ends: " + sizes;
    return "uncapped requests that fill the pool together do not take turns, the one that waits waiting for the other's end: " + sizes;
}

// The start line for a model that keeps a state and a server with no checkpoint slot for it: Scheduler::yields refuses every turn then, whatever the host tier holds.
inline std::string no_checkpoint_line() {
    return "uncapped requests that fill the pool together do not take turns, the one that waits waiting for the other's end: the model keeps a state and the server no checkpoint for it (--state-checkpoints)";
}

// What the start line says for a tier of `cap` bytes, `need` the longest history's and `tokens` what turn_tokens gives, on a model that keeps a state or not with `slots` checkpoint slots; empty where every request takes turns.
// The missing checkpoint slot comes first, since it refuses every history before any length is asked.
inline std::string turns_line(size_t cap, size_t need, size_t tokens, bool keeps_state, size_t slots) {
    if (keeps_state && !slots) return no_checkpoint_line();
    return cap && cap >= need ? std::string() : short_tier_line(cap, need, tokens);
}

// A pass slot as the round sees it: whether a pass is in flight in it, that pass's place in formation order, the stages recorded, and whether its next stage is being recorded now, on a thread of that stage's (the scheduler's recorders).
struct Flight {
    bool live = false;
    uint64_t formed = 0;
    size_t ran = 0;
    bool recording = false;
    bool apart = false;   // a pass of generated rows alone in its own arena, which a stage takes between the pieces of another pass
};

// What a round records and retires, by slot, before it makes room and forms passes.
struct Steps {
    std::vector<std::pair<size_t, size_t>> advance;   // (slot, stage), in the order to record them
    std::vector<size_t> retire;                       // oldest first
};

// From the last stage down to the first, each stage records the oldest pass waiting for it, and every pass whose last stage an earlier round recorded retires, oldest first.
// Passes apart are a kind of their own: a stage holds one pass of each kind at most, each kind in its formation order, and passes retire in order within a kind.
inline Steps round_steps(const std::vector<Flight>& slots, size_t stages) {
    Steps st;
    const size_t none = slots.size();
    for (size_t s = stages; s-- > 0;) {
        for (const bool apart : {true, false}) {
            size_t pick = none;
            bool busy = false;
            for (size_t k = 0; k < slots.size(); ++k) {
                if (!slots[k].live || slots[k].ran != s || slots[k].apart != apart) continue;
                if (slots[k].recording) busy = true;
                else if (pick == none || slots[k].formed < slots[pick].formed) pick = k;
            }
            if (pick != none && !busy) st.advance.push_back({pick, s});
        }
    }
    for (size_t k = 0; k < slots.size(); ++k)
        if (slots[k].live && slots[k].ran == stages) st.retire.push_back(k);
    std::sort(st.retire.begin(), st.retire.end(), [&](size_t a, size_t b) { return slots[a].formed < slots[b].formed; });
    return st;
}

// The most decode entries a new pass takes when `decoders` requests decode, those in flight included, over `passes` passes in flight on `stages` stages: an even share, so every pass carries about as many rows however the requests arrived.
// Only passes that fill the stages share, since then a pass retires every round and a request held back joins the next; with fewer, a request held back could wait for most of a pass, and every ready one goes.
inline size_t decode_share(size_t decoders, size_t passes, size_t stages) { return passes < stages ? decoders : (decoders + passes - 1) / passes; }

// Whether a stage whose recording waits on its devices goes to that stage's thread, with `live` passes in flight, the stage's own among them.
// A tensor group's does, since its recording waits at every sum; a single device's only while another pass is in flight, since a pass alone has nothing to overlap with and the hand-over would only add to its time.
inline bool recorded_apart(bool grouped, size_t live) { return grouped || live > 1; }

// Whether generated rows ride the pass that reads prompt rows: where it costs them no more than their share would, the prompt pass taking no longer than their own pass over the share.
// Without a share they always ride; with prices not yet measured they go apart.
inline bool rides(double prompt_ms, double generated_ms, double share) {
    return share <= 0 || (prompt_ms > 0 && generated_ms > 0 && prompt_ms * share <= generated_ms);
}

// How long generated rows go on riding a prompt's passes: after kProbe ridden in a row they go apart for one pass, which a stage measures since no piece is held ahead while it waits, so a ride on a stale price ends there.
// While they ride no pass apart is measured, so without this a ride would last until the prompt's own price moved.
struct RideRun {
    size_t ridden = 0;
    static constexpr size_t kProbe = 8;
    bool allow(bool rides) const { return rides && ridden < kProbe; }
    void rode() { ++ridden; }
    void apart() { ridden = 0; }
};

// What the generated rows are owed of a stage's time beside a prompt's pieces, by the stage's thread's clock: a piece earns them their share of its time, a pass of theirs spends the rest of its own.
// A pass is priced as the least of the last kRecent measured: the credit stops at one pass at that price and no pass spends more than two (docs/SERVER.md, A generating request beside a prompt).
struct StageShare {
    double share = 0;
    double owed_ms = 0;        // at 0 or above the generated rows may take the stage
    size_t pieces = kQuiet;    // pieces since the last pass of generated rows
    static constexpr size_t kRecent = 8;
    double recent[kRecent] = {};   // the last passes measured, a ring
    size_t measured = 0;

    static constexpr size_t kQuiet = 32;
    // What a pass of generated rows takes there: the least of the last kRecent measured; 0 before any.
    double price() const {
        const size_t n = std::min(measured, kRecent);
        return n ? *std::min_element(recent, recent + n) : 0.0;
    }
    void piece(double ms) {
        owed_ms = std::min(owed_ms + share * ms, (1 - share) * std::max(price(), ms));
        ++pieces;
    }
    void generated(double ms) {
        const double known = price();
        recent[measured++ % kRecent] = ms;
        owed_ms -= (1 - share) * (known > 0 ? std::min(ms, 2 * known) : ms);
        pieces = 0;
    }
    // A pass of generated rows recorded behind a piece the device held ahead, whose own time is not seen: it spends the price, or a piece's time until one is measured.
    void unmeasured(double piece_ms) {
        const double known = price();
        owed_ms -= (1 - share) * (known > 0 ? known : piece_ms);
        pieces = 0;
    }
    // The stage stood still for the scheduler's own work on its devices, which is not the prompt's time either: the generated rows spend it, up to two passes owed.
    void held(double ms) { owed_ms = std::max(owed_ms - (1 - share) * ms, std::min(owed_ms, -2 * (1 - share) * price())); }
    // Whether the device may hold a piece ahead: once no pass of generated rows has come for kQuiet pieces; the stage's thread holds one only while none waits either.
    bool ahead() const { return pieces >= kQuiet; }
    // Whether generated rows may take the stage at this boundary.
    bool due() const { return share > 0 && owed_ms >= 0; }
};

// How generated rows are served while prompts are read: with the prompts' passes alone (off), by a fixed share of each stage's time (StageShare), or by the automatic rule (ServiceClock, PromptFloor).
struct Pacing {
    enum class Mode { off, fixed, automatic };
    Mode mode = Mode::off;
    double share = 0;   // the fixed share
    bool refused = false;   // off because the operator said so (--no-pacing), not because the placement feeds no stage in pieces; background work is then given no time beside waiting work
    static Pacing fixed(double share) { return share > 0 ? Pacing{Mode::fixed, share} : Pacing{}; }
    static Pacing automatic() { return {Mode::automatic, 0}; }
};

// The gap the automatic rule aims at between two services of generated rows at a stage while a prompt is read: ten tokens a second a reply, faster than a reader reads, so a stream reads as continuous on any hardware.
// It is a goal, not a bound: a piece and a service are each indivisible, and the counters say how often the gap was missed (docs/SERVER.md, A generating request beside a prompt).
inline constexpr double kServiceGapMs = 100;
// The part of a stage's accounted time prompts keep however long a service of generated rows takes: half, so neither side waits on the other for longer than its own time.
inline constexpr double kPromptFloor = 0.5;

// What generated rows owe prompts of a stage's time under the automatic rule: a service of theirs adds the floor's part of its whole time, a prompt piece pays off the rest of its own, and they take the stage only with nothing owed.
// So the services taken that way use at most (1 - floor) / floor of the pieces' accounted time and one service more; work charged without that gate (HeldTime) adds its own time.
struct PromptFloor {
    double owed_ms = 0;
    void service(double ms) { owed_ms += kPromptFloor * ms; }
    void piece(double ms) { owed_ms = std::max(0.0, owed_ms - (1 - kPromptFloor) * ms); }
    // Replies and prompts no longer wait together: what was owed is owed to nobody.
    void demand_ended() { owed_ms = 0; }
    bool clear() const { return owed_ms <= 0; }
};

// When a stage last began a pass serving generated rows, by a clock in milliseconds, and how the gaps between such passes met the target while replies and prompts both waited (`overlap`).
struct ServiceClock {
    double last_ms = -1e300;   // never
    double from_ms = 0;        // since when both wait
    bool overlap = false;
    uint64_t intervals = 0, late = 0;   // the gaps between two services that both began in one overlap, and those of them above the target
    uint64_t deferred = 0;              // passes that were due and waited for the floor, each once
    uint64_t overlaps = 0;              // the overlaps begun
    double longest_piece_ms = 0;
    // Whether a service is due at a piece boundary: at the boundary nearest the target, so where half of one more piece of `piece_ms` would pass it.
    // The gaps then centre on the target and are not all above it, at the price of a service up to half a piece early.
    bool due(double now_ms, double piece_ms) const { return now_ms - last_ms + piece_ms / 2 >= kServiceGapMs; }
    // A pass serving generated rows begins: ridden, apart or alone; the wait for an overlap's first service is no gap.
    void began(double now_ms) {
        if (overlap && last_ms >= from_ms) {
            ++intervals;
            if (now_ms - last_ms > kServiceGapMs) ++late;
        }
        last_ms = now_ms;
    }
    // A pass that was due waits for the floor: counted once, `counted` being the pass's own mark.
    void defer(bool& counted) {
        if (!counted) ++deferred;
        counted = true;
    }
    // Whether replies and prompts both wait changed; it gives whether the overlap ended.
    bool demand(bool both, double now_ms) {
        const bool ended = overlap && !both;
        if (both && !overlap) {
            from_ms = now_ms;
            ++overlaps;
        }
        overlap = both;
        return ended;
    }
};

// Whether a request counts as a prompt for the overlap: before its first token, and while its cache lacks more than its last token, as after a pause.
// Otherwise it is a reply, whether its rows are in a pass in flight or not.
inline bool reads_prompt(size_t generated, size_t cached, size_t history) { return !generated || cached + 1 < history; }

// The scheduler's holds of a stage, which may nest and follow each other while its thread stands still.
// It keeps the time of each outermost hold taken for the generated rows' own work, from where a prompt's pass stood parked for it, for the floor to charge; the wait for that boundary is the prompt's piece.
struct HeldTime {
    size_t depth = 0;
    bool generated = false, standing = false;
    double from_ms = 0, parked_ms = 0;
    double charge_ms = 0;   // not yet taken
    void begin(bool for_generated, double now_ms) {
        if (!depth++) {
            generated = for_generated;
            from_ms = now_ms;
        }
    }
    // The stage's thread stands still at a piece boundary, and goes on.
    void parked(double now_ms) {
        standing = true;
        parked_ms = now_ms;
    }
    void resumed() { standing = false; }
    void end(double now_ms) {
        if (!--depth && generated && standing) charge_ms += now_ms - std::max(from_ms, parked_ms);
    }
    double take() {
        const double ms = charge_ms;
        charge_ms = 0;
        return ms;
    }
};

// How long a request waits at most for a restore or a save it needs, until the server has measured the whole path (step 8 of docs/CACHE-PLAN.md): past it the request goes on and computes, or takes the room, said and counted.
// Production read 9.9 GiB back in 10 s; the bound leaves a write of that size its time too.
inline constexpr std::chrono::seconds kWaitBound{15};

// How long a request waits for a donor's write to disk in all, `elapsed_ms` into its wait, before it takes the donor's room (step 8 of docs/CACHE-PLAN.md): the write's remaining bytes at its measured rate, half as long again and 2 s more.
// Never longer than computing the rows again (`recompute_ms`); no rate, or no measure of recomputing, gives no bound, the caller keeping kWaitBound.
inline double write_wait_ms(double remaining_bytes, double bytes_per_s, double elapsed_ms, double recompute_ms) {
    if (bytes_per_s <= 0 || recompute_ms <= 0) return 0;
    return std::min(std::max(0.0, elapsed_ms) + 1.5 * 1000.0 * std::max(0.0, remaining_bytes) / bytes_per_s + 2000.0, recompute_ms);
}

// A write in flight gives its own rate once it has run a second or written 256 MiB, so a server's first write is not left to the start's probe; 0 before that.
inline double in_flight_rate(uint64_t bytes, double seconds) {
    if (seconds <= 0 || (seconds < 1.0 && bytes < (uint64_t(256) << 20))) return 0;
    return (double)bytes / seconds;
}

// What a measured cost a row keeps of its old value at each new measurement, a forgetting mean (forget); spec::PassCost weighs its passes by its own kForget.
inline constexpr double kCostKeep = 0.75;
inline double forget(double old, double now) { return old > 0 ? kCostKeep * old + (1 - kCostKeep) * now : now; }

// The part of the target gap background work may add to a reply's gap (promise P6 in docs/CACHE-PLAN.md).
inline constexpr double kBackgroundShare = 0.25;

// The time background work may still take of a stage in this service interval: work that reaches the device and is no request's own, as a read-ahead job's rows.
// A file read or write on a thread of its own touches no stage and takes none of it, and a read for a waiting request is foreground, as its request is.
struct BackgroundGrant {
    double used_ms = 0, since_ms = 0;   // spent in the service interval that began at since_ms by the stage's clock; a later service makes the allowance whole
    uint64_t grants = 0, overruns = 0;  // grants spent, and those whose cost passed what was left
    double total_ms = 0, largest_ms = 0;
    enum class Demand { none, prompts, replies };   // the foreground work that waits: none, prompts alone, or generated rows
    static constexpr double kUnbounded = 1e300;
    // What is left, in ms: unbounded with no foreground work waiting, where the caller bounds a call's work itself.
    // Nothing with pacing off while any waits, while the prompt floor is owed, and while generated rows wait unserved or with a service due; else the allowance unspent, and under replies no more than the time to the next service.
    double left_ms(Demand demand, bool paced, const ServiceClock& clock, bool floor_clear, double now_ms) const {
        if (demand == Demand::none) return kUnbounded;
        if (!paced || !floor_clear) return 0;
        const double allowance = kBackgroundShare * kServiceGapMs - (clock.last_ms == since_ms ? used_ms : 0);
        if (demand == Demand::prompts) return std::max(0.0, allowance);
        return std::max(0.0, std::min(allowance, kServiceGapMs - (now_ms - clock.last_ms)));
    }
    // The units of a consumer's work that fit `left` ms (Scheduler::background_left), at most `at_most`: `left` over the consumer's measured cost a unit.
    // One, its probe, where it has measured none and anything is left.
    static size_t units(double left, size_t at_most, double unit_ms) {
        if (left <= 0 || !at_most) return 0;
        if (left >= kUnbounded) return at_most;
        if (unit_ms <= 0) return 1;
        return (size_t)std::min((double)at_most, left / unit_ms);
    }
    // What granted work cost, whatever its unit, once: `left` is what left_ms gave when it was granted, and the clock says which service interval pays.
    void spent(double ms, double left, const ServiceClock& clock) {
        ++grants;
        if (ms > left) ++overruns;
        if (clock.last_ms != since_ms) used_ms = 0, since_ms = clock.last_ms;
        used_ms += ms;
        total_ms += ms;
        largest_ms = std::max(largest_ms, ms);
    }
};

// The foreground demand background work gives way to (promise P6 in docs/CACHE-PLAN.md): none only where at most one request asks and no other is active, queued or paused.
// A second asking request or another generating one is a reply's demand, and others reading their prompts or waiting for a seat or for room a prompt's.
inline BackgroundGrant::Demand demand_beside(size_t askers, size_t other_replies, size_t other_prompts, size_t waiting) {
    if (askers > 1 || other_replies) return BackgroundGrant::Demand::replies;
    return other_prompts || waiting ? BackgroundGrant::Demand::prompts : BackgroundGrant::Demand::none;
}

// Whether generated rows ride the pass that reads prompt rows under the automatic rule: where that pass, of a measured `prompt_ms`, takes no longer than the target gap.
inline bool rides_within_gap(double prompt_ms) { return prompt_ms > 0 && prompt_ms <= kServiceGapMs; }

// The most rows of a request's next prompt slice, and whether that slice closes its pass to other prompts' rows.
struct PromptSlice {
    size_t most;
    bool closes;
};

// A request's next prompt slice when it lacks `waiting` prompt rows on `stages` stages: a ubatch, but for a lone request (`alone`, having read `read_alone` rows) about a 2 * stages-th of what it lacks, at least kMinSlice rows.
// Once company comes, a prompt whose lone slices left it off a whole ubatch first takes the rows back to one, in a pass no other prompt shares (docs/STATUS.md, a lone prompt read by every stage).
inline PromptSlice prompt_slice(size_t waiting, size_t stages, size_t ubatch, bool alone, size_t read_alone) {
    static constexpr size_t kMinSlice = 128;
    if (!alone) return read_alone % ubatch ? PromptSlice{ubatch - read_alone % ubatch, true} : PromptSlice{ubatch, false};
    if (stages < 2) return {ubatch, false};
    const size_t cut = (waiting + 2 * stages - 1) / (2 * stages);
    return {std::min(ubatch, std::max(std::min(kMinSlice, ubatch), cut)), false};
}

// The logits rows of a context reserved for passes, which each pass takes a run of as it is formed: after the newest run, or from row 0 when that does not fit.
// Runs come back oldest first, a run given back early, an aborted pass's, once every run taken before it has, so a run always fits while it and the runs held want at most half the rows, and when no run is held all of them.
struct LogitRows {
    struct Run {
        size_t base, n;
        bool back;
    };
    size_t size = 0;
    std::deque<Run> runs;   // oldest first
};

// The logits rows of a context for `slots` passes in flight that each want `per_pass` rows at most: twice that once passes overlap, so a run always fits, and one pass's alone.
inline LogitRows logit_rows(size_t slots, size_t per_pass) {
    LogitRows rows;
    rows.size = std::min<size_t>(slots, 2) * per_pass;
    return rows;
}

// The first row of a run of `n` rows, or `rows.size` when none fits; no rows take nothing and start at row 0.
inline size_t take_rows(LogitRows& rows, size_t n) {
    if (!n) return 0;
    size_t base = rows.size;
    if (rows.runs.empty()) {
        if (n <= rows.size) base = 0;
    } else {
        const size_t tail = rows.runs.front().base, head = rows.runs.back().base + rows.runs.back().n;
        if (tail < head) {
            if (n <= rows.size - head) base = head;
            else if (n <= tail) base = 0;
        } else if (n <= tail - head) {
            base = head;
        }
    }
    if (base != rows.size) rows.runs.push_back({base, n, false});
    return base;
}

// The run starting at `base` comes back; no rows give nothing back.
inline void give_rows(LogitRows& rows, size_t base, size_t n) {
    if (!n) return;
    for (auto& r : rows.runs)
        if (r.base == base && !r.back) {
            r.back = true;
            break;
        }
    while (!rows.runs.empty() && rows.runs.front().back) rows.runs.pop_front();
}

// The order waiting requests are admitted in, one line: those paused for another's growth first (`line` 0), then yielders and queued requests merged by `line`, until one does not fit.
// A request whose client left goes where it waits (`gone`), and one being read from disk keeps its place while those behind it that fit pass it (docs/SERVER.md, turns at the pool's edge).
template <class Queue, class Seat, class Gone, class Reading, class Enter, class Line>
void admit_waiting(Queue& paused, Queue& queue, Seat seat, Gone gone, Reading reading, Enter enter, Line line) {
    for (size_t p = 0, q = 0; seat() && (p < paused.size() || q < queue.size());) {
        const bool first = p < paused.size() && (q == queue.size() || line(paused[p]) < line(queue[q]));
        Queue& from = first ? paused : queue;
        size_t& i = first ? p : q;
        if (gone(from, i)) continue;
        if (reading(from[i])) { ++i; continue; }
        if (!enter(from[i])) return;
        from.erase(from.begin() + (std::ptrdiff_t)i);
    }
}

} // namespace server
