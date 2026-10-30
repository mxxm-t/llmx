#pragma once
// The scheduler's policy core (docs/SERVER.md, the round): pool blocks, reservation growth, who gives up blocks, which stages a round records, decode entries per pass and logits rows, as free functions over plain data.
// The scheduler calls them with its requests and passes, and the server-passes CTest with a simulated executor's.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <tuple>
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

// A stretch of a history computed one way: the rows before `end`, from the stretch before it on, took this extent (Model::row_class), and are the rows of the image of `image`'s digest, zero for text (docs/src/server.md).
// A prompt takes its whole length, a generated token extent 1, and a forked prefix keeps the stretches its donor recorded.
struct RowClass {
    size_t end, extent;
    std::array<uint8_t, 32> image{};
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

// Where each message of a chat prompt ends in its rows, and its role: what reuse_miss names the message a difference falls in by; empty for a prompt without messages.
struct MessageSpans {
    std::vector<size_t> ends;
    std::vector<std::string> roles;
};

// A history the cache holds, as reuse_miss reads it: its tokens, how far below `n` its row classes agree with the request's, and on a model that keeps a state the furthest state it can be forked at, at or below `n`.
struct HeldHistory {
    const std::vector<uint32_t>* tokens = nullptr;
    std::function<size_t(size_t)> alike;
    std::function<size_t(size_t)> state;
};

// A first admission that reused under half of its prompt: how far the closest held history agrees with it, and what held the reuse below that (docs/SERVER.md, A turn that reuses little).
struct ReuseMiss {
    enum class Limit { tokens, classes, state, room };
    size_t prompt = 0, reused = 0, agreed = 0, at = 0;
    size_t message = 0, messages = 0;   // the message `agreed` falls in, of `messages`; none where messages is 0
    std::string role;
    Limit limit = Limit::tokens;
};

inline const char* limit_name(ReuseMiss::Limit l) {
    switch (l) {
    case ReuseMiss::Limit::tokens: return "tokens";
    case ReuseMiss::Limit::classes: return "classes";
    case ReuseMiss::Limit::state: return "state";
    case ReuseMiss::Limit::room: return "room";
    }
    return "tokens";
}

// The closest of `held` to `tokens`, which reused `reused`, with forks in whole blocks of `block`: each compare stops at the first differing token, and row classes and states are asked of the closest alone.
inline ReuseMiss reuse_miss(const std::vector<uint32_t>& tokens, size_t reused, const std::vector<HeldHistory>& held, size_t block, const MessageSpans& spans) {
    ReuseMiss m;
    m.prompt = tokens.size();
    m.reused = reused;
    const HeldHistory* best = nullptr;
    for (const HeldHistory& h : held) {
        const size_t limit = std::min(tokens.size(), h.tokens->size());
        size_t n = 0;
        while (n < limit && (*h.tokens)[n] == tokens[n]) ++n;
        if (!best || n > m.agreed) {
            best = &h;
            m.agreed = n;
        }
    }
    if (!spans.ends.empty()) {
        m.messages = spans.ends.size();
        m.message = std::min<size_t>((size_t)(std::upper_bound(spans.ends.begin(), spans.ends.end(), m.agreed) - spans.ends.begin()), spans.ends.size() - 1);
        if (m.message < spans.roles.size()) m.role = spans.roles[m.message];
    }
    m.at = m.agreed;
    if (!best || m.agreed < block) return m;
    const size_t alike = best->alike ? best->alike(m.agreed) : m.agreed, whole = alike / block * block;
    if (alike < m.agreed && whole <= reused) {
        m.limit = ReuseMiss::Limit::classes;
        m.at = alike;
        return m;
    }
    if (best->state) {
        const size_t s = best->state(whole);
        if (s <= reused && s < whole) {
            m.limit = ReuseMiss::Limit::state;
            m.at = s;
            return m;
        }
    }
    if (whole > reused) {
        m.limit = ReuseMiss::Limit::room;
        m.at = whole;
    }
    return m;
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

// Who gives up blocks for whom (docs/SERVER.md, room by first admission): for `need` blocks beside `free`, donors go in `order`, the scheduler's ranking, the forked one (`keep`) last or, with `keep_first`, first.
// Only growth (`preempt`) then pauses uncapped requests admitted after `after`, latest first, each giving its headroom and, if still short, its donor; nothing is taken unless the whole is enough.
inline Taken make_room(const std::vector<size_t>& free, const std::vector<size_t>& order, size_t keep, bool keep_first, const std::vector<Holder>& active, uint64_t after, bool preempt,
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
    std::vector<size_t> victims;
    const bool kept = std::find(order.begin(), order.end(), keep) != order.end();
    if (kept && keep_first) victims.push_back(keep);
    for (size_t d : order)
        if (d != keep) victims.push_back(d);
    if (kept && !keep_first) victims.push_back(keep);
    for (size_t i = 0; i < victims.size() && !t.enough; ++i) {
        t.donors.push_back(victims[i]);
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

// The host tier's default size (Scheduler, --host-cache-bytes): `seqs` histories of `history` bytes, within `cap`, the donors' share of the host budget (infer::host_shares); none where that is unknown.
inline size_t host_cache_default(size_t history, size_t seqs, std::optional<size_t> cap) {
    if (!cap) return 0;
    return history && seqs > *cap / history ? *cap : std::min(*cap, history * seqs);
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
};

// What a round records and retires, by slot, before it makes room and forms passes.
struct Steps {
    std::vector<std::pair<size_t, size_t>> advance;   // (slot, stage), in the order to record them
    std::vector<size_t> retire;                       // oldest first
};

// From the last stage down to the first, each stage records the oldest pass waiting for it, and every pass whose last stage an earlier round recorded retires, oldest first.
inline Steps round_steps(const std::vector<Flight>& slots, size_t stages) {
    Steps st;
    const size_t none = slots.size();
    for (size_t s = stages; s-- > 0;) {
        size_t pick = none;
        bool busy = false;
        for (size_t k = 0; k < slots.size(); ++k) {
            if (!slots[k].live || slots[k].ran != s) continue;
            if (slots[k].recording) busy = true;
            else if (pick == none || slots[k].formed < slots[pick].formed) pick = k;
        }
        if (pick != none && !busy) st.advance.push_back({pick, s});
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

// Whether a request counts as a prompt for the service rule: before its first token, and while its cache lacks more than its last token, as after a pause.
// Otherwise it is a reply, whether its rows are in a pass in flight or not.
inline bool reads_prompt(size_t generated, size_t cached, size_t history) { return !generated || cached + 1 < history; }

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

// The time a segment of a turn-end write may take beside prompts (Cache::flush_one), 50 ms, and the step a prompt's slices are measured and cut in (PriceTable).
inline constexpr double kPieceMs = 50;
inline constexpr size_t kPieceStep = 16;
// The depth band of rows read at history position `depth` (docs/SERVER.md, The price table): 0 below kBandTokens, then one band a doubling, so 4k to 8k tokens is band 1 and 128k to 256k band 6.
inline constexpr size_t kBandTokens = 4096;
inline size_t depth_band(size_t depth) {
    size_t b = 0;
    for (size_t d = depth / kBandTokens; d; d >>= 1) ++b;
    return b;
}

// A pass's work at a stage as the price table keys it: its rows of extent 1 (decode entries, verifies and a resume's recomputed tokens) with the band of the deepest, and each prompt slice by its row class, band and rows.
struct PassWork {
    struct Chunk {
        size_t cls = 0, band = 0, rows = 0;
    };
    size_t decode_rows = 0, decode_band = 0;
    std::vector<Chunk> chunks;
    void decode(size_t rows, size_t depth) {
        decode_rows += rows;
        decode_band = std::max(decode_band, depth_band(depth));
    }
    void chunk(size_t cls, size_t rows, size_t depth) { chunks.push_back({cls, depth_band(depth), rows}); }
};

// A stage's measured prices (docs/SERVER.md, The price table): a pass of decode rows alone by its band and rows, and a pass of one prompt slice alone by the slice's row class, band and rows in steps of kPieceStep.
// A cell keeps a forgetting mean, which predicts, and its last measure, which the slice search reads so that an overrun acts at once.
struct PriceTable {
    struct Cell {
        double ms = 0, last_ms = 0;
        uint64_t samples = 0;
    };
    std::map<std::pair<size_t, size_t>, Cell> decode;              // by (band, rows)
    std::map<std::tuple<size_t, size_t, size_t>, Cell> chunks;     // by (class, band, rows in steps)
    static size_t step_of(size_t rows) { return (rows + kPieceStep - 1) / kPieceStep * kPieceStep; }

    // A measured stage time: a pass of decode rows alone prices its decode cell and one slice alone its slice's; a pass of several parts prices nothing, which says so.
    bool observe(const PassWork& w, double ms) {
        if (!(ms >= 0)) return false;
        Cell* c = nullptr;
        if (w.chunks.empty() && w.decode_rows) c = &decode[{w.decode_band, w.decode_rows}];
        else if (w.chunks.size() == 1) {
            // A slice beside decode rows already priced takes what remains, so a prompt riding the replies' pass learns its price.
            const double d = w.decode_rows ? decode_ms(w.decode_band, w.decode_rows) : 0;
            if (w.decode_rows && !(d > 0 && ms > d)) return false;
            ms -= d;
            c = &chunks[{w.chunks[0].cls, w.chunks[0].band, step_of(w.chunks[0].rows)}];
        }
        if (!c) return false;
        c->ms = forget(c->ms, ms);
        c->last_ms = ms;
        ++c->samples;
        return true;
    }
    // Decode rows' price: their cell, else the nearest measured rows above in their band, else below; 0 where none is.
    // A band not yet measured takes the nearest band below, so a reply crossing into a deeper band is still priced.
    double decode_ms(size_t band, size_t rows) const {
        for (size_t b = band + 1; b-- > 0;) {
            const auto it = decode.lower_bound({b, rows});
            if (it != decode.end() && it->first.first == b) return it->second.ms;
            if (it != decode.begin() && std::prev(it)->first.first == b) return std::prev(it)->second.ms;
        }
        return 0;
    }
    // A slice's price for the quick rule (quick_prompt): its cell or the nearest measured above, else a line through the two largest measured below.
    // 0 where neither is measured, with `step` then the rows to read next to learn it: twice the largest measured below, at least kPieceStep.
    double slice_ms(const PassWork::Chunk& c, size_t& step) const {
        const size_t rows = step_of(c.rows);
        const auto same = [&](const auto& key) { return std::get<0>(key) == c.cls && std::get<1>(key) == c.band; };
        step = kPieceStep;
        const auto it = chunks.lower_bound({c.cls, c.band, rows});
        if (it != chunks.end() && same(it->first)) return it->second.ms;
        if (it == chunks.begin() || !same(std::prev(it)->first)) return 0;
        const auto hi = std::prev(it);
        step = 2 * std::get<2>(hi->first);
        if (hi == chunks.begin() || !same(std::prev(hi)->first)) return 0;
        const auto lo = std::prev(hi);
        const double r1 = (double)std::get<2>(lo->first), r2 = (double)std::get<2>(hi->first);
        const double slope = std::max(0.0, (hi->second.ms - lo->second.ms) / (r2 - r1));
        return hi->second.ms + slope * ((double)rows - r2);
    }
    double chunk_ms(const PassWork::Chunk& c) const {
        const size_t rows = step_of(c.rows);
        const auto same = [&](const auto& key) { return std::get<0>(key) == c.cls && std::get<1>(key) == c.band; };
        const auto it = chunks.lower_bound({c.cls, c.band, rows});
        if (it != chunks.end() && same(it->first)) return it->second.ms;
        if (it != chunks.begin() && same(std::prev(it)->first)) return std::prev(it)->second.ms * (double)rows / (double)std::get<2>(std::prev(it)->first);
        return 0;
    }
    // A pass's predicted stage time: its decode rows' price and each slice's, each measured as a pass of its own, so a pass of several parts is priced high by their fixed costs; 0 where a part has no price yet.
    double predict(const PassWork& w) const {
        double ms = 0;
        if (w.decode_rows) {
            const double d = decode_ms(w.decode_band, w.decode_rows);
            if (d <= 0) return 0;
            ms += d;
        }
        for (const PassWork::Chunk& c : w.chunks) {
            const double x = chunk_ms(c);
            if (x <= 0) return 0;
            ms += x;
        }
        return ms;
    }
    // A class and band's measured slices against `target_ms` by their last measures: the smallest over it (`hi`), the largest within it below that (`lo`), 0 for none, and whether any is measured.
    struct Bracket {
        size_t lo = 0, hi = 0;
        double hi_ms = 0;
        bool any = false;
    };
    Bracket bracket(size_t cls, size_t band, double target_ms) const {
        Bracket b;
        for (auto it = chunks.lower_bound({cls, band, 0}); it != chunks.end() && std::get<0>(it->first) == cls && std::get<1>(it->first) == band; ++it) {
            b.any = true;
            if (it->second.last_ms > target_ms && !b.hi) b.hi = std::get<2>(it->first), b.hi_ms = it->second.last_ms;
        }
        for (auto it = chunks.lower_bound({cls, band, 0}); it != chunks.end() && std::get<0>(it->first) == cls && std::get<1>(it->first) == band; ++it)
            if (it->second.last_ms <= target_ms && (!b.hi || std::get<2>(it->first) < b.hi)) b.lo = std::get<2>(it->first);
        return b;
    }
    // The rows of the next slice of class `cls` in band `band` within `target_ms`, at most `most`: the search over the measured cells (docs/SERVER.md, The price table).
    size_t chunk_rows(size_t cls, size_t band, double target_ms, size_t most) const {
        const Bracket b = bracket(cls, band, target_ms);
        size_t rows = kPieceStep;
        if (!b.any) {
            for (size_t below = band; below-- > 0;) {
                const Bracket p = bracket(cls, below, target_ms);
                if (!p.any) continue;
                rows = std::max(kPieceStep, p.lo);
                break;
            }
        } else if (!b.lo) {
            // Half the smallest over, or its rows scaled to the target where it is far over, as a band measured only at a whole ubatch is.
            rows = std::min(b.hi / 2, (size_t)((double)b.hi * target_ms / b.hi_ms)) / kPieceStep * kPieceStep;
        } else if (!b.hi) {
            rows = 2 * b.lo;
        } else {
            rows = b.hi - b.lo > kPieceStep ? (b.lo + b.hi) / 2 / kPieceStep * kPieceStep : b.lo;
        }
        return std::min(most, std::max(kPieceStep, rows));
    }
};

// The reply rate the service rule aims for while prompt work waits, a product objective and not a guaranteed rate (docs/SERVER.md, The service rule).
inline constexpr double kReplyTokensPerSecond = 8;
// The longest a reply's steps may lie apart while prompt work waits, however many tokens a step brings, so a drafting reply is delivered without long pauses; a reader's property, not the hardware's.
inline constexpr double kMaxStepGapMs = 500;
// A reply's tokens a step over its last kSteps steps, a verify's kept drafts and its pick or one decoded token; 1 until it has that many, so one lucky verify cannot widen the gap.
class StepTokens {
public:
    static constexpr size_t kSteps = 8;
    void add(size_t tokens) {
        sum_ += tokens - last_[at_];
        last_[at_] = tokens;
        at_ = (at_ + 1) % kSteps;
        seen_ += seen_ < kSteps;
    }
    double mean() const { return seen_ < kSteps ? 1.0 : std::max(1.0, (double)sum_ / (double)kSteps); }

private:
    std::array<size_t, kSteps> last_{};
    size_t sum_ = 0, at_ = 0, seen_ = 0;
};
// What the gap leaves a chunk pass at each of `stages` stages: a reply's pass meets one chunk at every stage, so the gap is the stages times a chunk and the replies' own pass.
inline double chunk_budget(double gap_ms, double replies_ms, size_t stages) {
    const double per_stage = gap_ms / (double)std::max<size_t>(1, stages);
    return per_stage > replies_ms ? per_stage - replies_ms : 0.0;
}
// A chunk's stage time while replies and prompt work both wait: the largest gap that keeps every reply at the objective, from the fewest tokens a step among them and at most kMaxStepGapMs, less the replies' pass at each stage.
// Where even the replies' pass leaves nothing the objective is unmet, and a chunk goes only to a prompt the wait bound makes due, taking one stage's share of that gap.
struct ChunkPlan {
    double gap_ms = 0, budget_ms = 0;
    bool met = true, capped = false;
};
inline ChunkPlan chunk_plan(double fewest_tokens_per_step, double replies_ms, size_t stages, double rate = kReplyTokensPerSecond) {
    const double by_tokens = 1000.0 * fewest_tokens_per_step / rate, gap = std::min(by_tokens, kMaxStepGapMs), budget = chunk_budget(gap, replies_ms, stages);
    const bool capped = by_tokens > kMaxStepGapMs;
    if (budget > 0) return {gap, budget, true, capped};
    return {gap, gap / (double)std::max<size_t>(1, stages), false, capped};
}
// How long a ready prompt goes without rows at most before it reads first, the bound of least-remaining-first (Share::wait_ms) the replays chose (docs/SERVER.md, Prompt shares).
inline constexpr double kPromptWaitMs = 10000;
// A lone prompt's next slice on `stages` stages when it lacks `waiting` rows: about a 2 * stages-th of them, at least 128 rows and at most a ubatch.
// So every stage of a split has a slice to read while the prompt is the only work; on one stage a ubatch.
inline size_t lone_slice(size_t waiting, size_t stages, size_t ubatch) {
    if (stages < 2) return ubatch;
    const size_t cut = (waiting + 2 * stages - 1) / (2 * stages);
    return std::min(ubatch, std::max(std::min<size_t>(128, ubatch), cut));
}
// Whether a prompt is quick while the reply objective is unmet: its whole remaining cost and the replies-only pass fit within the step gap's cap by the measured prices.
// A quick prompt runs whole in a pass shared with the replies, at most every other replies' pass, shortest first, so a newcomer never waits the wait bound and no reply step lies further apart than the cap.
inline bool quick_prompt(double remaining_ms, double replies_ms) { return remaining_ms > 0 && remaining_ms + replies_ms <= kMaxStepGapMs; }
// The passes a round keeps in flight of the `slots` reserved: one a stage, and the one more only while prompt rows and replies overlap, when a prompt's chunk continues beside a pass of replies.
inline size_t passes_in_flight(size_t slots, size_t stages, bool overlap) { return overlap ? slots : std::min(slots, std::max<size_t>(1, stages)); }
// Replies while prompt work waits go in as few passes as the decode kernels' columns hold them, leaving prompt work a slot of `slots`, and in the one pass where there is one.
inline size_t packed_passes(size_t decoders, size_t columns, size_t slots) {
    if (!decoders) return 0;
    const size_t n = columns ? (decoders + columns - 1) / columns : decoders;
    return slots > 1 ? std::min(n, slots - 1) : 1;
}
// The replies each of those passes takes.
inline size_t packed_share(size_t decoders, size_t passes) { return passes ? (decoders + passes - 1) / passes : decoders; }

// A prompt with rows left to read as the share rule sees it, the rule taking them in arrival order: its rows left and their price, how long it has been ready without rows and the stage time it has had.
struct ReadyPrompt {
    double remaining_ms = 0, idle_ms = 0, attained_ms = 0;
    size_t remaining_rows = 0;
};
// How prompt time is divided among the ready prompts: a rule that orders them or one that gives each a part (docs/SERVER.md, Prompt shares).
// `part` is the oldest's part or floor, `age` and `wait_ms` least remaining's aging and idle bound, and `short_rows` what today's order reads first.
struct Share {
    enum class Rule { today, oldest_first, oldest_keeps, equal, least_remaining, least_remaining_floor, least_attained };
    Rule rule = Rule::oldest_first;
    double part = 0.75;
    double age = 0;
    double wait_ms = 10000;
    size_t short_rows = 0;
};
// Each ready prompt's part of prompt time under a rule that divides it, by index into `p` in arrival order; empty for a rule that orders.
inline std::vector<double> share_parts(const Share& s, const std::vector<ReadyPrompt>& p) {
    using R = Share::Rule;
    if (s.rule != R::oldest_keeps && s.rule != R::equal && s.rule != R::least_remaining_floor) return {};
    std::vector<double> parts(p.size(), 0.0);
    if (p.size() == 1) parts[0] = 1;
    if (p.size() < 2) return parts;
    if (s.rule == R::equal) {
        std::fill(parts.begin(), parts.end(), 1.0 / (double)p.size());
        return parts;
    }
    size_t other = 1;
    if (s.rule == R::least_remaining_floor)
        for (size_t i = 2; i < p.size(); ++i)
            if (p[i].remaining_ms < p[other].remaining_ms) other = i;
    parts[0] = s.part;
    parts[other] = 1 - s.part;
    return parts;
}
// The prompt the next chunk goes to, an index into `p` (in arrival order): by the rule's order, or under a rule that divides, the largest credit (charge_share), the oldest of equals.
inline size_t next_prompt(const Share& s, const std::vector<ReadyPrompt>& p, const std::vector<double>& credit) {
    using R = Share::Rule;
    if (p.empty()) return 0;
    const auto least = [&](auto key) {
        size_t best = 0;
        for (size_t i = 1; i < p.size(); ++i)
            if (key(i) < key(best)) best = i;
        return best;
    };
    switch (s.rule) {
    case R::today:
        for (size_t i = 0; i < p.size(); ++i)
            if (p[i].remaining_rows <= s.short_rows) return i;
        return 0;
    case R::oldest_first:
        return 0;
    case R::least_remaining: {
        size_t idle = p.size();
        for (size_t i = 0; i < p.size(); ++i)
            if (p[i].idle_ms >= s.wait_ms && (idle == p.size() || p[i].idle_ms > p[idle].idle_ms)) idle = i;
        if (idle < p.size()) return idle;
        return least([&](size_t i) { return p[i].remaining_ms - s.age * p[i].idle_ms; });
    }
    case R::least_attained:
        return least([&](size_t i) { return p[i].attained_ms; });
    default: {
        // The largest credit among the prompts the rule gives a part, so one it gives none is never read by a rounding of the others' credits.
        const std::vector<double> parts = share_parts(s, p);
        const auto of = [&](size_t i) { return i < credit.size() ? credit[i] : 0.0; };
        size_t best = p.size();
        for (size_t i = 0; i < p.size(); ++i)
            if (parts[i] > 0 && (best == p.size() || of(i) > of(best))) best = i;
        return best < p.size() ? best : 0;
    }
    }
}
// A chunk of `ms` given to prompt `served`: under a rule that divides, every ready prompt's credit grows by its part of it and the served one's falls by all of it.
inline void charge_share(const Share& s, const std::vector<ReadyPrompt>& p, std::vector<double>& credit, size_t served, double ms) {
    const std::vector<double> parts = share_parts(s, p);
    if (parts.empty()) return;
    credit.resize(p.size(), 0.0);
    for (size_t i = 0; i < p.size(); ++i) credit[i] += parts[i] * ms;
    credit[served] -= ms;
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
