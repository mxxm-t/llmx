#pragma once
// The scheduler's policy core (docs/SERVER.md, the round): what the pools' blocks hold, how an uncapped request's reservation grows, who gives up blocks for whom, which stages a round records and which passes it retires, how many decode entries a pass takes, and where a pass's logits rows go, as free functions over plain data.
// The scheduler calls them with its requests and passes, and the server-passes CTest with a simulated executor's.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
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
};

// A running request as make_room sees it: its first admission, whether it may be paused (an uncapped one), the blocks it has reserved, those its history would keep as a donor once paused, per pool, and whether a pass in flight holds it.
struct Holder {
    uint64_t admission;
    bool uncapped;
    std::vector<size_t> need, kept;
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

// Who gives up blocks for whom, the one owner of that rule (docs/SERVER.md, room by first admission): for `need` more blocks per pool, donors go first, oldest first, the one the request forks (`keep`) last or, with `keep_first`, first.
// Only growth (`preempt`) then pauses uncapped requests admitted after `after`, latest first, each giving its headroom and, if still short, its donor; nothing is taken unless the whole is enough.
inline Taken make_room(const std::vector<size_t>& pool, const std::vector<size_t>& reserved, const std::vector<std::vector<size_t>>& donors,
                       size_t keep, bool keep_first, const std::vector<Holder>& active, uint64_t after, bool preempt,
                       const std::vector<size_t>& need) {
    std::vector<size_t> free(pool.size());
    for (size_t s = 0; s < pool.size(); ++s) free[s] = pool[s] - reserved[s];
    const auto fits = [&] {
        for (size_t s = 0; s < need.size(); ++s)
            if (need[s] > free[s]) return false;
        return true;
    };
    const auto gain = [&](const std::vector<size_t>& b, const std::vector<size_t>& less) {
        for (size_t s = 0; s < free.size(); ++s) {
            const size_t l = s < less.size() ? less[s] : 0, v = s < b.size() ? b[s] : 0;
            free[s] += v > l ? v - l : 0;
        }
    };
    Taken t;
    t.enough = fits();
    std::vector<size_t> order;
    if (keep < donors.size() && keep_first) order.push_back(keep);
    for (size_t d = 0; d < donors.size(); ++d)
        if (d != keep) order.push_back(d);
    if (keep < donors.size() && !keep_first) order.push_back(keep);
    for (size_t i = 0; i < order.size() && !t.enough; ++i) {
        gain(donors[order[i]], {});
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
            gain(h.need, h.kept);
            t.paused.push_back({later[i], false});
            t.wait = t.wait || h.flying;
            if ((t.enough = fits()) || std::all_of(h.kept.begin(), h.kept.end(), [](size_t b) { return b == 0; })) continue;
            gain(h.kept, {});
            t.paused.back().second = true;
            t.enough = fits();
        }
    }
    if (!t.enough) return Taken{};
    return t;
}

// A pass slot as the round sees it: whether a pass is in flight in it, that pass's place in formation order, and the stages recorded.
struct Flight {
    bool live = false;
    uint64_t formed = 0;
    size_t ran = 0;
};

// What a round records and retires, by slot, before it makes room and forms passes.
struct Steps {
    std::vector<std::pair<size_t, size_t>> advance;   // (slot, stage), in the order to record them
    std::vector<size_t> retire;                       // oldest first
};

// From the last stage down to stage 1, each stage records the oldest pass waiting for it, and every pass whose last stage an earlier round recorded retires, oldest first.
// So each device runs its passes in formation order, a pass advances a stage at most a round, and the later stages have their work before the host samples.
inline Steps round_steps(const std::vector<Flight>& slots, size_t stages) {
    Steps st;
    const size_t none = slots.size();
    for (size_t s = stages; s-- > 1;) {
        size_t pick = none;
        for (size_t k = 0; k < slots.size(); ++k)
            if (slots[k].live && slots[k].ran == s && (pick == none || slots[k].formed < slots[pick].formed)) pick = k;
        if (pick != none) st.advance.push_back({pick, s});
    }
    for (size_t k = 0; k < slots.size(); ++k)
        if (slots[k].live && slots[k].ran == stages) st.retire.push_back(k);
    std::sort(st.retire.begin(), st.retire.end(), [&](size_t a, size_t b) { return slots[a].formed < slots[b].formed; });
    return st;
}

// The most decode entries a new pass takes when `decoders` requests decode, those in flight included, over `passes` passes in flight on `stages` stages: an even share, so every pass carries about as many rows however the requests arrived.
// Only passes that fill the stages share, since then a pass retires every round and a request held back joins the next; with fewer, a request held back could wait for most of a pass, and every ready one goes.
inline size_t decode_share(size_t decoders, size_t passes, size_t stages) { return passes < stages ? decoders : (decoders + passes - 1) / passes; }

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

} // namespace server
