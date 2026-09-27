#pragma once
// The scheduler's policy core (docs/SERVER.md): who gives up blocks for whom, as a free function over plain data that the scheduler calls with its requests.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace server {

// A running request as make_room sees it: its first admission, whether it may be paused (an uncapped one), the blocks it has reserved, and those its history would keep as a donor once paused, per pool.
struct Holder {
    uint64_t admission;
    bool uncapped;
    std::vector<size_t> need, kept;
};

// What make_room takes, in the order it takes it: donors by index, then running requests to pause by index, each with whether the donor its history becomes goes too.
struct Taken {
    bool enough = false;
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
            if ((t.enough = fits()) || std::all_of(h.kept.begin(), h.kept.end(), [](size_t b) { return b == 0; })) continue;
            gain(h.kept, {});
            t.paused.back().second = true;
            t.enough = fits();
        }
    }
    if (!t.enough) return Taken{};
    return t;
}

} // namespace server
