// make_room, the one owner of who gives up blocks for whom (docs/SERVER.md, room by first admission), by hand and through random runs of a simulation of the scheduler's rules over two pools of different block sizes.
// After every operation no pool is over-reserved, the ledger adds up, the oldest request is refused room only when capped requests hold it, no pass is empty while requests are active, and once submissions stop every request ends.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "server/scheduler.hpp"

namespace {

size_t checks = 0;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}

struct Sim {
    struct Req {
        uint64_t id = 0, admission = 0, donor = 0;   // donor: the one its last pause left, which it takes back if it is still there
        bool uncapped = false, stalled = false;
        size_t prompt = 0, max_tokens = 0, gen = 0, len = 0;   // len: what its cache holds
        std::vector<size_t> need;
        size_t history() const { return prompt + gen; }
        bool decoding() const { return gen && len + 1 == history(); }
    };
    struct Donor {
        uint64_t id = 0;
        size_t len = 0;
        std::vector<size_t> blocks;
    };
    std::vector<size_t> pool{64, 32}, block{64, 128}, reserved{0, 0};
    std::vector<Donor> donors;               // oldest first
    std::vector<Req> queue, active, paused;  // active and paused in order of first admission
    size_t max_seqs, ubatch, grow = 256, next_id = 1, admissions = 0, ended = 0, stalls = 0, pauses = 0, taken_back = 0;
    uint64_t donor_ids = 0;
    std::mt19937 rng;
    std::string at;   // what the simulation was doing, for a failure's message

    Sim(uint32_t seed, size_t seqs, size_t ub) : max_seqs(seqs), ubatch(ub), rng(seed) {}

    size_t largest_block() const { return *std::max_element(block.begin(), block.end()); }
    size_t limit() const { return std::min(pool[0] * block[0], pool[1] * block[1]); }
    std::vector<size_t> blocks_for(size_t tokens) const {
        std::vector<size_t> b(pool.size());
        for (size_t s = 0; s < b.size(); ++s) b[s] = std::min((tokens + block[s] - 1) / block[s], pool[s]);
        return b;
    }
    // A paused request's history stays as a donor however short, so what it keeps is what it holds.
    std::vector<server::Holder> holders() const {
        std::vector<server::Holder> h;
        for (const Req& r : active)
            h.push_back({r.admission, r.uncapped, r.need, r.len ? blocks_for(r.len) : std::vector<size_t>(pool.size(), 0)});
        return h;
    }
    std::vector<std::vector<size_t>> donor_blocks() const {
        std::vector<std::vector<size_t>> b;
        for (const Donor& d : donors) b.push_back(d.blocks);
        return b;
    }
    void drop(size_t i) {
        sub(reserved, donors[i].blocks);
        donors.erase(donors.begin() + (std::ptrdiff_t)i);
    }
    static void add(std::vector<size_t>& a, const std::vector<size_t>& b) { for (size_t s = 0; s < a.size(); ++s) a[s] += b[s]; }
    static void sub(std::vector<size_t>& a, const std::vector<size_t>& b) { for (size_t s = 0; s < a.size(); ++s) a[s] -= b[s]; }

    // After growth, every request that is not sitting the pass out has reserved room for what its pass writes.
    void check(bool grown = false) {
        std::vector<size_t> sum(pool.size(), 0);
        for (const Req& r : active) {
            add(sum, r.need);
            const std::vector<size_t> held = blocks_for(r.len + (grown && r.decoding() && !r.stalled ? 1 : 0));
            for (size_t s = 0; s < pool.size(); ++s) require(held[s] <= r.need[s], at + ": a request holds more than it reserved");
        }
        for (const auto& d : donors) add(sum, d.blocks);
        for (size_t s = 0; s < pool.size(); ++s) {
            require(reserved[s] <= pool[s], at + ": pool " + std::to_string(s) + " over-reserved");
            require(reserved[s] == sum[s], at + ": the ledger does not add up in pool " + std::to_string(s));
        }
    }

    void submit() {
        Req r;
        r.id = next_id++;
        r.uncapped = rng() % 3 != 0;
        r.prompt = 1 + rng() % 600;
        if (r.prompt >= limit()) r.prompt = limit() - 1;
        r.max_tokens = r.uncapped ? limit() - r.prompt : 1 + rng() % std::min<size_t>(limit() - r.prompt, 900);
        queue.push_back(r);
    }

    // A request leaves the active set: its history a donor when it holds at least `least` tokens (a full block for a finished request, any for a paused one's), its blocks returned otherwise; the donor's id, 0 when none.
    uint64_t park(size_t i, size_t least) {
        Req r = active[i];
        active.erase(active.begin() + (std::ptrdiff_t)i);
        sub(reserved, r.need);
        if (!r.len || r.len < least) return 0;
        if (donors.size() >= max_seqs) drop(0);
        donors.push_back({++donor_ids, r.len, blocks_for(r.len)});
        add(reserved, donors.back().blocks);
        return donor_ids;
    }

    bool enter(Req& r) {
        std::vector<size_t> need = blocks_for(r.history() + (r.uncapped ? grow : r.max_tokens - r.gen));
        // A resumed request whose donor is still there takes it back, first; otherwise now and then the request forks a donor, which make_room keeps for last, or takes first when told to.
        size_t own = donors.size();
        for (size_t d = 0; r.donor && d < donors.size(); ++d)
            if (donors[d].id == r.donor) own = d;
        const bool take = own < donors.size();
        const size_t keep = take ? own : !donors.empty() && rng() % 2 ? rng() % donors.size() : (size_t)-1;
        const bool keep_first = take || (keep < donors.size() && rng() % 2);
        const server::Taken t = server::make_room(pool, reserved, donor_blocks(), keep, keep_first, {}, 0, false, need);
        if (!t.enough) {
            // Refused, so even every donor would not do.
            std::vector<size_t> all = reserved;
            for (const auto& d : donors) sub(all, d.blocks);
            bool short_somewhere = false;
            for (size_t s = 0; s < pool.size(); ++s) short_somewhere = short_somewhere || all[s] + need[s] > pool[s];
            require(short_somewhere, at + ": admission refused though the donors held enough");
            return false;
        }
        require(t.paused.empty(), at + ": admission paused a request");
        std::vector<size_t> gone = t.donors;
        if (take && std::find(gone.begin(), gone.end(), own) == gone.end()) gone.push_back(own);
        std::sort(gone.begin(), gone.end(), std::greater<size_t>());
        // A donor taken back gives the request what it held; a resume otherwise recomputes all it holds, a first admission all its prompt, which is what the ledger sees.
        r.len = take ? donors[own].len : 0;
        for (size_t i : gone) drop(i);
        add(reserved, need);
        r.need = need;
        r.donor = 0;
        r.stalled = false;
        taken_back += take;
        for (size_t s = 0; s < pool.size(); ++s)
            require(blocks_for(r.len)[s] <= need[s], at + ": a request took back more than it reserved");
        if (!r.admission) r.admission = ++admissions;
        active.insert(std::upper_bound(active.begin(), active.end(), r, [](const Req& a, const Req& b) { return a.admission < b.admission; }), r);
        return true;
    }

    // Whether it left a donor.
    bool pause(size_t i) {
        Req r = active[i];
        r.donor = park(i, 1);
        paused.insert(std::upper_bound(paused.begin(), paused.end(), r, [](const Req& a, const Req& b) { return a.admission < b.admission; }), r);
        ++pauses;
        return r.donor != 0;
    }

    void grow_all() {
        for (size_t i = 0; i < active.size(); ++i) {
            Req& r = active[i];
            r.stalled = false;
            const std::vector<size_t> next = blocks_for(r.len + 1);
            bool beyond = false;
            for (size_t s = 0; s < pool.size(); ++s) beyond = beyond || next[s] > r.need[s];
            if (!r.uncapped || !r.decoding() || !beyond) continue;
            std::vector<size_t> step = blocks_for(r.len + 1 + grow);
            for (size_t s = 0; s < step.size(); ++s) step[s] = step[s] > r.need[s] ? step[s] - r.need[s] : 0;
            const server::Taken t = server::make_room(pool, reserved, donor_blocks(), (size_t)-1, false, holders(), r.admission, true, step);
            if (!t.enough) {
                // Refused: even every donor and every uncapped request admitted after it would not do, so what it lacks is held by capped requests or by requests admitted before it.
                std::vector<size_t> could = reserved;
                for (const auto& d : donors) sub(could, d.blocks);
                for (size_t j = i + 1; j < active.size(); ++j)
                    if (active[j].uncapped) sub(could, active[j].need);
                bool short_somewhere = false;
                for (size_t s = 0; s < pool.size(); ++s) short_somewhere = short_somewhere || could[s] + step[s] > pool[s];
                require(short_somewhere, at + ": a request was refused room that donors or requests admitted after it held");
                if (i == 0) {
                    bool capped = false;
                    for (size_t j = 1; j < active.size(); ++j) capped = capped || !active[j].uncapped;
                    require(capped, at + ": the oldest request was refused though only uncapped requests after it hold room");
                }
                r.stalled = true;
                ++stalls;
                continue;
            }
            std::vector<size_t> gone = t.donors;
            std::sort(gone.begin(), gone.end(), std::greater<size_t>());
            for (size_t d : gone) drop(d);
            std::vector<uint64_t> victims;
            for (const auto& p : t.paused) {
                require(active[p.first].uncapped && active[p.first].admission > r.admission, at + ": a request paused one it may not");
                victims.push_back(active[p.first].id);
            }
            for (size_t v = 0; v < victims.size(); ++v) {
                const size_t k = (size_t)(std::find_if(active.begin(), active.end(), [&](const Req& a) { return a.id == victims[v]; }) - active.begin());
                if (pause(k) && t.paused[v].second) drop(donors.size() - 1);
            }
            add(reserved, step);
            add(r.need, step);
        }
    }

    // One iteration of the loop: admission, growth, the pass.
    void iterate() {
        const bool stalled = std::any_of(active.begin(), active.end(), [](const Req& r) { return r.stalled; });
        at = "admission";
        while (!stalled && !paused.empty() && active.size() < max_seqs && enter(paused.front())) paused.erase(paused.begin());
        while (!stalled && paused.empty() && !queue.empty() && active.size() < max_seqs && enter(queue.front())) queue.erase(queue.begin());
        check();
        if (active.empty()) return;
        at = "growth";
        grow_all();
        check(true);
        at = "the pass";
        size_t budget = ubatch, rows = 0;
        for (Req& r : active) {
            if (r.stalled) continue;
            if (r.decoding()) { ++r.len; ++r.gen; ++rows; continue; }
            if (!budget) continue;
            const size_t n = std::min(budget, r.history() - r.len);
            r.len += n;
            budget -= n;
            rows += n;
            if (r.len == r.history()) ++r.gen;   // the pass that ends its history samples its next token
        }
        require(rows > 0, "an iteration had active requests and an empty pass");
        for (size_t i = 0; i < active.size();)
            if (active[i].gen >= active[i].max_tokens) { park(i, largest_block()); ++ended; }
            else ++i;
        check();
    }

    void cancel() {
        const size_t n = active.size() + paused.size() + queue.size();
        if (!n) return;
        size_t k = rng() % n;
        at = "a cancellation";
        if (k < active.size()) { park(k, largest_block()); ++ended; }
        else if ((k -= active.size()) < paused.size()) { paused.erase(paused.begin() + (std::ptrdiff_t)k); ++ended; }
        else { queue.erase(queue.begin() + (std::ptrdiff_t)(k - paused.size())); ++ended; }
        check();
    }
};

void random_runs() {
    size_t pauses = 0, stalls = 0, ended = 0, taken_back = 0;
    for (uint32_t seed = 1; seed <= 16; ++seed) {
        Sim sim(seed, 2 + seed % 7, seed % 3 ? 512 : 64);
        if (seed % 4 == 0) sim.grow = 64;
        for (int i = 0; i < 3000; ++i) {
            const uint32_t roll = sim.rng() % 100;
            if (roll < 4) sim.submit();
            else if (roll < 5) sim.cancel();
            sim.iterate();
        }
        // Submissions stop, and every request ends.
        const size_t submitted = sim.next_id - 1;
        for (int i = 0; i < 400000 && sim.ended < submitted; ++i) sim.iterate();
        require(sim.ended == submitted, "seed " + std::to_string(seed) + ": " + std::to_string(submitted - sim.ended) + " requests never ended");
        require(sim.reserved == std::vector<size_t>(sim.reserved.size(), 0) || !sim.donors.empty(), "a drained pool holds blocks no donor holds");
        pauses += sim.pauses;
        stalls += sim.stalls;
        taken_back += sim.taken_back;
        ended += sim.ended;
    }
    require(pauses > 0 && stalls > 0 && taken_back > 0, "the runs paused " + std::to_string(pauses) + " times, stalled " + std::to_string(stalls) +
            " and took back " + std::to_string(taken_back) + " donors; all three must happen");
    std::printf("server-room: 16 runs, %zu requests, %zu pauses, %zu donors taken back, %zu stalls\n", ended, pauses, taken_back, stalls);
}

// A few cases by hand: a request that fits takes nothing, one that cannot fit takes nothing, the donor it forks goes last or first, and growth pauses the latest admitted uncapped request after it, its donor going only when its headroom is short.
void by_hand() {
    const std::vector<size_t> pool{10};
    const std::vector<std::vector<size_t>> donors{{2}, {3}};
    auto t = server::make_room(pool, {4}, donors, (size_t)-1, false, {}, 0, false, {1});
    require(t.enough && t.donors.empty(), "a request that fits took something");
    t = server::make_room(pool, {9}, donors, (size_t)-1, false, {}, 0, false, {7});
    require(!t.enough && t.donors.empty(), "a request that cannot fit took donors");
    t = server::make_room(pool, {9}, donors, 0, false, {}, 0, false, {5});
    require(t.enough && t.donors == std::vector<size_t>({1, 0}), "the donor a request forks did not go last");
    t = server::make_room(pool, {9}, donors, 1, true, {}, 0, false, {3});
    require(t.enough && t.donors == std::vector<size_t>({1}), "the donor a request shares whole did not go first");
    const std::vector<server::Holder> active{{1, true, {2}, {1}}, {2, false, {2}, {1}}, {3, true, {3}, {2}}, {4, true, {2}, {0}}};
    t = server::make_room(pool, {10}, {}, (size_t)-1, false, active, 1, true, {3});
    require(t.enough && t.paused == std::vector<std::pair<size_t, bool>>({{3, false}, {2, false}}), "growth did not pause the latest uncapped requests after it, headroom first");
    t = server::make_room(pool, {10}, {}, (size_t)-1, false, active, 1, true, {5});
    require(t.enough && t.paused == std::vector<std::pair<size_t, bool>>({{3, false}, {2, true}}), "a paused request's donor did not go when its headroom was short");
    t = server::make_room(pool, {10}, {}, (size_t)-1, false, active, 1, false, {1});
    require(!t.enough && t.paused.empty(), "admission paused a request");
    t = server::make_room(pool, {10}, {}, (size_t)-1, false, active, 3, true, {3});
    require(!t.enough && t.paused.empty(), "a request paused one admitted before it or a capped one");
}

} // namespace

int main() {
    try {
        by_hand();
        random_runs();
        std::cout << "server-room: " << checks << " checks pass\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "server-room: " << e.what() << '\n';
        return 1;
    }
}
