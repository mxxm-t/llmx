// The scheduler's policy core (server/policy.hpp): make_room, the round's stages and the logits rows by hand, then under a simulated executor in random schedules over 1 to 4 stages, 1 to 2S pass slots and two pools of different block sizes, with random stage times, arrivals, growth, pauses, cancellations, failures and stops.
// After every event a request is in at most one pass and no pass is empty, each device runs its passes in formation order, a slot and a run of logits rows belong to one pass until it ends, no pool is over-reserved and nothing in flight is paused, parked or ended, and the oldest request is refused room only when capped requests hold it.
// A growth plan pauses one request at most, and the oldest request's plan that waits on it in flight takes it in the round its pass retires; within one lap every decoder that is not stalled gets a token and a cancellation ends; admission is first-come and no free slot idles while work is ready; after a stop the ledger and every pool are at zero.
// `llmx-server-passes-test N` runs N schedules, 2000 by default.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "server/policy.hpp"

namespace {

size_t checks = 0;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}

constexpr size_t npos = (size_t)-1;

// The scheduler's rules over the policy core, driven by a simulated executor: stage s runs on device s, the host waits for a stage's handoff before it records the next one and for a pass's last stage before it samples, and each stage takes a random time on its device.
struct Sim {
    struct Req {
        uint64_t id = 0, admission = 0, donor = 0;   // donor: the one its last pause left, which it takes back if it is still there
        bool uncapped = false, stalled = false, cancel = false;
        size_t prompt = 0, max_tokens = 0, gen = 0, len = 0;   // len: what its cache holds, a pass in flight's rows not counted
        size_t slot = npos;                                   // the slot of the pass it is in flight in
        uint64_t since = 0, cancelled = 0;                    // the round it last got a token or could not, and the round its cancel was first seen in
        std::vector<size_t> need;
        size_t history() const { return prompt + gen; }
        bool decoding() const { return gen && len + 1 == history(); }
    };
    struct Donor {
        uint64_t id = 0;
        size_t len = 0;
        std::vector<size_t> blocks;
    };
    struct Pass {
        bool live = false;
        uint64_t formed = 0;
        size_t ran = 0, base = 0, want = 0;
        std::vector<uint64_t> ids;   // by entry
        std::vector<size_t> rows;
        std::vector<char> wants;
        double ready = 0;            // when its last recorded stage ends on its device
    };
    size_t S, P, lap, max_seqs = 4, ubatch = 16, grow = 16, fail_per_mille = 2;
    std::vector<size_t> pool{64, 32}, block{64, 128}, reserved{0, 0};
    std::vector<Donor> donors;               // oldest first
    std::vector<Req> queue, active, paused;  // active and paused in order of first admission
    std::vector<Pass> slots;
    server::LogitRows rows;
    std::vector<double> device_free;
    std::vector<uint64_t> device_last;       // per device, the formation order of the last pass it ran
    double clock = 0;
    uint64_t round_no = 0, formed = 0, next_id = 1, admissions = 0, donor_ids = 0, first_admitted = 0, blocked_at = 0;
    uint64_t waiter = 0, waited = 0, waiting_from = 0;   // the oldest request while its plan waits on a request in flight, that request, and the round it first waited
    size_t ended = 0, stalls = 0, pauses = 0, taken_back = 0, waits = 0, resolved = 0, failures = 0, flying_cancels = 0, unrecorded_cancels = 0;
    std::mt19937 rng;
    std::string at;   // what the simulation was doing, for a failure's message

    Sim(uint32_t seed, size_t stages, size_t passes) : S(stages), P(passes), lap(stages + passes - 1), rng(seed) {
        slots.resize(P);
        device_free.assign(S, 0.0);
        device_last.assign(S, 0);
        rows.size = std::min<size_t>(P, 2) * max_seqs;
    }
    // A small random configuration, so schedules pause, stall and hold often.
    void randomize() {
        max_seqs = 2 + rng() % 7;
        ubatch = rng() % 2 ? 16 : 4;
        const size_t b = rng() % 2 ? 4 : 8;
        block = {b, 2 * b};
        pool = {12 + rng() % 36, 0};
        pool[1] = pool[0] / 2 + rng() % 4;
        grow = b * (1 + rng() % 4);
        rows.size = std::min<size_t>(P, 2) * max_seqs;
    }

    size_t largest_block() const { return *std::max_element(block.begin(), block.end()); }
    size_t limit() const { return std::min(pool[0] * block[0], pool[1] * block[1]); }
    std::vector<size_t> blocks_for(size_t tokens) const {
        std::vector<size_t> b(pool.size());
        for (size_t s = 0; s < b.size(); ++s) b[s] = std::min((tokens + block[s] - 1) / block[s], pool[s]);
        return b;
    }
    size_t find(uint64_t id) const {
        for (size_t i = 0; i < active.size(); ++i)
            if (active[i].id == id) return i;
        return npos;
    }
    size_t in_flight() const {
        size_t n = 0;
        for (const Pass& p : slots) n += p.live;
        return n;
    }
    size_t free_slot() const {
        for (size_t k = 0; k < slots.size(); ++k)
            if (!slots[k].live) return k;
        return npos;
    }
    // Rows a pass in flight adds to a request's history.
    size_t flying_rows(const Req& r) const {
        if (r.slot == npos) return 0;
        const Pass& p = slots[r.slot];
        for (size_t e = 0; e < p.ids.size(); ++e)
            if (p.ids[e] == r.id) return p.rows[e];
        return 0;
    }
    bool ready(const Req& r) const { return r.slot == npos && !r.stalled; }
    // A paused request's history stays as a donor however short, so what it keeps is what it holds.
    std::vector<server::Holder> holders() const {
        std::vector<server::Holder> h;
        for (const Req& r : active)
            h.push_back({r.admission, r.uncapped, r.need, r.len ? blocks_for(r.len) : std::vector<size_t>(pool.size(), 0), r.slot != npos});
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

    // The invariants that hold after every event; after growth, every ready decoding request has reserved room for its next token.
    void check(bool grown = false) {
        std::vector<size_t> sum(pool.size(), 0);
        for (const Req& r : active) {
            add(sum, r.need);
            const bool next = grown && r.decoding() && ready(r);
            const std::vector<size_t> held = blocks_for(r.len + flying_rows(r) + (next ? 1 : 0));
            for (size_t s = 0; s < pool.size(); ++s) require(held[s] <= r.need[s], at + ": a request holds more than it reserved");
            require(r.slot == npos || (r.slot < slots.size() && slots[r.slot].live), at + ": a request in flight in no pass");
        }
        for (const auto& d : donors) add(sum, d.blocks);
        for (size_t s = 0; s < pool.size(); ++s) {
            require(reserved[s] <= pool[s], at + ": pool " + std::to_string(s) + " over-reserved");
            require(reserved[s] == sum[s], at + ": the ledger does not add up in pool " + std::to_string(s));
        }
        for (const auto* waiting : {&queue, &paused})
            for (const Req& r : *waiting) require(r.slot == npos, at + ": a waiting request in flight");
        std::vector<std::pair<size_t, size_t>> runs;
        for (size_t k = 0; k < slots.size(); ++k) {
            const Pass& p = slots[k];
            if (!p.live) continue;
            require(!p.ids.empty(), at + ": an empty pass in flight");
            for (uint64_t id : p.ids) {
                const size_t i = find(id);
                require(i != npos && active[i].slot == k, at + ": a pass holds a request that is not in flight in it");
            }
            for (size_t j = 0; j < slots.size(); ++j)
                if (j != k && slots[j].live) require(slots[j].formed != p.formed, at + ": two passes formed as one");
            if (p.want) {
                require(p.base + p.want <= rows.size, at + ": a pass's logits rows past the reservation");
                bool held = false;
                for (const auto& r : rows.runs) held = held || (r.base == p.base && r.n == p.want && !r.back);
                require(held, at + ": a pass in flight whose logits rows came back");
                runs.push_back({p.base, p.want});
            }
        }
        std::sort(runs.begin(), runs.end());
        for (size_t i = 1; i < runs.size(); ++i) require(runs[i - 1].first + runs[i - 1].second <= runs[i].first, at + ": two passes share logits rows");
    }

    void submit() {
        Req r;
        r.id = next_id++;
        r.uncapped = rng() % 3 != 0;
        r.prompt = 1 + rng() % std::min<size_t>(limit() - 1, 4 * largest_block());
        r.max_tokens = r.uncapped ? limit() - r.prompt : 1 + rng() % std::min<size_t>(limit() - r.prompt, 48);
        queue.push_back(r);
    }

    // A request leaves the active set: its history a donor when it holds at least `least` tokens (a full block for a finished request, any for a paused one's), its blocks returned otherwise; the donor's id, 0 when none.
    uint64_t park(size_t i, size_t least) {
        Req r = active[i];
        require(r.slot == npos, at + ": a request in flight parked");
        active.erase(active.begin() + (std::ptrdiff_t)i);
        sub(reserved, r.need);
        if (!r.len || r.len < least) return 0;
        if (donors.size() >= max_seqs) drop(0);
        donors.push_back({++donor_ids, r.len, blocks_for(r.len)});
        add(reserved, donors.back().blocks);
        return donor_ids;
    }
    void finish(size_t i) {
        park(i, largest_block());
        ++ended;
    }

    bool enter(Req& r) {
        std::vector<size_t> need = blocks_for(r.history() + (r.uncapped ? grow : r.max_tokens - r.gen));
        // A resumed request whose donor is still there takes it back, first; otherwise now and then the request forks a donor, which make_room keeps for last, or takes first when told to.
        size_t own = donors.size();
        for (size_t d = 0; r.donor && d < donors.size(); ++d)
            if (donors[d].id == r.donor) own = d;
        const bool take = own < donors.size();
        const size_t keep = take ? own : !donors.empty() && rng() % 2 ? rng() % donors.size() : npos;
        const bool keep_first = take || (keep < donors.size() && rng() % 2);
        const server::Taken t = server::make_room(pool, reserved, donor_blocks(), keep, keep_first, holders(), 0, false, need);
        if (!t.enough) {
            // Refused, so even every donor would not do.
            std::vector<size_t> all = reserved;
            for (const auto& d : donors) sub(all, d.blocks);
            bool short_somewhere = false;
            for (size_t s = 0; s < pool.size(); ++s) short_somewhere = short_somewhere || all[s] + need[s] > pool[s];
            require(short_somewhere, at + ": admission refused though the donors held enough");
            return false;
        }
        require(t.paused.empty() && !t.wait, at + ": admission paused a request");
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
        r.since = round_no;
        taken_back += take;
        for (size_t s = 0; s < pool.size(); ++s)
            require(blocks_for(r.len)[s] <= need[s], at + ": a request took back more than it reserved");
        if (!r.admission) {
            require(r.id > first_admitted, at + ": a request admitted before one submitted earlier");
            first_admitted = r.id;
            r.admission = ++admissions;
        }
        active.insert(std::upper_bound(active.begin(), active.end(), r, [](const Req& a, const Req& b) { return a.admission < b.admission; }), r);
        return true;
    }

    // Whether it left a donor.
    bool pause(size_t i) {
        Req r = active[i];
        r.donor = park(i, 1);
        r.stalled = false;
        paused.insert(std::upper_bound(paused.begin(), paused.end(), r, [](const Req& a, const Req& b) { return a.admission < b.admission; }), r);
        ++pauses;
        return r.donor != 0;
    }

    // Cancelled requests end where they wait, and active ones once no pass holds them.
    void sweep_waiting() {
        for (size_t i = 0; i < queue.size();)
            if (queue[i].cancel) { queue.erase(queue.begin() + (std::ptrdiff_t)i); ++ended; }
            else ++i;
        for (size_t i = 0; i < paused.size();) {
            if (!paused[i].cancel) { ++i; continue; }
            // A paused request's own donor goes with it when it holds less than a block, which no fork can share.
            for (size_t d = 0; paused[i].donor && d < donors.size(); ++d)
                if (donors[d].id == paused[i].donor && donors[d].len < largest_block()) { drop(d); break; }
            paused.erase(paused.begin() + (std::ptrdiff_t)i);
            ++ended;
        }
    }
    void sweep_active() {
        for (size_t i = 0; i < active.size();)
            if (active[i].cancel && active[i].slot == npos) finish(i);
            else ++i;
    }

    // Stage s of the pass in slot k; now and then it fails, and the pass is abandoned; whether it was recorded.
    bool record(size_t k, size_t s) {
        Pass& p = slots[k];
        require(p.live && p.ran == s, at + ": a stage recorded out of order");
        require(device_last[s] < p.formed, at + ": device " + std::to_string(s) + " ran a pass before one formed earlier");
        device_last[s] = p.formed;
        if (rng() % 1000 < fail_per_mille) {
            abort(k);
            return false;
        }
        if (s) clock = std::max(clock, p.ready);
        size_t n = 0;
        for (size_t r : p.rows) n += r;
        const double start = std::max(clock, device_free[s]);
        p.ready = device_free[s] = start + (0.5 + (double)(rng() % 100) / 100.0) * (1.0 + 0.05 * (double)n);
        clock += 0.05;
        ++p.ran;
        return true;
    }

    // A failed pass: every device drained, its requests end with the error and give their blocks back without a donor, and the other passes go on.
    void abort(size_t k) {
        Pass& p = slots[k];
        for (double t : device_free) clock = std::max(clock, t);
        for (uint64_t id : p.ids) {
            const size_t i = find(id);
            require(i != npos && active[i].slot == k, at + ": a failed pass's request not in flight in it");
            sub(reserved, active[i].need);
            active.erase(active.begin() + (std::ptrdiff_t)i);
            ++ended;
        }
        server::give_rows(rows, p.base, p.want);
        p = Pass{};
        ++failures;
    }

    // The pass in slot k after its last stage: its rows committed, each wanting row sampled unless its request was cancelled, and the requests that end parked.
    void retire(size_t k) {
        Pass& p = slots[k];
        require(p.live && p.ran == S, at + ": a pass retired before its last stage");
        clock = std::max(clock, p.ready);
        std::vector<uint64_t> ids = p.ids;
        for (size_t e = 0; e < p.ids.size(); ++e) {
            Req& r = active[find(p.ids[e])];
            r.len += p.rows[e];
            r.slot = npos;
            if (p.wants[e] && !r.cancel) {
                ++r.gen;
                r.since = round_no;
            }
        }
        server::give_rows(rows, p.base, p.want);
        p = Pass{};
        for (uint64_t id : ids) {
            const size_t i = find(id);
            if (active[i].cancel || active[i].gen >= active[i].max_tokens) finish(i);
        }
    }

    // Growth, the earliest admitted first: a step make_room gives is taken, and one it refuses, or whose plan would pause a request in flight, is sat out.
    // A step never passes one uncapped request's reservation, so a plan pauses one request at most, and the round that request's pass retires in makes room with it out of flight.
    void grow_all() {
        for (size_t i = 0; i < active.size(); ++i) {
            Req& r = active[i];
            if (r.slot != npos) continue;
            r.stalled = false;
            const std::vector<size_t> next = blocks_for(r.len + 1);
            bool beyond = false;
            for (size_t s = 0; s < pool.size(); ++s) beyond = beyond || next[s] > r.need[s];
            if (!r.uncapped || !r.decoding() || !beyond) continue;
            std::vector<size_t> step = blocks_for(r.len + 1 + grow);
            for (size_t s = 0; s < step.size(); ++s) step[s] = step[s] > r.need[s] ? step[s] - r.need[s] : 0;
            const server::Taken t = server::make_room(pool, reserved, donor_blocks(), npos, false, holders(), r.admission, true, step);
            require(t.paused.size() <= 1, at + ": a growth plan paused " + std::to_string(t.paused.size()) + " requests, though one uncapped request's reservation holds a step");
            if (i == 0 && waiter == r.id) {
                if (find(waited) == npos || active[find(waited)].slot == npos) {
                    require(t.enough && !t.wait, at + ": the oldest request's plan still waits once the request it waited on left flight");
                    ++resolved;
                }
                require(!t.wait || round_no - waiting_from < lap, at + ": the oldest request's plan waited on requests in flight for a lap");
            }
            if (i == 0 && !t.wait) waiter = 0;
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
            for (const auto& p : t.paused) require(active[p.first].uncapped && active[p.first].admission > r.admission, at + ": a request paused one it may not");
            if (t.wait) {
                require(active[t.paused[0].first].slot != npos, at + ": a plan waited on no request in flight");
                if (i == 0) {
                    if (waiter != r.id) waiting_from = round_no;
                    waiter = r.id;
                    waited = active[t.paused[0].first].id;
                }
                r.stalled = true;
                ++stalls;
                ++waits;
                continue;
            }
            std::vector<size_t> gone = t.donors;
            std::sort(gone.begin(), gone.end(), std::greater<size_t>());
            for (size_t d : gone) drop(d);
            std::vector<uint64_t> victims;
            for (const auto& p : t.paused) victims.push_back(active[p.first].id);
            for (size_t v = 0; v < victims.size(); ++v)
                if (pause(find(victims[v])) && t.paused[v].second) drop(donors.size() - 1);
            add(reserved, step);
            add(r.need, step);
        }
    }

    // Room, only while a slot is free: growth, then, while nothing is stalled, the paused requests oldest first and then the queue.
    void room() {
        at = "growth";
        grow_all();
        check(true);
        const bool stalled = std::any_of(active.begin(), active.end(), [](const Req& r) { return r.stalled; });
        at = "admission";
        while (!stalled && !paused.empty() && active.size() < max_seqs && enter(paused.front())) paused.erase(paused.begin());
        while (!stalled && paused.empty() && !queue.empty() && active.size() < max_seqs && enter(queue.front())) queue.erase(queue.begin());
        check(true);
    }

    // New passes while a slot is free: every ready decoding request's next token, then prompt rows up to the ubatch, a pass's logits rows taken as it is formed, and its first stage.
    void form() {
        at = "formation";
        for (size_t k = free_slot(); k != npos; k = free_slot()) {
            Pass p;
            size_t budget = ubatch;
            for (const Req& r : active)
                if (r.decoding() && ready(r)) {
                    p.ids.push_back(r.id);
                    p.rows.push_back(1);
                    p.wants.push_back(1);
                }
            for (const Req& r : active) {
                if (r.decoding() || !ready(r) || !budget) continue;
                const size_t n = std::min(budget, r.history() - r.len);
                budget -= n;
                p.ids.push_back(r.id);
                p.rows.push_back(n);
                p.wants.push_back(r.len + n == r.history());
            }
            if (p.ids.empty()) return;
            for (char w : p.wants) p.want += (size_t)w;
            p.base = server::take_rows(rows, p.want);
            if (p.base == rows.size) {
                // Only a run an aborted pass gave back early, which waits for the passes formed before it, keeps a run from fitting.
                require(std::any_of(rows.runs.begin(), rows.runs.end(), [](const server::LogitRows::Run& r) { return r.back; }),
                        at + ": no logits rows for a pass though the passes in flight hold at most half of them");
                blocked_at = round_no;
                return;
            }
            p.live = true;
            p.formed = ++formed;
            slots[k] = p;
            for (uint64_t id : p.ids) active[find(id)].slot = k;
            check();
            // Now and then a request is cancelled while its pass is formed and not yet recorded.
            if (rng() % 40 == 0) {
                active[find(p.ids[rng() % p.ids.size()])].cancel = true;
                ++unrecorded_cancels;
            }
            record(k, 0);
            check();
        }
    }

    // One round: the waiting requests' cancellations, each stage's oldest waiting pass from the last stage down, the passes whose last stage an earlier round recorded, the active requests' cancellations, then room and new passes while a slot is free.
    void round() {
        ++round_no;
        for (auto* all : {&queue, &paused, &active})
            for (Req& r : *all)
                if (r.cancel && !r.cancelled) r.cancelled = round_no;
        at = "the sweep";
        sweep_waiting();
        check();
        std::vector<server::Flight> f;
        for (const Pass& p : slots) f.push_back({p.live, p.formed, p.ran});
        const server::Steps steps = server::round_steps(f, S);
        at = "advance";
        for (const auto& a : steps.advance) {
            require(slots[a.first].live && slots[a.first].ran == a.second, at + ": a stage given a pass that does not wait for it");
            record(a.first, a.second);
            check();
        }
        at = "retirement";
        for (size_t k : steps.retire) {
            retire(k);
            check();
        }
        at = "the active sweep";
        sweep_active();
        check();
        if (free_slot() != npos) {
            room();
            form();
        }
        // With nothing in flight the host waits for a submission.
        if (!in_flight()) clock += 1.0;
        // Laps: a ready decoder waits one lap at most, two after a pass's rows waited for an aborted pass's, and a cancellation ends within one.
        const uint64_t bound = lap + (blocked_at && round_no - blocked_at <= lap ? lap : 0);
        for (Req& r : active) {
            if (!r.decoding() || r.stalled) r.since = round_no;
            require(round_no - r.since <= bound, "round " + std::to_string(round_no) + ": a decoder went " + std::to_string(round_no - r.since) +
                    " rounds without a token, past a lap of " + std::to_string(bound));
        }
        for (const auto* all : {&queue, &paused, &active})
            for (const Req& r : *all)
                require(!r.cancelled || round_no < r.cancelled + lap, "round " + std::to_string(round_no) + ": a cancellation outlived a lap");
    }

    void cancel() {
        const size_t n = active.size() + paused.size() + queue.size();
        if (!n) return;
        size_t k = rng() % n;
        if (k < active.size()) {
            active[k].cancel = true;
            flying_cancels += active[k].slot != npos;
        } else if ((k -= active.size()) < paused.size()) {
            paused[k].cancel = true;
        } else {
            queue[k - paused.size()].cancel = true;
        }
    }

    // Every pass in flight is abandoned, and then every request ends and every donor goes.
    void stop() {
        at = "the stop";
        for (Pass& p : slots)
            if (p.live) {
                for (uint64_t id : p.ids) active[find(id)].slot = npos;
                server::give_rows(rows, p.base, p.want);
                p = Pass{};
            }
        for (const Req& r : active) sub(reserved, r.need);
        ended += active.size() + paused.size() + queue.size();
        active.clear();
        paused.clear();
        queue.clear();
        while (!donors.empty()) drop(0);
        require(reserved == std::vector<size_t>(pool.size(), 0) && rows.runs.empty(), "after a stop the ledger or the logits rows hold something");
    }
};

// A few cases by hand: a request that fits takes nothing, one that cannot fit takes nothing, the donor it forks goes last or first, and growth pauses the latest admitted uncapped request after it, its donor going only when its headroom is short.
// A plan that would pause a request in flight waits and takes nothing, and one that pauses only requests not in flight does not wait.
void rooms_by_hand() {
    const std::vector<size_t> pool{10};
    const std::vector<std::vector<size_t>> donors{{2}, {3}};
    auto t = server::make_room(pool, {4}, donors, npos, false, {}, 0, false, {1});
    require(t.enough && t.donors.empty(), "a request that fits took something");
    t = server::make_room(pool, {9}, donors, npos, false, {}, 0, false, {7});
    require(!t.enough && t.donors.empty(), "a request that cannot fit took donors");
    t = server::make_room(pool, {9}, donors, 0, false, {}, 0, false, {5});
    require(t.enough && t.donors == std::vector<size_t>({1, 0}), "the donor a request forks did not go last");
    t = server::make_room(pool, {9}, donors, 1, true, {}, 0, false, {3});
    require(t.enough && t.donors == std::vector<size_t>({1}), "the donor a request shares whole did not go first");
    std::vector<server::Holder> active{{1, true, {2}, {1}}, {2, false, {2}, {1}}, {3, true, {3}, {2}}, {4, true, {2}, {0}}};
    t = server::make_room(pool, {10}, {}, npos, false, active, 1, true, {3});
    require(t.enough && !t.wait && t.paused == std::vector<std::pair<size_t, bool>>({{3, false}, {2, false}}), "growth did not pause the latest uncapped requests after it, headroom first");
    t = server::make_room(pool, {10}, {}, npos, false, active, 1, true, {5});
    require(t.enough && t.paused == std::vector<std::pair<size_t, bool>>({{3, false}, {2, true}}), "a paused request's donor did not go when its headroom was short");
    t = server::make_room(pool, {10}, {}, npos, false, active, 1, false, {1});
    require(!t.enough && t.paused.empty(), "admission paused a request");
    t = server::make_room(pool, {10}, {}, npos, false, active, 3, true, {3});
    require(!t.enough && t.paused.empty(), "a request paused one admitted before it or a capped one");
    active[2].flying = true;
    t = server::make_room(pool, {10}, {}, npos, false, active, 1, true, {3});
    require(t.enough && t.wait && t.paused == std::vector<std::pair<size_t, bool>>({{3, false}, {2, false}}), "a plan pausing a request in flight did not wait");
    t = server::make_room(pool, {10}, {}, npos, false, active, 1, true, {2});
    require(t.enough && !t.wait && t.paused == std::vector<std::pair<size_t, bool>>({{3, false}}), "a plan pausing only requests not in flight waited");
    active[2].flying = false;
    active[1].flying = true;
    t = server::make_room(pool, {10}, {}, npos, false, active, 1, true, {3});
    require(t.enough && !t.wait, "a capped request in flight held up a plan that does not pause it");
}

// The round by hand: each stage records its oldest waiting pass from the last stage down, one stage a pass, and passes past their last stage retire oldest first.
void rounds_by_hand() {
    auto st = server::round_steps({{true, 5, 1}, {true, 3, 1}, {false, 1, 2}, {true, 4, 2}, {true, 2, 3}, {true, 1, 3}}, 3);
    require(st.advance == std::vector<std::pair<size_t, size_t>>({{3, 2}, {1, 1}}), "a stage did not take its oldest waiting pass, or took a free slot");
    require(st.retire == std::vector<size_t>({5, 4}), "passes past their last stage did not retire oldest first");
    st = server::round_steps({{true, 7, 1}}, 1);
    require(st.advance.empty() && st.retire == std::vector<size_t>({0}), "one stage's pass did not retire the round after it was formed");
    st = server::round_steps({{false, 0, 0}, {false, 0, 0}}, 4);
    require(st.advance.empty() && st.retire.empty(), "a round without passes did something");
}

// The logits rows by hand: runs follow the newest and wrap to row 0, come back oldest first, and a run given back early waits for the ones before it.
void rows_by_hand() {
    server::LogitRows rows;
    rows.size = 8;
    require(server::take_rows(rows, 0) == 0 && rows.runs.empty(), "no rows took a run");
    require(server::take_rows(rows, 3) == 0 && server::take_rows(rows, 3) == 3, "runs did not follow the newest");
    require(server::take_rows(rows, 3) == 8, "a run fit past the end or over a run in flight");
    server::give_rows(rows, 0, 3);
    require(server::take_rows(rows, 4) == 8, "a run wrapped to row 0 over the oldest run");
    require(server::take_rows(rows, 2) == 6 && server::take_rows(rows, 3) == 0, "runs did not fill the end and then wrap to row 0");
    require(server::take_rows(rows, 1) == 8, "a wrapped run fit over the oldest run");
    server::give_rows(rows, 6, 2);
    require(rows.runs.size() == 3 && server::take_rows(rows, 1) == 8, "a run given back early was reclaimed before the runs taken before it");
    server::give_rows(rows, 3, 3);
    require(rows.runs.size() == 1 && rows.runs.front().base == 0 && server::take_rows(rows, 5) == 3, "runs given back did not come back oldest first");
    server::give_rows(rows, 0, 3);
    server::give_rows(rows, 3, 5);
    require(rows.runs.empty() && server::take_rows(rows, 8) == 0, "an empty ring did not start at row 0");
}

// An uncapped request whose growth step falls due and a queued capped request that fits only in the room that step needs, in one round: the step takes the room and the newer request waits.
void due_step_first() {
    Sim sim(1, 1, 1);
    sim.max_seqs = 3;
    sim.ubatch = 512;
    sim.grow = 256;
    sim.pool = {18, 9};
    sim.rows.size = sim.max_seqs;
    Sim::Req a;
    a.id = 1, a.admission = 1, a.uncapped = true, a.prompt = 40, a.gen = 345, a.len = 384;
    a.max_tokens = sim.limit() - a.prompt;
    a.need = sim.blocks_for(384);
    sim.active.push_back(a);
    sim.reserved = a.need;
    Sim::Req d;
    d.id = 2, d.prompt = 138, d.max_tokens = 600;
    sim.queue.push_back(d);
    sim.next_id = 3, sim.admissions = 1, sim.first_admitted = 1;
    require(sim.blocks_for(d.prompt + d.max_tokens) == std::vector<size_t>({12, 6}), "the case's queued request needs the room its step leaves free");
    sim.fail_per_mille = 0;
    sim.round();
    require(sim.stalls == 0 && !sim.active[0].stalled && sim.queue.size() == 1,
            "a growth step due beside a newer request's admission: " + std::to_string(sim.stalls) + " stalls and " + std::to_string(sim.queue.size()) +
            " queued, against 0 and 1");
}

// Random schedules: schedule n runs over 1 + n % 4 stages and 1 to twice that many pass slots, submissions arriving with the host's time, and ends either in a stop at a random round or, one in eight, once every request has ended.
void random_schedules(size_t n) {
    size_t totals[9] = {0}, rounds = 0;
    for (uint32_t seed = 1; seed <= n; ++seed) {
        const size_t S = 1 + seed % 4, P = 1 + (seed / 4) % (2 * S);
        Sim sim(seed, S, P);
        sim.randomize();
        const bool drain = seed % 8 == 0;
        const size_t events = 20 + sim.rng() % 200;
        const double rate = 0.05 + (double)(sim.rng() % 100) / 400.0;
        double last = 0;
        try {
            for (size_t e = 0; e < events; ++e) {
                for (double t = sim.clock - last; t > 0; t -= 1.0)
                    if ((double)(sim.rng() % 1000) < 1000.0 * rate * std::min(t, 1.0)) sim.submit();
                last = sim.clock;
                if (sim.rng() % 100 < 3) sim.cancel();
                sim.round();
            }
            if (drain) {
                // Submissions stop, and every request ends.
                const size_t submitted = sim.next_id - 1;
                for (int i = 0; i < 200000 && sim.ended < submitted; ++i) sim.round();
                require(sim.ended == submitted, std::to_string(submitted - sim.ended) + " requests never ended");
                require(sim.reserved == std::vector<size_t>(sim.reserved.size(), 0) || !sim.donors.empty(), "a drained pool holds blocks no donor holds");
            }
            sim.stop();
        } catch (const std::exception& e) {
            throw std::runtime_error("schedule " + std::to_string(seed) + " (" + std::to_string(S) + " stages, " + std::to_string(P) + " slots): " + e.what());
        }
        const size_t counts[9] = {sim.ended, sim.pauses, sim.stalls, sim.taken_back, sim.waits, sim.resolved, sim.failures, sim.flying_cancels, sim.unrecorded_cancels};
        for (size_t i = 0; i < 9; ++i) totals[i] += counts[i];
        rounds += sim.round_no;
    }
    // Enough schedules must meet every rule's case.
    if (n >= 1000)
        for (size_t i = 1; i < 9; ++i) require(totals[i] > 0, "the schedules met no case " + std::to_string(i) + " of pauses, stalls, donors taken back, waits on requests in flight, waits ended, failures and cancellations in flight");
    std::printf("server-passes: %zu schedules, %zu rounds, %zu requests, %zu pauses, %zu stalls, %zu donors taken back, %zu waits on requests in flight (%zu of the oldest ended as the request left flight), %zu failed passes, %zu cancellations in flight (%zu before a first stage)\n",
                n, rounds, totals[0], totals[1], totals[2], totals[3], totals[4], totals[5], totals[6], totals[7], totals[8]);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const size_t schedules = argc > 1 ? (size_t)std::stoul(argv[1]) : 2000;
        rooms_by_hand();
        rounds_by_hand();
        rows_by_hand();
        due_step_first();
        random_schedules(schedules);
        std::cout << "server-passes: " << checks << " checks pass\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "server-passes: " << e.what() << '\n';
        return 1;
    }
}
