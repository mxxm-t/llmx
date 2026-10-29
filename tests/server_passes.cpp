// The scheduler's policy core (server/policy.hpp) by hand, then its round under a simulated executor in random schedules (AGENTS.md, Tests).
// `llmx-server-passes-test N` runs N schedules, 2000 by default; `--replay FILE` or `--load NAME` replays loads under each share rule instead (docs/SERVER.md, Prompt shares).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "server/policy.hpp"

namespace {

size_t checks = 0;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}

constexpr size_t npos = (size_t)-1;

// make_room over counts that share nothing, as the simulation's ledger keeps them: a donor gives its blocks, a paused request its reservation (`needs`) less what it keeps, and with its donor that too.
server::Taken room_of(const std::vector<size_t>& pool, const std::vector<size_t>& reserved, const std::vector<std::vector<size_t>>& donors, size_t keep, bool keep_first, const std::vector<server::Holder>& active,
                      const std::vector<std::vector<size_t>>& needs, uint64_t after, bool preempt, const std::vector<size_t>& need) {
    std::vector<size_t> free(pool.size());
    for (size_t s = 0; s < pool.size(); ++s) free[s] = pool[s] - reserved[s];
    std::vector<size_t> order(donors.size());
    for (size_t d = 0; d < order.size(); ++d) order[d] = d;
    return server::make_room(free, order, keep, keep_first, active, after, preempt, need, [&](const server::Taken& t) {
        std::vector<size_t> g(pool.size(), 0);
        for (size_t d : t.donors) server::add(g, donors[d]);
        for (const auto& [i, with] : t.paused) {
            server::add(g, server::less(needs[i], active[i].kept));
            if (with) server::add(g, active[i].kept);
        }
        return g;
    });
}

// A stage's time in milliseconds as a replay prices it (docs/SERVER.md, Prompt shares): a replies' pass, a chunk pass's fixed time, a prompt row by its depth and each further row class.
// The defaults fit Qwen3.8-Flash-Next's stage device times over four MI50s in two stages of two (docs/STATUS.md, the serving rework); a reply's further rows and a further class are estimates.
struct Prices {
    double reply_ms = 11.9, reply_row_ms = 3.25, reply_depth_ms = 7.9e-6;
    double chunk_ms = 46, row_ms = 0.48, row_depth_ms = 3.99e-6, class_ms = 5;
    static size_t row_class(size_t extent) { return extent <= 1 ? 0 : (std::min<size_t>(extent, 512) + 63) / 64; }
};

// A replay's rule (docs/SERVER.md, The service rule) and what it measures: the share rule, the gap's ratio, the decode columns, each stage's price table and each prompt's credit.
struct Service {
    server::Share share;
    double mix = 3;
    size_t columns = 16;
    double record_ms = 0.2;
    Prices prices;
    std::vector<server::PriceTable> tables;
    std::map<uint64_t, double> credit;
    std::vector<double> gaps;   // every gap between two tokens of a request
    size_t chunk_passes = 0, chunk_rows = 0;
};

// The scheduler's round over the policy core, driven by a simulated executor: stage s runs on device s for a random time, the host waiting for a handoff before recording the next stage and for a last stage before it samples.
// The pools, the growth rule, make_room, the round's stages and the logits rows are the scheduler's own functions, and the order of the round's steps is the scheduler's run restated here.
struct Sim {
    struct Req {
        uint64_t id = 0, admission = 0, donor = 0;   // donor: the one its last pause left, which it takes back if it is still there
        uint64_t line = 0;                           // its place in the line of what waits: its submission's number, 0 once paused for another's growth, a new number once it gave the pool's edge up
        size_t edge = 0;                             // growth steps it took, its admission one of them, while an uncapped request was paused
        bool uncapped = false, stalled = false, cancel = false;
        size_t prompt = 0, max_tokens = 0, gen = 0, len = 0;   // len: what its cache holds, a pass in flight's rows not counted
        size_t read_alone = 0;                                // prompt rows it read while alone (server::prompt_slice)
        std::vector<size_t> in;                               // the slots of its passes in flight, oldest first: one, or a prompt's consecutive slices (docs/MULTI-DEVICE.md, phase 4)
        uint64_t since = 0, cancelled = 0;                    // the round it last got a token or could not, and the round its cancel was first seen in
        uint64_t landed = 0;                                  // the formation order of the last pass it left flight from
        uint64_t sliced = 0, prompt_wait = 0;                 // the round a pass last read its prompt, and the round its wait for one began
        // A read of its history from disk (docs/DISK-TIER.md, Restore): whether it has one, the round the read ends, failed or not, and the round its bound passes; it waits while both are ahead.
        bool reads = false, read_fails = false;
        uint64_t read_end = 0, bound = 0;
        std::vector<size_t> need;
        // Under a replay: the history its first admission forks, its place in the load, its times and its longest wait for rows while ready.
        size_t base = 0, tag = 0;
        double arrived = 0, first_at = -1, last_token = -1, rows_at = 0, attained = 0, worst_wait = 0, ended_at = 0;
        size_t chunks = 0;
        size_t history() const { return prompt + gen; }
        bool decoding() const { return gen && len + 1 == history(); }
    };
    struct Donor {
        uint64_t id = 0;
        size_t len = 0;
        std::vector<size_t> blocks;
    };
    struct Pass : server::Flight {
        size_t base = 0, want = 0, decoders = 0;
        std::vector<uint64_t> ids;   // by entry
        std::vector<size_t> rows;
        std::vector<char> wants;
        double ready = 0;            // when its last recorded stage ends on its device
        std::vector<size_t> from, extent;   // by entry, the history it found and its extent, which a replay prices it by
    };
    size_t S, P, lap, max_seqs = 4, ubatch = 16, fail_per_mille = 2;
    std::vector<size_t> failed_now;   // the slots a failure emptied in this round's steps
    uint64_t formed_before = 0;       // the passes formed before this round's steps
    bool short_ahead = false;         // a short prompt went ahead of a long one in flight this round (Scheduler::form, server::short_prompts)
    uint64_t short_round = 0;         // the last round a pass read a short prompt ahead of a long one
    // A model whose layers keep a recurrent state and no checkpoint slots: each active request holds one of max_seqs state slots, and no request leaves a donor.
    bool stateful = false;
    // Turns at the pool's edge (server::Growth::yields): whether a yielding request's history can be kept, as with a host tier that has room, and the steps of a turn.
    bool tier = false;
    size_t turn_steps = 1, yields = 0;
    server::Pools pools{{64, 32}, {64, 128}};
    server::Growth growth{16};
    std::vector<size_t> reserved{0, 0};
    std::vector<Donor> donors;               // oldest first
    std::vector<Req> queue, active, paused;  // active and paused in order of first admission
    std::vector<Pass> slots;
    std::vector<char> host;                  // per stage, whether it runs on the host, whose stages the round records after its device stages
    // A stage recorded on a thread of its own, as a tensor group's is (the scheduler's recorders): its recording takes 0 to `slow` rounds, the round going on meanwhile, and now and then fails as it ends.
    // Half of those schedules are a layer split's, whose stage of one device goes to its thread only while another pass is in flight (recorded_apart).
    std::vector<char> threaded;
    bool grouped = true;
    // Some of those with no stage on the host form the generating requests' rows in passes apart, as a scheduler with a share does: a slot a stage beside the P slots, with logits rows of their own.
    // Whether the rows ride a round's passes is drawn, where the scheduler prices it (rides).
    bool apart_mode = false;
    server::LogitRows rows_apart;
    // Under a replay the stages take the prices' times, each starting once its device is free, the pass's stage before has ended and the host has recorded it, and the passes are formed by the service rule (form_service); null otherwise.
    Service* service = nullptr;
    std::vector<Req> done;   // under a replay, every request that ended
    server::LogitRows& ring(const Pass& p) { return p.apart ? rows_apart : rows; }
    void clear(Pass& p) {
        const bool apart = p.apart;
        p = Pass{};
        p.apart = apart;
    }
    struct Recording {
        bool busy = false, fails = false;
        size_t slot = 0;
        uint64_t until = 0;
    };
    std::vector<Recording> recording;
    size_t slow = 0, recorded = 0, record_failures = 0, held_formations = 0, formed_apart = 0, passed = 0, continued_passes = 0, short_reads = 0, shorts_ahead = 0;
    server::LogitRows rows;
    std::vector<double> device_free;
    std::vector<uint64_t> device_last;       // per device, the formation order of the last pass it ran
    double clock = 0;
    uint64_t round_no = 0, formed = 0, next_id = 1, admissions = 0, donor_ids = 0, first_admitted = 0;
    uint64_t waiter = 0, waited = 0, waiting_from = 0;   // the oldest request while its plan waits on a request in flight, that request, and the round it first waited
    size_t ended = 0, stalls = 0, pauses = 0, taken_back = 0, waits = 0, resolved = 0, failures = 0, flying_cancels = 0, unrecorded_cancels = 0;
    size_t passed_reads = 0, reads_past_bound = 0, failed_reads = 0;   // requests admitted past one waiting for its read, reads given up at their bound, reads that failed
    std::mt19937 rng;
    std::string at;   // what the simulation was doing, for a failure's message

    Sim(uint32_t seed, size_t stages, size_t passes) : S(stages), P(passes), lap(stages + passes - 1), rng(seed) {
        slots.resize(P);
        host.assign(S, 0);
        threaded.assign(S, 0);
        recording.assign(S, Recording{});
        device_free.assign(S, 0.0);
        device_last.assign(S, 0);
        rows = server::logit_rows(P, max_seqs);
    }
    // A small random configuration, so schedules pause, stall and hold often.
    void randomize() {
        max_seqs = 2 + rng() % 7;
        ubatch = rng() % 2 ? 16 : 4;
        const size_t b = rng() % 2 ? 4 : 8;
        pools.block_tokens = {b, 2 * b};
        pools.blocks = {12 + rng() % 36, 0};
        pools.blocks[1] = pools.blocks[0] / 2 + rng() % 4;
        growth.tokens = b * (1 + rng() % 4);
        rows = server::logit_rows(P, max_seqs);
        stateful = rng() % 4 == 0;
        tier = !stateful && rng() % 2;
        turn_steps = 1 + rng() % 3;
        // Now and then one stage runs on the host, and now and then every one, as a split over CPUs.
        const size_t h = rng() % 8;
        if (h < 2) host[rng() % S] = 1;
        else if (h == 2) host.assign(S, 1);
        // One schedule in three records every stage on a thread of its own, each recording 0 to 2 rounds long; a stage's every step then takes up to slow + 1 rounds, and so does a lap.
        if (rng() % 3 == 0) {
            threaded.assign(S, 1);
            grouped = rng() % 2;
            // A layer split's first stage receives nothing and has no thread.
            if (!grouped) threaded[0] = 0;
            slow = rng() % 3;
            lap *= slow + 1;
            // Formation waits for the first stage's recorder, a stage's time more before a pass's first step.
            lap += slow + 1;
            if (std::none_of(host.begin(), host.end(), [](char h) { return h != 0; }) && rng() % 2) {
                apart_mode = true;
                threaded.assign(S, 1);
                slots.resize(P + S);
                for (size_t k = P; k < slots.size(); ++k) slots[k].apart = true;
                recording.assign(2 * S, Recording{});
                device_last.assign(2 * S, 0);
                rows_apart = server::logit_rows(S, max_seqs);
                // A decoder told to ride waits for a pass of the other kind to leave its slot, and then for its own.
                lap = 2 * lap + 1;
            }
        }
    }

    // Whether a pass waits for the first stage's recorder or is being recorded there, when no other is formed behind it.
    bool first_stage_held(bool apart) const {
        if (!threaded[0]) return false;
        for (const Pass& p : slots)
            if (p.live && p.ran == 0 && p.apart == apart) return true;
        return false;
    }

    // What a stage's device or thread does with the pass once it is recorded.
    void recorded_stage(Pass& p, size_t s) {
        if (service) {
            const double ms = stage_ms(p);
            const double start = std::max({clock, device_free[s], s ? p.ready : 0.0});
            p.ready = device_free[s] = start + ms;
            service->tables[s].observe(work_of(p), ms);
            clock = host[s] ? p.ready : clock + service->record_ms;
            ++p.ran;
            return;
        }
        if (s) clock = std::max(clock, p.ready);
        size_t n = 0;
        for (size_t r : p.rows) n += r;
        const double start = std::max(clock, device_free[s]);
        p.ready = device_free[s] = start + (0.5 + (double)(rng() % 100) / 100.0) * (1.0 + 0.05 * (double)n);
        clock = host[s] ? p.ready : clock + 0.05;
        ++p.ran;
    }

    // A pass's work as the price table keys it, a prompt's class by its extent (Prices::row_class).
    server::PassWork work_of(const Pass& p) const {
        server::PassWork w;
        for (size_t e = 0; e < p.ids.size(); ++e) {
            if (p.extent[e] == 1) w.decode(p.rows[e], p.from[e]);
            else w.chunk(Prices::row_class(p.extent[e]), p.rows[e], p.from[e]);
        }
        return w;
    }
    // A stage's time for a pass at the replay's prices.
    double stage_ms(const Pass& p) const {
        const Prices& c = service->prices;
        double ms = 0;
        size_t decode = 0, deepest = 0;
        std::vector<size_t> classes;
        for (size_t e = 0; e < p.ids.size(); ++e) {
            if (p.extent[e] == 1) {
                decode += p.rows[e];
                deepest = std::max(deepest, p.from[e]);
                continue;
            }
            ms += (double)p.rows[e] * (c.row_ms + c.row_depth_ms * ((double)p.from[e] + (double)p.rows[e] / 2));
            classes.push_back(Prices::row_class(p.extent[e]));
        }
        if (decode) {
            ms += c.reply_ms + c.reply_row_ms * (double)decode + c.reply_depth_ms * (double)deepest;
            classes.push_back(0);
        } else {
            ms += c.chunk_ms;
        }
        std::sort(classes.begin(), classes.end());
        ms += c.class_ms * (double)(std::unique(classes.begin(), classes.end()) - classes.begin() - 1);
        return ms;
    }

    // The recorders that are done, taken at the start of a round: a recorded stage counts, a failed one fails its pass alone.
    void collect() {
        at = "collection";
        for (size_t i = 0; i < recording.size(); ++i) {
            const size_t s = i % S;
            Recording& rec = recording[i];
            if (!rec.busy || rec.until > round_no) continue;
            rec.busy = false;
            Pass& p = slots[rec.slot];
            require(p.live && p.recording && p.ran == s, at + ": a recorder held a pass that was not waiting on it");
            p.recording = false;
            if (rec.fails) {
                fail(rec.slot);
                ++record_failures;
            } else {
                recorded_stage(p, s);
                ++recorded;
            }
            check();
        }
    }

    size_t largest_block() const { return *std::max_element(pools.block_tokens.begin(), pools.block_tokens.end()); }
    size_t limit() const { return std::min(pools.blocks[0] * pools.block_tokens[0], pools.blocks[1] * pools.block_tokens[1]); }
    std::vector<size_t> blocks_for(size_t tokens) const { return pools.blocks_for(tokens); }
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
    size_t free_slot(bool apart = false) const {
        for (size_t k = apart ? P : 0; k < (apart ? slots.size() : P); ++k)
            if (!slots[k].live) return k;
        return npos;
    }
    // Rows its passes in flight add to a request's history, each pass's prepared blocks counted once.
    size_t flying_rows(const Req& r) const {
        size_t n = 0;
        for (size_t k : r.in)
            for (size_t e = 0; e < slots[k].ids.size(); ++e)
                if (slots[k].ids[e] == r.id) n += slots[k].rows[e];
        return n;
    }
    bool ready(const Req& r) const { return r.in.empty() && !r.stalled; }
    // Whether a request in flight continues its prompt in a pass of its own, as the scheduler's continues: each of its passes holds it alone, has run the first stage and does not want its logits.
    // It follows a slice formed in an earlier round; nothing continues while a request could not grow, nor a cancelled request, and beside generated rows a slot stays free.
    bool continues(const Req& r) const {
        if (r.in.empty() || r.cancel || r.decoding() || std::any_of(active.begin(), active.end(), [](const Req& q) { return q.stalled; })) return false;
        for (size_t k : r.in)
            if (slots[k].ids.size() != 1 || !slots[k].ran || slots[k].wants[0] || slots[k].formed > formed_before) return false;
        size_t free = 0;
        for (size_t k = 0; k < P; ++k) free += !slots[k].live;
        const bool generating = std::any_of(active.begin(), active.end(), [](const Req& q) { return !server::reads_prompt(q.gen, q.len, q.history()); });
        return free > (generating ? 1u : 0u);
    }
    // A paused request's history stays as a donor however short, so what it keeps is what it holds.
    std::vector<server::Holder> holders() const {
        std::vector<server::Holder> h;
        for (const Req& r : active)
            h.push_back({r.admission, r.uncapped, r.len ? blocks_for(r.len) : std::vector<size_t>(pools.blocks.size(), 0), !r.in.empty()});
        return h;
    }
    std::vector<std::vector<size_t>> needs() const {
        std::vector<std::vector<size_t>> n;
        for (const Req& r : active) n.push_back(r.need);
        return n;
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
        std::vector<size_t> sum(pools.blocks.size(), 0);
        for (const Req& r : active) {
            add(sum, r.need);
            const bool next = grown && r.decoding() && ready(r);
            const std::vector<size_t> held = blocks_for(r.len + flying_rows(r) + (next ? 1 : 0));
            for (size_t s = 0; s < held.size(); ++s) require(held[s] <= r.need[s], at + ": a request holds more than it reserved");
            for (size_t k : r.in) require(k < slots.size() && slots[k].live, at + ": a request in flight in no pass");
            // A request's passes in flight are its prompt's consecutive slices, each a pass of its own, oldest first, each past the stages of the one after it, and only the last wanting logits.
            for (size_t j = 1; j < r.in.size(); ++j) {
                const Pass& a = slots[r.in[j - 1]];
                const Pass& b = slots[r.in[j]];
                require(a.formed < b.formed && a.ran >= b.ran && a.ids.size() == 1 && b.ids.size() == 1 && !a.wants[0],
                        at + ": a request's passes in flight are not its prompt's consecutive slices");
            }
        }
        for (const auto& d : donors) add(sum, d.blocks);
        for (size_t s = 0; s < sum.size(); ++s) {
            require(reserved[s] <= pools.blocks[s], at + ": pool " + std::to_string(s) + " over-reserved");
            require(reserved[s] == sum[s], at + ": the ledger does not add up in pool " + std::to_string(s));
        }
        for (const auto* waiting : {&queue, &paused})
            for (const Req& r : *waiting) require(r.in.empty(), at + ": a waiting request in flight");
        // The slot ledger: a state slot for each active request, and none held by a donor.
        require(active.size() <= max_seqs, at + ": more requests active than there are slots");
        require(!stateful || donors.empty(), at + ": a model keeping a recurrent state kept a donor");
        std::vector<std::pair<size_t, size_t>> runs;
        for (size_t k = 0; k < slots.size(); ++k) {
            const Pass& p = slots[k];
            if (!p.live) continue;
            require(!p.ids.empty(), at + ": an empty pass in flight");
            for (uint64_t id : p.ids) {
                const size_t i = find(id);
                require(i != npos && std::find(active[i].in.begin(), active[i].in.end(), k) != active[i].in.end(), at + ": a pass holds a request that is not in flight in it");
            }
            for (size_t j = 0; j < slots.size(); ++j)
                if (j != k && slots[j].live) require(slots[j].formed != p.formed, at + ": two passes formed as one");
            if (p.want) {
                const server::LogitRows& own = p.apart ? rows_apart : rows;
                require(p.base + p.want <= own.size, at + ": a pass's logits rows past the reservation");
                bool held = false;
                for (const auto& r : own.runs) held = held || (r.base == p.base && r.n == p.want && !r.back);
                require(held, at + ": a pass in flight whose logits rows came back");
                runs.push_back({p.base + (p.apart ? rows.size : 0), p.want});
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
        // One request in three waits for a read of a random time, which fails one time in four, under a random bound.
        if (rng() % 3 == 0) {
            r.reads = true;
            r.read_fails = rng() % 4 == 0;
            r.read_end = round_no + rng() % 6;
            r.bound = round_no + rng() % 6;
        }
        r.line = ++admissions;
        queue.push_back(r);
    }

    // A request leaves the active set: its history a donor when it holds at least `least` tokens and the model keeps no state, its blocks returned otherwise; the donor's id, 0 when none.
    uint64_t park(size_t i, size_t least) {
        Req r = active[i];
        require(r.in.empty(), at + ": a request in flight parked");
        active.erase(active.begin() + (std::ptrdiff_t)i);
        sub(reserved, r.need);
        if (stateful || !r.len || r.len < least) return 0;
        if (donors.size() >= max_seqs) drop(0);
        donors.push_back({++donor_ids, r.len, blocks_for(r.len)});
        add(reserved, donors.back().blocks);
        return donor_ids;
    }
    void finish(size_t i) {
        if (service) {
            done.push_back(active[i]);
            done.back().ended_at = clock;
        }
        park(i, largest_block());
        ++ended;
    }

    bool enter(Req& r) {
        std::vector<size_t> need = blocks_for(growth.entry(r.history(), r.uncapped, r.max_tokens, r.gen));
        // A resumed request whose donor is still there takes it back, first; otherwise now and then the request forks a donor, which make_room keeps for last, or takes first when told to.
        size_t own = donors.size();
        for (size_t d = 0; r.donor && d < donors.size(); ++d)
            if (donors[d].id == r.donor) own = d;
        const bool take = own < donors.size();
        const size_t keep = take ? own : !donors.empty() && rng() % 2 ? rng() % donors.size() : npos;
        const bool keep_first = take || (keep < donors.size() && rng() % 2);
        const server::Taken t = room_of(pools.blocks, reserved, donor_blocks(), keep, keep_first, holders(), needs(), 0, false, need);
        if (!t.enough) {
            // Refused, so even every donor would not do.
            std::vector<size_t> all = reserved;
            for (const auto& d : donors) sub(all, d.blocks);
            bool short_somewhere = false;
            for (size_t s = 0; s < all.size(); ++s) short_somewhere = short_somewhere || all[s] + need[s] > pools.blocks[s];
            require(short_somewhere, at + ": admission refused though the donors held enough");
            return false;
        }
        require(t.paused.empty() && !t.wait, at + ": admission paused a request");
        std::vector<size_t> gone = t.donors;
        if (take && std::find(gone.begin(), gone.end(), own) == gone.end()) gone.push_back(own);
        std::sort(gone.begin(), gone.end(), std::greater<size_t>());
        // A donor taken back gives the request what it held; a resume otherwise recomputes all it holds, a first admission all its prompt, which is what the ledger sees.
        r.len = take ? donors[own].len : 0;
        if (service && !r.admission) {
            r.len = r.base;
            r.rows_at = clock;
        }
        for (size_t i : gone) drop(i);
        add(reserved, need);
        r.need = need;
        r.donor = 0;
        r.stalled = false;
        r.edge = 1;
        r.since = round_no;
        taken_back += take;
        for (size_t s = 0; s < need.size(); ++s)
            require(blocks_for(r.len)[s] <= need[s], at + ": a request took back more than it reserved");
// One line of what waits: a queued request is admitted behind every paused one that began to wait before it, those paused for another's growth always.
// A request that gave the edge up is admitted behind every queued one submitted before its yield; requests waiting for a read are passed.
        if (!r.admission)
            for (const Req& w : paused) require(reading(w) || (w.line && w.line > r.line), at + ": a queued request admitted before a paused one that waited before it");
        else if (r.line)
            for (const Req& w : queue) require(reading(w) || w.line > r.line, at + ": a request that gave the edge up admitted before a queued one submitted before its yield");
        if (!r.admission) {
            // First-come, but for requests still waiting for their reads, which those behind them pass.
            for (const auto* waiting : {&queue, &paused})
                for (const Req& w : *waiting)
                    if (w.id < r.id && !w.admission) {
                        require(reading(w), at + ": a request admitted before one submitted earlier that was not waiting for a read");
                        ++passed_reads;
                    }
            if (r.reads) {
                reads_past_bound += r.bound < r.read_end;
                failed_reads += r.read_fails && r.read_end <= r.bound;
            }
            first_admitted = std::max(first_admitted, r.id);
            r.admission = ++admissions;
        }
        active.insert(std::upper_bound(active.begin(), active.end(), r, [](const Req& a, const Req& b) { return a.admission < b.admission; }), r);
        return true;
    }

    // Whether it left a donor.
    bool pause(size_t i, bool yielded = false) {
        Req r = active[i];
        r.donor = park(i, 1);
        r.stalled = false;
        r.line = yielded ? ++admissions : 0;
        yields += yielded;
        paused.insert(std::upper_bound(paused.begin(), paused.end(), r, [](const Req& a, const Req& b) { return std::make_pair(a.line, a.admission) < std::make_pair(b.line, b.admission); }), r);
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
            if (active[i].cancel && active[i].in.empty()) finish(i);
            else ++i;
    }

    // Stage s of the pass in slot k; now and then it fails, which fails that pass alone; whether it was recorded.
    // A device stage takes its time on its device, a host stage on the thread, which computes it as it records it.
    bool record(size_t k, size_t s) {
        Pass& p = slots[k];
        require(p.live && p.ran == s, at + ": a stage recorded out of order");
        const size_t live = (size_t)std::count_if(slots.begin(), slots.end(), [](const Pass& q) { return q.live; });
        // A device runs the passes of a kind in formation order; a pass apart may pass a pass of the other kind there.
        const size_t i = s + (p.apart ? S : 0);
        // A pass apart recorded at a stage an older pass of the other kind has yet to leave passes it there.
        const bool passing = p.apart && std::any_of(slots.begin(), slots.end(), [&](const Pass& q) { return q.live && !q.apart && q.formed < p.formed && q.ran <= s; });
        // Where passes go apart a pass with prompt rows is fed in pieces on its stage's thread, and a pass apart goes there while another pass is in flight.
        if (threaded[s] && (apart_mode ? !p.apart || live > 1 : server::recorded_apart(grouped, live))) {
            // A stage records one pass of a kind at a time: a pass that finds its recorder taken waits for a later round.
            if (recording[i].busy) return false;
            require(!p.recording, at + ": a pass handed to a recorder twice");
            require(device_last[i] < p.formed, at + ": device " + std::to_string(s) + " ran a pass before one of its kind formed earlier");
            device_last[i] = p.formed;
            p.recording = true;
            passed += passing;
            recording[i] = Recording{true, rng() % 1000 < 10 * fail_per_mille, k, round_no + rng() % (slow + 1)};
            return true;
        }
        require(device_last[i] < p.formed, at + ": device " + std::to_string(s) + " ran a pass before one of its kind formed earlier");
        device_last[i] = p.formed;
        if (rng() % 1000 < fail_per_mille) {
            fail(k);
            return false;
        }
        passed += passing;
        recorded_stage(p, s);
        return true;
    }

    // A failed pass fails as the scheduler's fail_pass does: every device drained, the pass abandoned and its requests ended with the error, while every other pass in flight goes on but a continued prompt's other slices, which go with it.
    void fail(size_t k) {
        for (double t : device_free) clock = std::max(clock, t);
        const size_t flying = in_flight();
        const std::vector<uint64_t> ids = slots[k].ids;
        size_t taken = 0;
        for (uint64_t id : ids)
            for (size_t j : active[find(id)].in) {
                if (j == k) continue;
                for (Recording& rec : recording)
                    if (rec.busy && rec.slot == j) rec.busy = false;
                server::give_rows(ring(slots[j]), slots[j].base, slots[j].want);
                clear(slots[j]);
                failed_now.push_back(j);
                ++taken;
            }
        server::give_rows(ring(slots[k]), slots[k].base, slots[k].want);
        clear(slots[k]);
        failed_now.push_back(k);
        for (uint64_t id : ids) {
            const size_t i = find(id);
            sub(reserved, active[i].need);
            active.erase(active.begin() + (std::ptrdiff_t)i);
            ++ended;
        }
        require(in_flight() + 1 + taken == flying, at + ": a failed pass took other passes with it than its prompt's slices");
        ++failures;
    }

    // The pass in slot k after its last stage: its rows committed and each wanting row sampled but a cancelled request's, as the scheduler's retire does; now and then reading its logits fails, which fails the pass.
    void retire(size_t k) {
        Pass& p = slots[k];
        require(p.live && p.ran == S, at + ": a pass retired before its last stage");
        clock = std::max(clock, p.ready);
        if (rng() % 1000 < fail_per_mille) {
            fail(k);
            return;
        }
        for (size_t e = 0; e < p.ids.size(); ++e) {
            const size_t i = find(p.ids[e]);
            require(i != npos && !active[i].in.empty() && active[i].in.front() == k, at + ": a retired pass's request not in flight in it, or a later slice retired first");
            Req& r = active[i];
            r.len += p.rows[e];
            r.in.erase(r.in.begin());
            r.landed = p.formed;
            if (p.wants[e] && !r.cancel) {
                ++r.gen;
                r.since = round_no;
                if (service) {
                    if (r.first_at < 0) r.first_at = clock;
                    else service->gaps.push_back(clock - r.last_token);
                    r.last_token = clock;
                }
            }
            if (service && p.extent[e] != 1) r.rows_at = clock;
        }
        server::give_rows(ring(p), p.base, p.want);
        clear(p);
    }

    // After the round's retirements, the requests whose last token has been sampled finish, in order of first admission.
    void finish_ended() {
        for (size_t i = 0; i < active.size();)
            if (active[i].gen >= active[i].max_tokens) finish(i);
            else ++i;
    }

    // Growth, the earliest admitted first, of the requests not in flight: the step the growth rule gives, which make_room makes room for, is taken, and one it refuses, or whose plan would pause a request in flight, is sat out.
    // A step never passes one uncapped request's reservation, so a plan pauses one request at most, and the round that request's pass retires in makes room with it out of flight.
    void grow_all() {
        for (size_t i = 0; i < active.size(); ++i) {
            Req& r = active[i];
            r.stalled = false;
            if (!r.in.empty()) continue;
            const std::vector<size_t> step = growth.step(pools, r.uncapped, r.decoding(), r.len, r.need);
            if (step.empty()) continue;
            // Its turn at the pool's edge over, it gives its room to the uncapped request paused longest, which must then fit: that one is ahead of it in the line, so it is admitted before the yielder is.
            const Req* w = waiting();
            if (w && growth.yields(r.edge, turn_steps, true, tier)) {
                std::vector<size_t> held = reserved;
                sub(held, r.need);
                if (room_of(pools.blocks, held, donor_blocks(), npos, false, {}, {}, 0, false, blocks_for(growth.entry(w->history(), true, 0, w->gen))).enough) {
                    const uint64_t first = w->id, self = r.id;
                    pause(i--, true);
                    require(paused.back().id == self, at + ": a request that gave the edge up is not last in the line");
                    require(waiting() && waiting()->id == first, at + ": the request a turn went to is no longer first among the uncapped paused");
                    continue;
                }
            }
            const server::Taken t = room_of(pools.blocks, reserved, donor_blocks(), npos, false, holders(), needs(), r.admission, true, step);
            require(t.paused.size() <= 1, at + ": a growth plan paused " + std::to_string(t.paused.size()) + " requests, though one uncapped request's reservation holds a step");
            if (i == 0 && waiter == r.id) {
                if (find(waited) == npos || active[find(waited)].in.empty()) {
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
                for (size_t s = 0; s < could.size(); ++s) short_somewhere = short_somewhere || could[s] + step[s] > pools.blocks[s];
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
                require(!active[t.paused[0].first].in.empty(), at + ": a plan waited on no request in flight");
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
            r.edge = waiting() ? r.edge + 1 : 0;
        }
    }
    // The uncapped request paused longest.
    const Req* waiting() const {
        for (const Req& p : paused)
            if (p.uncapped && !p.cancel) return &p;
        return nullptr;
    }

    // Room, only while a slot is free: growth, then, while nothing is stalled, what waits in its one line: the paused for another's growth, then yielders and the queue by number.
    void room() {
        at = "growth";
        grow_all();
        check(true);
        const bool stalled = std::any_of(active.begin(), active.end(), [](const Req& r) { return r.stalled; });
        at = "admission";
        if (!stalled)
            server::admit_waiting(paused, queue, [&] { return active.size() < max_seqs; }, [](std::vector<Req>&, size_t) { return false; },
                                  [&](const Req& r) { return reading(r); }, [&](Req& r) { return enter(r); }, [](const Req& r) { return r.line; });
        check(true);
    }

    // Whether a request waits for its read: its read and its bound both ahead.
    bool reading(const Req& r) const { return r.reads && round_no < std::min(r.read_end, r.bound); }

    // New passes while a slot is free: ready decoding requests up to the decode share, then prompt rows up to the ubatch, each pass taking its logits rows and recording its first stage, or leaving a host stage in `deferred`.
    void form(std::vector<std::pair<size_t, size_t>>& deferred) {
        if (service) {
            form_service(deferred);
            return;
        }
        const bool ride = !apart_mode || rng() % 2;
        if (!ride) form(deferred, true, true);
        form(deferred, false, ride);
    }
    // Passes of one kind: passes apart of the decoding requests' tokens alone, or the others with those tokens only where they ride.
    void form(std::vector<std::pair<size_t, size_t>>& deferred, bool apart, bool decodes) {
        at = "formation";
        for (size_t k = free_slot(apart); k != npos; k = free_slot(apart)) {
            // No pass is formed behind one of its kind that waits for the first stage's recorder.
            if (first_stage_held(apart)) {
                ++held_formations;
                return;
            }
            Pass p;
            p.apart = apart;
            formed_apart += apart;
            size_t budget = apart ? 0 : ubatch, decoders = 0;
            std::vector<const Req*> waiting;
            for (const Pass& q : slots) decoders += q.live ? q.decoders : 0;
            for (const Req& r : active)
                if (decodes && r.decoding() && ready(r)) waiting.push_back(&r);
            decoders += waiting.size();
            // The share goes to the decoders that left flight earliest, joining in order of first admission.
            const size_t share = std::min(server::decode_share(decoders, apart ? S : P, S), waiting.size());
            std::stable_sort(waiting.begin(), waiting.end(), [](const Req* a, const Req* b) { return a->landed < b->landed; });
            waiting.resize(share);
            for (const Req& r : active)
                if (std::find(waiting.begin(), waiting.end(), &r) != waiting.end()) {
                    p.ids.push_back(r.id);
                    p.rows.push_back(1);
                    p.wants.push_back(1);
                    p.from.push_back(r.len);
                    p.extent.push_back(1);
                    ++p.decoders;
                }
            const bool alone = active.size() == 1 && queue.empty() && paused.empty();
            // Short prompts as the scheduler's form takes them outside an overlap, which the simulation has none of: read whole ahead of the oldest long prompt, ahead of one in flight once a round.
            const auto left = [&](const Req& q) { return q.history() - std::min(q.history(), q.len + flying_rows(q)); };
            std::vector<Req*> shorts;
            const auto lead = std::find_if(active.begin(), active.end(), [&](const Req& q) { return !q.decoding() && left(q); });
            if (budget && lead != active.end() && left(*lead) > ubatch / server::kShortShare && !server::prompt_slice(left(*lead), S, ubatch, alone, lead->read_alone).closes &&
                (lead->in.empty() || !short_ahead)) {
                std::vector<size_t> waiting;
                std::vector<Req*> later;
                for (auto it = lead + 1; it != active.end(); ++it) {
                    if (it->decoding() || !left(*it)) continue;
                    later.push_back(&*it);
                    waiting.push_back(ready(*it) && !it->cancel ? left(*it) : 0);
                }
                for (size_t i : server::short_prompts(waiting, ubatch)) shorts.push_back(later[i]);
                short_ahead = short_ahead || (!lead->in.empty() && !shorts.empty());
            }
            std::vector<Req*> order(shorts);
            for (Req& r : active)
                if (std::find(shorts.begin(), shorts.end(), &r) == shorts.end()) order.push_back(&r);
            bool held = false;
            for (Req* q : order) {
                Req& r = *q;
                if (r.decoding() || !budget || r.len >= r.history()) continue;
                const size_t at_len = r.len + flying_rows(r);
                if (at_len >= r.history()) continue;
                const server::PromptSlice s = server::prompt_slice(r.history() - at_len, S, ubatch, alone, r.read_alone);
                // A prompt whose slices in flight have each run the first stage continues in a pass of its own.
                const bool continued = !r.in.empty();
                if (continued && !(p.ids.empty() && continues(r))) {
                    held = s.closes;
                    if (held) break;
                    continue;
                }
                if (!continued && !ready(r)) continue;
                const bool short_one = std::find(shorts.begin(), shorts.end(), &r) != shorts.end();
                if (short_one) {
                    short_round = round_no;
                    ++short_reads;
                    shorts_ahead += !lead->in.empty();
                }
                const size_t n = short_one ? std::min(budget, r.history() - at_len) : std::min({budget, s.most, r.history() - at_len});
                budget -= n;
                r.sliced = round_no;
                r.read_alone = alone && at_len + n < r.history() ? r.read_alone + n : 0;
                p.ids.push_back(r.id);
                p.rows.push_back(n);
                p.wants.push_back(at_len + n == r.history());
                p.from.push_back(at_len);
                p.extent.push_back(r.prompt);
                if (continued) {
                    ++continued_passes;
                    break;
                }
                if (s.closes) break;
            }
            if (p.ids.empty()) {
                // No free slot idles while a request is ready: a pass comes out empty only when every active request is in flight or sits the round out, or a prompt read alone, in flight, holds the passes for its slice back to a whole ubatch.
                for (const Req& r : active) require(held || !ready(r) || r.decoding() != decodes || (apart && !r.decoding()), at + ": a free slot idled while a request was ready");
                return;
            }
            for (char w : p.wants) p.want += (size_t)w;
            p.base = server::take_rows(ring(p), p.want);
            // A failure gives every pass's rows back, and the passes in flight want at most one row a request, half the rows, so a run always fits.
            require(p.base != ring(p).size, at + ": no logits rows for a pass though the passes in flight want at most half of them");
            p.live = true;
            p.formed = ++formed;
            slots[k] = p;
            for (uint64_t id : p.ids) active[find(id)].in.push_back(k);
            check();
            // Now and then a request is cancelled while its pass is formed and not yet recorded.
            if (rng() % 40 == 0) {
                active[find(p.ids[rng() % p.ids.size()])].cancel = true;
                ++unrecorded_cancels;
            }
            if (host[0]) deferred.push_back({k, 0});
            else record(k, 0);
            check();
        }
    }

    // Whether a prompt in flight continues in a pass of its own under the service rule: each of its passes holds it alone, has run the first stage and does not want its logits, formed in an earlier round, and nothing could not grow.
    bool continuing(const Req& r) const {
        if (r.in.empty() || r.cancel || r.decoding() || std::any_of(active.begin(), active.end(), [](const Req& q) { return q.stalled; })) return false;
        for (size_t k : r.in)
            if (slots[k].ids.size() != 1 || !slots[k].ran || slots[k].wants[0] || slots[k].formed > formed_before) return false;
        return true;
    }
    // A pass formed in slot k: its logits rows, its requests in flight, and its first stage recorded or left for the host.
    void launch(Pass& p, size_t k, std::vector<std::pair<size_t, size_t>>& deferred) {
        for (char w : p.wants) p.want += (size_t)w;
        p.base = server::take_rows(ring(p), p.want);
        require(p.base != ring(p).size, at + ": no logits rows for a pass though the passes in flight want at most half of them");
        p.live = true;
        p.formed = ++formed;
        slots[k] = p;
        for (uint64_t id : p.ids) active[find(id)].in.push_back(k);
        check();
        if (host[0]) deferred.push_back({k, 0});
        else record(k, 0);
        check();
    }
    // The service rule's passes (docs/SERVER.md, The service rule): replies packed while a prompt waits, then one prompt's chunk a free slot, by the share rule and the price table.
    void form_service(std::vector<std::pair<size_t, size_t>>& deferred) {
        at = "formation";
        Service& v = *service;
        const auto left = [&](const Req& q) { return q.history() - std::min(q.history(), q.len + flying_rows(q)); };
        if (std::none_of(active.begin(), active.end(), [&](const Req& q) { return !q.decoding() && left(q) && !q.cancel; })) {
            form(deferred, false, true);
            return;
        }
        size_t decoders = 0, replying = 0, deepest = 0;
        for (const Pass& q : slots) {
            decoders += q.live ? q.decoders : 0;
            replying += q.live && q.decoders;
        }
        std::vector<const Req*> ready_now;
        for (const Req& r : active) {
            if (!r.decoding()) continue;
            deepest = std::max(deepest, r.len);
            if (ready(r)) ready_now.push_back(&r);
        }
        decoders += ready_now.size();
        std::stable_sort(ready_now.begin(), ready_now.end(), [](const Req* a, const Req* b) { return a->landed < b->landed; });
        const size_t packed = server::packed_passes(decoders, v.columns);
        // The gap the replies are served within, from the replies' own pass alone over the stages, and what it leaves a chunk.
        double target = std::numeric_limits<double>::infinity();
        if (decoders) {
            const size_t band = server::depth_band(deepest);
            const double alone = (double)S * v.tables[0].decode_ms(band, (decoders + S - 1) / S), replies = v.tables[0].decode_ms(band, std::min(decoders, v.columns));
            if (alone > 0 && replies > 0) target = server::chunk_budget(server::service_gap(alone, v.mix), replies);
        }
        for (size_t k = free_slot(); k != npos; k = free_slot()) {
            std::vector<Req*> prompts;
            for (Req& r : active)
                if (!r.decoding() && left(r) && !r.cancel && (r.in.empty() ? ready(r) : continuing(r))) prompts.push_back(&r);
            Pass p;
            if (!ready_now.empty() && (replying < packed || prompts.empty())) {
                const size_t n = std::min(ready_now.size(), v.columns);
                for (const Req& r : active)
                    if (std::find(ready_now.begin(), ready_now.begin() + (std::ptrdiff_t)n, &r) != ready_now.begin() + (std::ptrdiff_t)n) {
                        p.ids.push_back(r.id);
                        p.rows.push_back(1);
                        p.wants.push_back(1);
                        p.from.push_back(r.len);
                        p.extent.push_back(1);
                        ++p.decoders;
                    }
                ready_now.erase(ready_now.begin(), ready_now.begin() + (std::ptrdiff_t)n);
                ++replying;
                launch(p, k, deferred);
                continue;
            }
            if (prompts.empty()) return;
            std::stable_sort(prompts.begin(), prompts.end(), [](const Req* a, const Req* b) { return a->arrived < b->arrived; });
            std::vector<server::ReadyPrompt> views;
            std::vector<double> credit;
            for (const Req* r : prompts) {
                const size_t at_len = r->len + flying_rows(*r), rows = r->history() - at_len;
                Pass whole;
                whole.ids = {r->id};
                whole.rows = {rows};
                whole.from = {at_len};
                whole.extent = {r->prompt};
                views.push_back({stage_ms(whole), r->in.empty() ? clock - r->rows_at : 0.0, r->attained, rows});
                credit.push_back(v.credit[r->id]);
            }
            const size_t i = server::next_prompt(v.share, views, credit);
            Req& r = *prompts[i];
            const size_t at_len = r.len + flying_rows(r), most = std::min(ubatch, r.history() - at_len);
            size_t n = most;
            if (decoders) {
                n = most;
                for (size_t s = 0; s < S; ++s) n = std::min(n, v.tables[s].chunk_rows(Prices::row_class(r.prompt), server::depth_band(at_len), target, most));
            }
            if (r.in.empty()) r.worst_wait = std::max(r.worst_wait, clock - r.rows_at);
            p.ids.push_back(r.id);
            p.rows.push_back(n);
            p.wants.push_back(at_len + n == r.history());
            p.from.push_back(at_len);
            p.extent.push_back(r.prompt);
            const double ms = stage_ms(p);
            r.attained += ms;
            ++r.chunks;
            server::charge_share(v.share, views, credit, i, ms);
            for (size_t j = 0; j < prompts.size(); ++j) v.credit[prompts[j]->id] = credit.empty() ? 0.0 : credit[j];
            ++v.chunk_passes;
            v.chunk_rows += n;
            continued_passes += !r.in.empty();
            launch(p, k, deferred);
        }
    }

    // The round's steps as the scheduler's run takes them: cancellations, each device stage's oldest waiting pass from the last stage down, retirements, ended and cancelled requests, new passes while a slot is free, and last the host stages due.
    void steps() {
        at = "the sweep";
        sweep_waiting();
        check();
        collect();
        const server::Steps st = server::round_steps(std::vector<server::Flight>(slots.begin(), slots.end()), S);
        for (const auto& a : st.advance) require(!slots[a.first].recording && !recording[a.second + (slots[a.first].apart ? S : 0)].busy, "the round gave a stage a pass while it records one of its kind, or a pass being recorded");
        for (size_t k : st.retire) require(!slots[k].recording, "the round retired a pass being recorded");
        std::vector<std::pair<size_t, size_t>> deferred;
        formed_before = formed;
        short_ahead = false;
        at = "advance";
        failed_now.clear();
        for (const auto& a : st.advance) {
            // A failure this round takes a continued prompt's other slices, which the round's steps may still name; the scheduler passes over them.
            if (!slots[a.first].live && std::count(failed_now.begin(), failed_now.end(), a.first)) continue;
            require(slots[a.first].live && slots[a.first].ran == a.second, at + ": a stage given a pass that does not wait for it");
            if (host[a.second]) {
                deferred.push_back(a);
                continue;
            }
            record(a.first, a.second);
            check();
        }
        at = "retirement";
        for (size_t k : st.retire) {
            if (!slots[k].live && std::count(failed_now.begin(), failed_now.end(), k)) continue;
            retire(k);
            check();
        }
        finish_ended();
        check();
        at = "the active sweep";
        sweep_active();
        check();
        if (free_slot() != npos || (apart_mode && free_slot(true) != npos)) {
            room();
            form(deferred);
        }
        at = "host stages";
        for (const auto& a : deferred)
            if (slots[a.first].live && slots[a.first].ran == a.second && (!a.second || slots[a.first].formed <= formed_before)) {
                record(a.first, a.second);
                check();
            }
    }

    // One round, and after it the bounds a lap sets.
    void round() {
        ++round_no;
        for (auto* all : {&queue, &paused, &active})
            for (Req& r : *all)
                if (r.cancel && !r.cancelled) r.cancelled = round_no;
        steps();
        // With nothing in flight the host waits for a submission.
        if (!in_flight()) clock += 1.0;
        // A replay measures its service; the laps bound today's rule.
        if (service) return;
        // Laps: a ready decoder waits one lap at most, and a cancellation ends within one.
        for (Req& r : active) {
            if (!r.decoding() || r.stalled) r.since = round_no;
            require(round_no - r.since <= lap, "round " + std::to_string(round_no) + ": a decoder went " + std::to_string(round_no - r.since) +
                    " rounds without a token, past a lap of " + std::to_string(lap));
        }
        for (const auto* all : {&queue, &paused, &active})
            for (const Req& r : *all)
                require(!r.cancelled || round_no < r.cancelled + lap, "round " + std::to_string(round_no) + ": a cancellation outlived a lap");
        // Short prompts beside a long one: a ready short prompt behind a long one forms a slice within a lap of its wait's start, and short prompts read ahead keep the oldest prompt, while ready, from a slice for a lap at most.
        const auto left = [&](const Req& q) { return q.history() - std::min(q.history(), q.len + flying_rows(q)); };
        const auto lead = std::find_if(active.begin(), active.end(), [&](const Req& q) { return !q.decoding() && left(q); });
        const bool long_lead = lead != active.end() && left(*lead) > ubatch / server::kShortShare;
        for (Req& r : active) {
            const bool is_lead = lead != active.end() && &r == &*lead;
            const bool waits = left(r) && !r.decoding() && ready(r) && !r.cancel && (is_lead ? short_round == round_no : long_lead && left(r) <= ubatch / server::kShortShare);
            if (!waits || r.sliced == round_no || !r.prompt_wait) r.prompt_wait = round_no;
            require(round_no - r.prompt_wait <= lap, "round " + std::to_string(round_no) + ": " + (is_lead ? "the oldest prompt" : "a short prompt behind a long one") + " went " +
                    std::to_string(round_no - r.prompt_wait) + " rounds without a slice, past a lap of " + std::to_string(lap));
        }
    }

    void cancel() {
        const size_t n = active.size() + paused.size() + queue.size();
        if (!n) return;
        size_t k = rng() % n;
        if (k < active.size()) {
            active[k].cancel = true;
            flying_cancels += !active[k].in.empty();
        } else if ((k -= active.size()) < paused.size()) {
            paused[k].cancel = true;
        } else {
            queue[k - paused.size()].cancel = true;
        }
    }

    // Every pass in flight is abandoned, and then every request ends and every donor goes.
    void stop() {
        at = "the stop";
        recording.assign(recording.size(), Recording{});
        for (Pass& p : slots)
            if (p.live) {
                for (uint64_t id : p.ids) active[find(id)].in.clear();
                server::give_rows(ring(p), p.base, p.want);
                clear(p);
            }
        for (const Req& r : active) sub(reserved, r.need);
        ended += active.size() + paused.size() + queue.size();
        active.clear();
        paused.clear();
        queue.clear();
        while (!donors.empty()) drop(0);
        require(reserved == std::vector<size_t>(pools.blocks.size(), 0) && rows.runs.empty() && rows_apart.runs.empty(), "after a stop the ledger or the logits rows hold something");
    }
};

// A few cases by hand: a request that fits or cannot fit takes nothing, the donor it forks goes last or first, and growth pauses the latest admitted uncapped request after it, its donor going only when its headroom is short.
// A plan that would pause a request in flight waits and takes nothing.
void rooms_by_hand() {
    const std::vector<size_t> pool{10};
    const std::vector<std::vector<size_t>> donors{{2}, {3}};
    auto t = room_of(pool, {4}, donors, npos, false, {}, {}, 0, false, {1});
    require(t.enough && t.donors.empty(), "a request that fits took something");
    t = room_of(pool, {9}, donors, npos, false, {}, {}, 0, false, {7});
    require(!t.enough && t.donors.empty(), "a request that cannot fit took donors");
    t = room_of(pool, {9}, donors, 0, false, {}, {}, 0, false, {5});
    require(t.enough && t.donors == std::vector<size_t>({1, 0}), "the donor a request forks did not go last");
    t = room_of(pool, {9}, donors, 1, true, {}, {}, 0, false, {3});
    require(t.enough && t.donors == std::vector<size_t>({1}), "the donor a request shares whole did not go first");
    t = room_of(pool, {9}, donors, npos, false, {}, {}, 0, false, {4});
    require(t.enough && t.donors == std::vector<size_t>({0, 1}), "donors did not go oldest first, the newer alone sparing the oldest");
    t = room_of(pool, {9}, {{0}, {3}}, 0, true, {}, {}, 0, false, {3});
    require(t.enough && t.donors == std::vector<size_t>({1}), "a donor freeing nothing beside the others was taken");
    std::vector<server::Holder> active{{1, true, {1}}, {2, false, {1}}, {3, true, {2}}, {4, true, {0}}};
    const std::vector<std::vector<size_t>> needs{{2}, {2}, {3}, {2}};
    t = room_of(pool, {10}, {}, npos, false, active, needs, 1, true, {3});
    require(t.enough && !t.wait && t.paused == std::vector<std::pair<size_t, bool>>({{3, false}, {2, false}}), "growth did not pause the latest uncapped requests after it, headroom first");
    t = room_of(pool, {10}, {}, npos, false, active, needs, 1, true, {5});
    require(t.enough && t.paused == std::vector<std::pair<size_t, bool>>({{3, false}, {2, true}}), "a paused request's donor did not go when its headroom was short");
    t = room_of(pool, {10}, {}, npos, false, active, needs, 1, false, {1});
    require(!t.enough && t.paused.empty(), "admission paused a request");
    t = room_of(pool, {10}, {}, npos, false, active, needs, 3, true, {3});
    require(!t.enough && t.paused.empty(), "a request paused one admitted before it or a capped one");
    active[2].flying = true;
    t = room_of(pool, {10}, {}, npos, false, active, needs, 1, true, {3});
    require(t.enough && t.wait && t.paused == std::vector<std::pair<size_t, bool>>({{3, false}, {2, false}}), "a plan pausing a request in flight did not wait");
    t = room_of(pool, {10}, {}, npos, false, active, needs, 1, true, {2});
    require(t.enough && !t.wait && t.paused == std::vector<std::pair<size_t, bool>>({{3, false}}), "a plan pausing only requests not in flight waited");
    active[2].flying = false;
    active[1].flying = true;
    t = room_of(pool, {10}, {}, npos, false, active, needs, 1, true, {3});
    require(t.enough && !t.wait, "a capped request in flight held up a plan that does not pause it");
    // Two forks of one prefix, the gain asked of the victims together as Model::freed asks it: donors of blocks {0, 1, 2} and {0, 1, 3} each free one block alone and all four together.
    const std::vector<std::vector<int>> sets{{0, 1, 2}, {0, 1, 3}};
    const server::Gain together = [&](const server::Taken& p) {
        std::vector<int> refs(4, 0), mine(4, 0);
        for (const auto& b : sets)
            for (int id : b) ++refs[(size_t)id];
        for (size_t d : p.donors)
            for (int id : sets[d]) ++mine[(size_t)id];
        size_t n = 0;
        for (size_t id = 0; id < refs.size(); ++id) n += mine[id] && mine[id] == refs[id];
        return std::vector<size_t>{n};
    };
    t = server::make_room({0}, {0, 1}, npos, false, {}, 0, false, {1}, together);
    require(t.enough && t.donors == std::vector<size_t>({0}), "one fork's own block did not do for one block");
    t = server::make_room({0}, {0, 1}, npos, false, {}, 0, false, {2}, together);
    require(t.enough && t.donors == std::vector<size_t>({0, 1}), "two forks of one prefix were counted as freeing it one at a time");
    t = server::make_room({0}, {0, 1}, npos, false, {}, 0, false, {5}, together);
    require(!t.enough && t.donors.empty(), "forks freeing less than asked took something");
    // Victims go in the order given, the scheduler's ranking (Scheduler::ranked): the second donor ranked first goes first and, with the first ranked last but forked, still last.
    const server::Gain apart = [](const server::Taken& p) { return std::vector<size_t>{2 * p.donors.size()}; };
    t = server::make_room({0}, {1, 0}, npos, false, {}, 0, false, {2}, apart);
    require(t.enough && t.donors == std::vector<size_t>({1}), "the donor ranked first did not go first");
    t = server::make_room({0}, {1, 0}, 1, false, {}, 0, false, {2}, apart);
    require(t.enough && t.donors == std::vector<size_t>({0}), "the forked donor did not go last whatever its rank");
}

// The host tier's default size: max_seqs histories within the prefixes' share of the host budget (infer::host_shares), none where the share is unknown or a history takes nothing.
void host_cache_by_hand() {
    require(server::host_cache_default(100, 8, 1000) == 800, "8 histories of 100 bytes below a cap of 1000 bytes did not take 800");
    require(server::host_cache_default(100, 8, 800) == 800, "8 histories of 100 bytes at a cap of 800 bytes did not take 800");
    require(server::host_cache_default(100, 8, 500) == 500, "8 histories of 100 bytes past a cap of 500 bytes did not take 500");
    require(server::host_cache_default(std::numeric_limits<size_t>::max() / 2, 4, 500) == 500, "histories whose sum overflows did not take the cap");
    require(server::host_cache_default(100, 8, std::nullopt) == 0, "an unknown cap did not take none");
    require(server::host_cache_default(0, 8, 500) == 0, "histories of no bytes did not take none");
}

// An idle step's history (grow_tokens): the longest whose copy fits the room and allocates no more fresh slabs than the step takes, counted a device, so idle slabs on one device do not pay for another's.
void grow_step_by_hand() {
    // Two devices: the first takes a slab for any copy, the second a slab for every 128 tokens; four slabs stand idle on the first and none on the second.
    const size_t slab = size_t(64) << 20;
    const auto bytes = [&](size_t tokens) { return (1 + tokens / 128) * slab; };
    const auto fresh = [&](size_t tokens) { return tokens / 128; };
    require(server::grow_tokens(8 * slab, 4, 128, 4096, true, bytes, fresh) == 4 * 128, "a step of four fresh slabs took a history that allocates more, its bytes fitting the room through another device's idle slabs");
    require(server::grow_tokens(3 * slab, 4, 128, 4096, true, bytes, fresh) == 2 * 128, "a step took a history past its room");
    require(server::grow_tokens(64 * slab, 64, 128, 4096, true, bytes, fresh) == 4096, "a step with room and slabs for the longest history did not take it");
    // A shortest copy that needs more fresh slabs than a step takes is taken by a step that has taken nothing, and by no other.
    const auto wide = [](size_t) { return size_t(6); };
    require(server::grow_tokens(8 * slab, 4, 128, 4096, true, bytes, wide) == 128 && server::grow_tokens(8 * slab, 4, 128, 4096, false, bytes, wide) == 0, "a model whose shortest copy passes a step's slabs did not grow by that copy alone");
    require(server::grow_tokens(slab, 4, 128, 4096, true, bytes, wide) == 0 && server::grow_tokens(8 * slab, 4, 0, 4096, true, bytes, wide) == 0, "a step took a history with no room or no block for it");
}

// The start line for a host tier that holds some histories: turn_tokens finds the longest in whole blocks (0 where not one fits), the line names it, and turns_line selects among the sentences.
void turns_line_by_hand() {
    const auto slabs = [](size_t tokens) { return (tokens / 128 + 1) * (size_t(1) << 20); };
    require(server::turn_tokens(10 << 20, 128, 4096, slabs) == 9 * 128, "a tier of ten blocks less one for the state did not hold nine");
    require(server::turn_tokens(1 << 20, 128, 4096, slabs) == 0, "a tier of one block's bytes held a history with its state");
    require(server::turn_tokens(size_t(1) << 40, 128, 4096, slabs) == 4096, "a tier past the longest history did not hold it");
    require(server::turn_tokens(10 << 20, 0, 4096, slabs) == 0, "a model with no block tokens held a history");
    const std::string line = server::short_tier_line(size_t(10) << 20, size_t(33) << 20, 1152);
    require(line == "uncapped requests that fill the pool together take turns only while a request's history is within 1152 tokens, a longer one keeping the edge until it ends: host memory for evicted prefixes holds 10 MiB and the longest history a request may have takes 33 MiB (--host-cache-bytes)", "the line for a tier holding part was: " + line);
    require(server::short_tier_line(0, size_t(33) << 20, 0).find("do not take turns") != std::string::npos, "the line for a tier of none did not say turns are off");
    require(server::short_tier_line(size_t(1) << 20, size_t(33) << 20, 0).find("do not take turns") != std::string::npos, "the line for a tier that holds no history did not say turns are off");
    // The selection: a model that keeps a state with no checkpoint slot refuses every turn (Scheduler::yields), so the line says that first, whatever the tier holds.
    const size_t part = size_t(10) << 20, whole = size_t(40) << 20, need = size_t(33) << 20;
    require(server::turns_line(part, need, 1152, true, 0) == server::no_checkpoint_line(), "a tier holding part, on a model that keeps a state with no checkpoint slot, did not say so first: " + server::turns_line(part, need, 1152, true, 0));
    require(server::turns_line(0, need, 0, true, 0) == server::no_checkpoint_line(), "a tier of none, on a model that keeps a state with no checkpoint slot, did not say so first");
    require(server::turns_line(whole, need, 4096, true, 0) == server::no_checkpoint_line(), "a tier holding a history, on a model that keeps a state with no checkpoint slot, did not say so");
    require(server::turns_line(whole, need, 4096, true, 3).empty() && server::turns_line(whole, need, 4096, false, 0).empty(), "a tier holding a history, with a checkpoint slot or no state, did not take turns silently");
    require(server::turns_line(part, need, 1152, true, 3) == server::short_tier_line(part, need, 1152) && server::turns_line(part, need, 1152, false, 0) == server::short_tier_line(part, need, 1152), "a tier holding part, with a checkpoint slot or no state, did not give the length line");
    require(server::turns_line(0, need, 0, false, 0) == server::short_tier_line(0, need, 0), "a tier of none, on a model with no state, did not give the old sentence");
}

// The growth rule by hand: admission reserves a capped request's history and what it may still generate, an uncapped one's history and a step, and a step falls due only for an uncapped decoding request whose next position passes its blocks.
// Then the logits rows a context reserves, one pass's alone and twice that once passes overlap, and the decode share.
void growth_by_hand() {
    const server::Pools pools{{10, 5}, {4, 8}};
    const server::Growth g{6};
    require(g.entry(9, false, 20, 3) == 26 && g.entry(9, true, 20, 3) == 15, "admission did not reserve the history and what is left, or uncapped a step past it");
    require(pools.blocks_for(0) == std::vector<size_t>({0, 0}) && pools.blocks_for(41) == std::vector<size_t>({10, 5}), "blocks were not counted whole up to the pool");
    require(g.step(pools, true, true, 7, {2, 1}).empty(), "a step fell due while the next position fit");
    require(g.step(pools, true, true, 8, {2, 1}) == std::vector<size_t>({2, 1}), "a due step did not reach a step past the next position");
    require(g.step(pools, false, true, 8, {2, 1}).empty() && g.step(pools, true, false, 8, {2, 1}).empty(), "a capped or prefilling request grew");
    require(g.step(pools, true, true, 8, {}) == std::vector<size_t>({4, 2}), "a request with nothing reserved did not take the whole step");
    require(g.step(pools, true, true, 36, {9, 5}) == std::vector<size_t>({1, 0}), "a step passed what a pool holds");
    // A turn at the pool's edge: the steps that make its change kTurnShare of it, never under one, and one where a time is not measured; a request yields only once it has held the edge a turn, with one waiting and its own history kept.
    require(g.turn(0, 100) == 1 && g.turn(5, 0) == 1 && g.turn(1, 100) == 1 && g.turn(10, 100) == 1 && g.turn(11, 100) == 2 && g.turn(95, 100) == 10, "a turn was not the steps its change is a tenth of, or under one");
    require(g.change(0, 900) == 900 && g.change(900, 60) == 60 && g.change(60, 900) == 60 && g.change(900, 900) == 900 && g.turn(g.change(60, 900), 100) == 6 && g.turn(g.change(900, 900), 100) == 90,
            "a change of turn was not timed as the smaller of the last two: one slow change between quick ones moved the turn, or two in a row did not");
    require(g.yields(2, 2, true, true) && g.yields(3, 2, true, true) && !g.yields(1, 2, true, true) && !g.yields(2, 2, false, true) && !g.yields(2, 2, true, false), "a request gave the edge up before its turn, to nobody, or with nowhere to keep its history");
    const server::LogitRows one = server::logit_rows(1, 5), many = server::logit_rows(4, 5);
    require(one.size == 5 && many.size == 10 && one.runs.empty(), "the logits rows were not one pass's alone, or twice that once passes overlap");
    require(server::decode_share(0, 3, 3) == 0 && server::decode_share(7, 1, 1) == 7 && server::decode_share(7, 3, 2) == 3 && server::decode_share(6, 3, 3) == 2 && server::decode_share(1, 4, 2) == 1,
            "the decode share was not the decoding requests over passes that fill the stages, rounded up");
    require(server::decode_share(7, 2, 3) == 7, "passes that do not fill the stages held a decoding request back");
    require(server::recorded_apart(true, 1) && server::recorded_apart(true, 3), "a tensor group's stage stayed on the scheduler's thread");
    require(!server::recorded_apart(false, 1) && server::recorded_apart(false, 2), "a single device's stage left the scheduler's thread for a pass alone, or stayed on it beside another");
    // Short prompts beside a long one: the arrival-order prefix of the ready ones that fit a kShortShare-th of the budget, 4 rows of 16 here.
    using Taken = std::vector<size_t>;
    require(server::short_prompts({4}, 16) == Taken{0} && server::short_prompts({5}, 16).empty(), "a prompt of the share's rows was not short, or one row more was");
    require(server::short_prompts({3, 2, 1}, 16) == Taken{0}, "a short prompt was passed over for a smaller later one when the share was tight");
    require(server::short_prompts({0, 9, 4}, 16) == Taken{2} && server::short_prompts({1, 1, 2}, 16) == (Taken{0, 1, 2}), "a not-ready or long prompt ended the prefix, or short ones that fit were left out");
    require(server::short_prompts({1}, 3).empty() && server::short_prompts({0, 0}, 16).empty(), "a share below one row, or no ready short prompt, took one");
    const auto most = [](size_t waiting, size_t stages, size_t ubatch, bool alone) { return server::prompt_slice(waiting, stages, ubatch, alone, 0).most; };
    require(most(512, 1, 512, true) == 512 && most(2048, 2, 512, true) == 512 && most(9000, 3, 512, true) == 512, "a prompt slice was cut on one stage or past the ubatch");
    require(most(512, 2, 512, false) == 512 && most(300, 3, 512, false) == 512, "a prompt slice was cut beside other requests");
    require(most(512, 2, 512, true) == 128 && most(1000, 2, 512, true) == 250 && most(1024, 2, 512, true) == 256 && most(900, 3, 512, true) == 150,
            "a lone prompt's slices were not a 2 * stages-th of its rows");
    require(most(300, 2, 512, true) == 128 && most(40, 2, 512, true) == 128 && most(100, 2, 64, true) == 64, "a prompt slice went below 128 rows, or past the ubatch");
    const server::PromptSlice back = server::prompt_slice(384, 2, 512, false, 128), whole = server::prompt_slice(384, 2, 512, false, 1024), lone = server::prompt_slice(384, 2, 512, true, 128);
    require(back.most == 384 && back.closes && whole.most == 512 && !whole.closes && !lone.closes,
            "company did not bring a prompt read alone back to a whole ubatch in a pass of its own, or a prompt on one already, or alone, was held to one");
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
    // A pass being recorded: its stage takes no other pass this round, and it neither advances nor retires; a pass formed without its first stage waits for the round to give it that stage.
    st = server::round_steps({{true, 5, 1, true}, {true, 3, 1, false}, {true, 4, 2, false}, {true, 6, 0, false}, {true, 7, 0, false}}, 3);
    require(st.advance == std::vector<std::pair<size_t, size_t>>({{2, 2}, {3, 0}}) && st.retire.empty(),
            "a stage took a pass while it recorded another, or the first stage did not take its oldest waiting pass");
    st = server::round_steps({{true, 1, 2, true}, {true, 2, 3, false}}, 3);
    require(st.advance.empty() && st.retire == std::vector<size_t>({1}), "a pass being recorded at its last stage advanced or retired");
    // Passes apart are a kind of their own: a stage that records a pass takes a pass apart beside it and no second pass of either kind, each kind in its formation order, and a pass apart retires past an older pass.
    st = server::round_steps({{true, 1, 0, true, false}, {true, 2, 0, false, false}, {true, 3, 0, false, true}, {true, 4, 0, false, true}}, 2);
    require(st.advance == (std::vector<std::pair<size_t, size_t>>{{2, 0}}), "a stage recording a pass took no pass apart, a second pass of its kind, or the younger pass apart first");
    st = server::round_steps({{true, 1, 0, false, false}, {true, 2, 0, true, true}, {true, 3, 0, false, true}}, 2);
    require(st.advance == (std::vector<std::pair<size_t, size_t>>{{0, 0}}), "a stage recording a pass apart took a second one, or held a pass of the other kind back");
    st = server::round_steps({{true, 1, 1, true, false}, {true, 2, 2, false, true}}, 2);
    require(st.advance.empty() && st.retire == std::vector<size_t>({1}), "a pass apart did not retire past an older pass still at a stage");
    // Generated rows ride a prompt pass where it costs them no more than their share would: always without a share, never before both prices are measured.
    require(server::rides(1000, 10, 0) && !server::rides(0, 10, 0.5) && !server::rides(1000, 0, 0.5), "generated rows did not ride without a share, or rode before the prices were known");
    require(server::rides(20, 10, 0.5) && !server::rides(21, 10, 0.5) && server::rides(40, 10, 0.25) && !server::rides(41, 10, 0.25), "generated rows rode a pass longer than their own over the share, or went apart from a shorter one");
    // A ride is bounded: on a price of eight slow passes the rows ride prompt passes of 1500 ms at a third, after kProbe of them one pass goes apart, a stage measures it at 46 ms, and the price sends the rows apart from then on.
    {
        server::StageShare stage;
        stage.share = 1.0 / 3;
        for (size_t i = 0; i < server::StageShare::kRecent; ++i) stage.generated(800);
        server::RideRun run;
        size_t ridden = 0;
        while (run.allow(server::rides(1500, stage.price(), stage.share))) {
            run.rode();
            require(++ridden <= server::RideRun::kProbe, "a ride on a stale price went on past its bound");
        }
        require(ridden == server::RideRun::kProbe && server::rides(1500, stage.price(), stage.share), "the rows did not ride a stale price up to the bound, or left it before a pass apart was measured");
        run.apart();
        stage.generated(46);
        require(!run.allow(server::rides(1500, stage.price(), stage.share)), "a pass apart measured at 46 ms did not end the ride");
        // A ride that pays goes on, one pass apart in every kProbe + 1.
        size_t rode = 0, apart = 0;
        for (size_t i = 0; i < 90; ++i) {
            if (run.allow(true)) {
                run.rode();
                ++rode;
            } else {
                run.apart();
                ++apart;
            }
        }
        require(rode == 80 && apart == 10 && !run.allow(false), "a ride that pays was not one pass apart in nine, or rows rode against the prices");
    }
    // A stage's share: a pass of generated rows of 50 ms between pieces of 10 ms is due again once the pieces since have earned what it spent, 15 pieces at a quarter and 5 at one half.
    server::StageShare none;
    require(!none.due() && none.ahead() && none.price() == 0, "a stage without a share was due, held no piece ahead, or had a price before any pass");
    for (const double share : {0.25, 0.5}) {
        server::StageShare stage;
        stage.share = share;
        require(stage.due(), "generated rows were not due on a stage with no time yet");
        double generated = 0, prompt = 0;
        size_t passes = 0, since = 0, longest = 0;
        for (int i = 0; i < 4000; ++i) {
            if (stage.due()) {
                stage.generated(50);
                generated += 50;
                longest = std::max(longest, since);
                since = 0;
                ++passes;
                require(!stage.ahead(), "the device held a piece ahead right after a pass of generated rows");
            }
            stage.piece(10);
            prompt += 10;
            ++since;
        }
        const double had = generated / (generated + prompt);
        require(passes > 100 && had > share - 0.01 && had < share + 0.01, "generated rows had " + std::to_string(had) + " of a stage at a share of " + std::to_string(share));
        require(longest == (share == 0.25 ? 15u : 5u), "a pass of generated rows waited " + std::to_string(longest) + " pieces at a share of " + std::to_string(share));
        for (size_t i = 0; i < server::StageShare::kQuiet; ++i) stage.piece(10);
        require(stage.ahead() && stage.due() && stage.owed_ms <= (1 - share) * 50, "a quiet stage held no piece ahead, or owed generated rows more than one pass");
        {
            const double before = stage.owed_ms;
            stage.unmeasured(10);
            require(stage.owed_ms == before - (1 - share) * 50, "a pass not measured did not spend the price");
        }
        // Time the stage stood still for the scheduler's work is spent as a pass's is, and what is owed for it stops at two passes.
        {
            server::StageShare still = stage;
            const double before = still.owed_ms;
            still.held(10);
            require(still.owed_ms == before - (1 - share) * 10, "ten milliseconds held were not spent");
            still.held(100000);
            require(still.owed_ms == -2 * (1 - share) * 50, "a long hold left the generated rows owing more than two passes");
        }
        // The price of a ride is the least of the last passes measured: one slow pass leaves it, and it follows once every one of them is slow.
        require(stage.price() == 50, "passes of 50 ms were priced " + std::to_string(stage.price()));
        // And one slow pass spends no more than two passes at the price, so the rows are due again before the device may hold a piece ahead.
        {
            const double before = stage.owed_ms;
            stage.generated(800);
            require(stage.price() == 50 && stage.owed_ms == before - (1 - share) * 100, "one slow pass priced the rest, or spent more than two passes");
            size_t waited = 0;
            while (!stage.due()) {
                stage.piece(10);
                ++waited;
            }
            require(waited < server::StageShare::kQuiet && !stage.ahead(), "after one slow pass the rows waited " + std::to_string(waited) + " pieces");
        }
        for (size_t i = 1; i < server::StageShare::kRecent; ++i) stage.generated(800);
        require(stage.price() == 800, "the price did not follow passes that are all slow");
    }
}

// The logits rows by hand: runs follow the newest and wrap to row 0, come back oldest first, and a run given back early waits for the ones before it.
// The automatic rule's account and clock (server::PromptFloor, server::ServiceClock), by hand.
void automatic_by_hand() {
    const double F = server::kPromptFloor, G = server::kServiceGapMs;
    {
        // A cold start: nothing owed and a service due at once, with no price asked for.
        const server::PromptFloor floor;
        const server::ServiceClock clock;
        require(floor.clear() && clock.due(0, 0), "a first service of generated rows was not due at a cold start");
    }
    {
        // A service is charged the floor's part of its whole time, whatever it took before: eight of 1 ms, then one of 100 ms at the same width.
        server::PromptFloor floor;
        for (int i = 0; i < 8; ++i) {
            floor.service(1);
            require(!floor.clear(), "a service of 1 ms left nothing owed");
            floor.piece(1);
            require(floor.clear(), "a piece as long as the service before it did not pay it off");
        }
        floor.service(100);
        require(floor.owed_ms == F * 100, "a service of 100 ms after eight of 1 ms was not charged whole");
        // Pieces pay it off with the rest of their own time, and no stage without a service resets it while replies and prompts go on waiting.
        size_t pieces = 0;
        while (!floor.clear()) {
            floor.piece(10);
            ++pieces;
        }
        require(pieces == (size_t)std::ceil(F * 100 / ((1 - F) * 10)), "what a slow service owed was paid off in " + std::to_string(pieces) + " pieces of 10 ms");
        floor.service(100);
        floor.piece(10);
        require(!floor.clear(), "what the generated rows owed was dropped while replies and prompts both waited");
        // Only the end of the overlap drops it.
        floor.demand_ended();
        require(floor.clear(), "the end of the overlap left the generated rows owing");
        // A zero-time service owes nothing.
        floor.service(0);
        require(floor.clear(), "a service of no time left something owed");
    }
    // The bound: with services taken only when nothing is owed, their time is at most (1 - F) / F of the pieces' and one service more, for fast, slow and changing services.
    for (const int kind : {0, 1, 2, 3}) {
        server::PromptFloor floor;
        double services = 0, pieces = 0, last = 0;
        size_t taken = 0, read = 0;
        uint64_t x = 88172645463325252ull + (uint64_t)kind;
        const auto next = [&x] { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
        for (int step = 0; step < 4000; ++step) {
            if (floor.clear()) {
                // 1 ms, 500 ms against the target of 100, a width that changes, and a jump after eight fast ones.
                last = kind == 0 ? 1 : kind == 1 ? 500 : kind == 2 ? 1 + (double)(next() % 300) : (taken % 9 == 8 ? 100 : 1);
                floor.service(last);
                services += last;
                ++taken;
            }
            const double piece = 1 + (double)(next() % 90);
            floor.piece(piece);
            pieces += piece;
            ++read;
            require(services <= (1 - F) / F * pieces + last + 1e-6, "the services took " + std::to_string(services) + " ms beside " + std::to_string(pieces) + " ms of pieces");
        }
        // Neither side stands still, an impossible target included: every service is followed by pieces and every wait by a service.
        require(taken > 10 && read == 4000, "kind " + std::to_string(kind) + ": " + std::to_string(taken) + " services in 4000 pieces");
    }
    {
        // The clock: a service is due where one more piece would pass the target, and a gap counts only while replies and prompts both wait.
        server::ServiceClock clock;
        clock.began(1000);
        require(clock.intervals == 0, "a gap was counted with no prompt waiting");
        // Due at the boundary nearest the target: with pieces of 80 ms not 30 ms after the last service, and 110 ms after it.
        require(!clock.due(1030, 80) && clock.due(1110, 80) && !clock.due(1000 + G - 6, 10) && clock.due(1000 + G - 5, 10) && clock.due(1000 + G, 0),
                "a service was not due at the piece boundary nearest the target");
        // A service that began 100 ms after its grant was offered is due by its own start: 30 ms later it is not, though by the grant's time it would be.
        {
            server::ServiceClock offered, taken;
            offered.began(0);
            taken.began(100);
            require(offered.due(130, 20) && !taken.due(130, 20), "a service's start at its grant and at its taking up were not told apart");
        }
        // From idle to overlap: the wait for the overlap's first service is no gap, whenever the service before it was.
        require(!clock.demand(true, 50000), "the start of an overlap was read as its end");
        clock.began(50020);
        require(clock.intervals == 0, "the wait for an overlap's first service was counted as a gap");
        // A pass that carries generated rows counts whatever kind it is, so one that misses the target shows.
        clock.began(50020 + G);
        clock.began(50020 + 2 * G + 1);
        clock.began(50020 + 3 * G);
        require(clock.intervals == 3 && clock.late == 1, std::to_string(clock.late) + " of " + std::to_string(clock.intervals) + " gaps above the target, against 1 of 3");
        // A pass that was due and waited for the floor is counted once, however many boundaries it waited at.
        bool counted = false;
        for (int i = 0; i < 5; ++i) clock.defer(counted);
        require(clock.deferred == 1, "a deferred pass was counted " + std::to_string(clock.deferred) + " times");
        // A round with no pass ready ends no overlap; only the requests do, and then no gap is counted.
        require(!clock.demand(true, 60000) && clock.demand(false, 61000) && !clock.demand(false, 62000) && clock.overlaps == 1, "the overlap's end was not told once, or its start counted twice");
        clock.began(70000);
        require(clock.intervals == 3, "a gap was counted after the overlap ended");
    }
    {
        // Holds that follow each other while the stage's thread stands parked: only the time of those taken for the generated rows is charged, in either order.
        server::HeldTime held;
        held.parked(0);
        held.begin(false, 0);
        held.end(10);
        held.begin(true, 10);
        held.end(30);
        require(held.take() == 20 && held.take() == 0, "a hold for other work before one for the generated rows changed what was charged");
        held.begin(true, 0);
        held.end(20);
        held.begin(false, 20);
        held.end(30);
        require(held.take() == 20, "a hold for other work after one for the generated rows was charged to them, or erased their charge");
        // Nested holds are the outermost's.
        held.begin(true, 0);
        held.begin(false, 5);
        held.end(8);
        held.end(10);
        held.begin(false, 10);
        held.begin(true, 12);
        held.end(14);
        held.end(20);
        require(held.take() == 10, "nested holds were charged wrongly");
        // A hold asked for while a piece still runs: the wait for the piece's boundary is the prompt's time, and only what follows the parking is charged.
        held.resumed();
        held.begin(true, 100);
        held.parked(140);
        held.end(160);
        require(held.take() == 20, "the wait for a prompt's piece boundary was charged to the generated rows");
        // A hold while no prompt's pass stands parked is nobody's.
        held.resumed();
        held.begin(true, 200);
        held.end(250);
        require(held.take() == 0, "a hold with no pass parked was charged");
    }
    {
        // A prompt's slices beside generated rows (server::PieceSearch): from the probe of one step, each next slice of a cost given by rows, its longest piece measured as that cost.
        const auto slices = [](server::PieceSearch& f, const std::function<double(size_t)>& cost, size_t n, size_t from = server::kPieceStep, double apart = 0) {
            std::vector<size_t> seen;
            for (size_t rows = from; seen.size() < n; rows = f.next(512)) {
                seen.push_back(rows);
                f.add(rows, cost(rows), apart);
            }
            return seen;
        };
        const auto search = [&](const std::function<double(size_t)>& cost, size_t n, double apart = 0) {
            server::PieceSearch f;
            return slices(f, cost, n, server::kPieceStep, apart);
        };
        // The 27B's attention at 211k is a staircase (F2DEV, 2026-10-19 10:35): 115 ms to 160 rows, 190 to 320, 288 past; the target the probe's 115 and 50 more, so the search settles on 160 and stays (12:38).
        const auto stair = [](size_t r) { return r <= 160 ? 115.0 : r <= 320 ? 190.0 : 288.0; };
        require(search(stair, 12) == std::vector<size_t>{16, 32, 64, 128, 256, 192, 160, 176, 160, 160, 160, 160}, "a prompt's slices did not settle on the top of the staircase's step");
        // A cost in proportion to the rows, 0.3 ms a row: the target is the probe's 4.8 ms and 50 more, and the search settles on the largest slice within it, 176 rows; a plateau doubles to the ubatch and stays.
        require(search([](size_t r) { return 0.3 * (double)r; }, 10) == std::vector<size_t>{16, 32, 64, 128, 256, 192, 160, 176, 176, 176} &&
                    search([](size_t) { return 115.0; }, 7) == std::vector<size_t>{16, 32, 64, 128, 256, 512, 512},
                "a prompt's slices did not settle on the largest within the target where the cost is a line or a plateau");
        // A reply's pass of 300 ms sets the target, which every slice of the staircase meets; and when the depth moves the step to 128 rows, the settled 160 is over, and the bracket opens again below it.
        server::PieceSearch moved;
        slices(moved, stair, 11);
        require(search(stair, 7, 300) == std::vector<size_t>{16, 32, 64, 128, 256, 512, 512} &&
                    slices(moved, [](size_t r) { return r <= 128 ? 115.0 : r <= 288 ? 190.0 : 288.0; }, 8, 160) == std::vector<size_t>{160, 112, 128, 144, 128, 128, 128, 128},
                "a prompt's slices did not take a reply's longer pass as the target, or did not open the bracket again when the step moved");
        // Two late host waits push acceptable cuts over the unchanged target, reopening [32, 64] down to [16, 32].
        server::PieceSearch late;
        const std::vector<size_t> late_rows{16, 32, 64, 48, 32, 16, 16};
        const std::vector<double> late_ms{100, 100, 175, 300, 300, 100, 100};
        for (size_t i = 0; i < late_rows.size(); ++i) {
            require((i ? late.next(512) : server::kPieceStep) == late_rows[i], "a late prompt piece did not reopen the staircase bracket below its former minimum");
            late.add(late_rows[i], late_ms[i], 0);
        }
        server::PieceSearch none;
        require(none.next(512) == 512 && none.next(16) == 16, "a prompt's slice with nothing measured was not a ubatch");
        // A short prompt's pass between a long prompt's slices (server::take_piece) leaves the stage's search to the long prompt: its next cut is the cut it would have had without the short pass.
        {
            server::PieceSearch with, without;
            uint64_t of_with = 0, of_without = 0;
            size_t end_with = 0, end_without = 0, at = 0;
            const std::vector<size_t> rows{16, 32, 64, 128};
            for (size_t i = 0; i < rows.size(); ++i) {
                server::take_piece(with, of_with, end_with, false, 7, at, at + rows[i], rows[i], 0.3 * (double)rows[i], 0);
                server::take_piece(without, of_without, end_without, false, 7, at, at + rows[i], rows[i], 0.3 * (double)rows[i], 0);
                at += rows[i];
                if (i == 1) require(!server::take_piece(with, of_with, end_with, true, 9, 0, 60, 60, 18, 0), "a short prompt's pass was taken by the stage's search");
            }
            require(with.next(512) == without.next(512) && of_with == of_without && end_with == end_without, "a short prompt's pass changed the long prompt's next cut");
            // Under the automatic rule a slice continues the search only where it starts at the last piece's end exactly; a short pass alone leaves that end at the long prompt's.
            require(of_with == 7 && end_with == at, "a short prompt's pass moved the end the long prompt's next slice must continue from under the automatic rule");
            require(server::take_piece(with, of_with, end_with, false, 9, 0, 60, 60, 18, 0) && with.next(512) == 120, "a pass of another prompt did not start the search again");
        }
        // A wait for a donor's write (server::write_wait_ms): the time waited, then the remaining bytes at the measured rate, half as long again and 2 s more, never past recomputing, and nothing to go by with no rate.
        require(server::write_wait_ms(1e9, 0, 0, 1e9) == 0 && server::write_wait_ms(1e9, 1e9, 0, 0) == 0, "a write's wait was bounded with no rate or no measure of recomputing");
        require(server::write_wait_ms(1e9, 1e9, 0, 1e9) == 3500 && server::write_wait_ms(0, 1e9, 0, 1e9) == 2000 && server::write_wait_ms(-5, 1e9, 0, 1e9) == 2000,
                "a write's wait was not 1.5 times its remaining time at the rate and 2 s more");
        require(server::write_wait_ms(1e9, 1e9, 12000, 1e9) == 15500 && server::write_wait_ms(1e9, 1e9, -1, 1e9) == 3500, "a write's wait did not count the time already waited");
        require(server::write_wait_ms(10e9, 0.5e9, 0, 1e9) == 32000 && server::write_wait_ms(10e9, 0.5e9, 0, 20000) == 20000 && server::write_wait_ms(1e9, 1e9, 0, 60000) == 3500 &&
                    server::write_wait_ms(1e9, 1e9, 19000, 20000) == 20000,
                "a write's wait was longer than recomputing the rows, or capped below its time");
        // The write in flight's own rate (server::in_flight_rate): none before a second or 256 MiB, then its bytes over its time.
        require(server::in_flight_rate(100 << 20, 0.5) == 0 && server::in_flight_rate(0, 0) == 0 && server::in_flight_rate(uint64_t(512) << 20, 0) == 0 &&
                    server::in_flight_rate(100 << 20, 2.0) == double(50 << 20) && server::in_flight_rate(uint64_t(512) << 20, 0.5) == double(uint64_t(1) << 30) && server::in_flight_rate(0, 3.0) == 0,
                "a write in flight's rate was not its bytes over its time past its minimum");
    }
    // A request is a prompt's before its first token and while its cache lacks more than its last token, as a resumed one's does; a reply's rows in a pass in flight, its cache one short or whole, keep it a reply.
    require(server::reads_prompt(0, 0, 40) && server::reads_prompt(0, 40, 40) && server::reads_prompt(30, 0, 70) && server::reads_prompt(30, 68, 70) && !server::reads_prompt(30, 69, 70) &&
                !server::reads_prompt(30, 70, 70),
            "a request was read as a prompt's or a reply's wrongly");
    // Generated rows ride a prompt's pass that takes no longer than the target, and go apart before one is measured.
    require(!server::rides_within_gap(0) && server::rides_within_gap(G) && !server::rides_within_gap(G + 0.5), "the ride did not follow the target gap");
    require(server::Pacing::fixed(0).mode == server::Pacing::Mode::off && server::Pacing::fixed(0.5).mode == server::Pacing::Mode::fixed && server::Pacing::automatic().mode == server::Pacing::Mode::automatic && server::Pacing::interleaved().mode == server::Pacing::Mode::interleaved && server::Pacing{}.mode == server::Pacing::Mode::off,
            "a pacing mode was read wrongly");
}

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
    sim.growth.tokens = 256;
    sim.pools.blocks = {18, 9};
    sim.rows = server::logit_rows(1, sim.max_seqs);
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
    size_t totals[14] = {0}, rounds = 0, recorded = 0, record_failures = 0, held_formations = 0, formed_apart = 0, passed = 0, continued_passes = 0, short_reads = 0, shorts_ahead = 0;
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
        const size_t counts[14] = {sim.ended, sim.pauses, sim.stalls, sim.taken_back, sim.waits, sim.resolved, sim.failures, sim.flying_cancels, sim.unrecorded_cancels,
                                   sim.stateful ? sim.pauses : 0, sim.passed_reads, sim.reads_past_bound, sim.failed_reads, sim.yields};
        for (size_t i = 0; i < 14; ++i) totals[i] += counts[i];
        rounds += sim.round_no;
        recorded += sim.recorded;
        record_failures += sim.record_failures;
        held_formations += sim.held_formations;
        formed_apart += sim.formed_apart;
        passed += sim.passed;
        continued_passes += sim.continued_passes;
        short_reads += sim.short_reads;
        shorts_ahead += sim.shorts_ahead;
    }
    // Enough schedules must meet every rule's case.
    if (n >= 1000) {
        require(formed_apart > 0 && passed > 0, "the schedules formed " + std::to_string(formed_apart) + " passes apart, of which " + std::to_string(passed) + " stages passed an older pass");
        require(recorded > 0 && record_failures > 0 && held_formations > 0, "the schedules met no stage recorded on a thread of its own, none that failed while it was recorded, or no formation held for the first stage's recorder");
        require(continued_passes > 0, "the schedules met no prompt continued while its slice before was in flight");
        require(short_reads > 0 && shorts_ahead > 0, "the schedules read no short prompt ahead of a long one, or none ahead of one in flight");
    }
    if (n >= 1000)
        for (size_t i = 1; i < 14; ++i) require(totals[i] > 0, "the schedules met no case " + std::to_string(i) + " of pauses, stalls, donors taken back, plans waiting on a request in flight, the oldest request's waits ended, failures, cancellations in flight, cancellations before a first stage, pauses of a model keeping a state, requests passing one waiting for its read, reads given up at their bound, reads that failed and turns at the pool's edge");
    std::printf("server-passes: %zu schedules, %zu rounds, %zu requests, %zu pauses (%zu keeping a state), %zu stalls, %zu donors taken back, %zu growth plans that waited on a request in flight, %zu waits of the oldest request ended in the round the request left flight, %zu failed passes, %zu cancellations in flight (%zu before a first stage), %zu admissions past a request waiting for its read, %zu reads given up at their bound, %zu failed reads, %zu turns at the pool's edge\n",
                n, rounds, totals[0], totals[1], totals[9], totals[2], totals[3], totals[4], totals[5], totals[6], totals[7], totals[8], totals[10], totals[11], totals[12], totals[13]);
    std::printf("server-passes: %zu stages recorded on a thread of their own, %zu of them failing as they ended, %zu formations held for the first stage's recorder, %zu prompt passes continuing a slice in flight, %zu short prompts read ahead of a long one (%zu ahead of one in flight)\n",
                recorded, record_failures, held_formations, continued_passes, short_reads, shorts_ahead);
}

// The service rule's policy by hand (docs/SERVER.md, The price table, The service rule, Prompt shares): depth bands, the price table's prices and its slice search, the gap and packing, and each share rule's choice.
void service_by_hand() {
    require(server::depth_band(0) == 0 && server::depth_band(4095) == 0 && server::depth_band(4096) == 1 && server::depth_band(8191) == 1 && server::depth_band(8192) == 2 &&
                server::depth_band(150000) == 6 && server::depth_band(262143) == 6 && server::depth_band(262144) == 7,
            "a depth band was read wrongly");
    server::PriceTable t;
    server::PassWork decode;
    decode.decode(1, 100);
    decode.decode(1, 5000);
    require(decode.decode_rows == 2 && decode.decode_band == 1, "decode rows took another band than their deepest's");
    require(t.predict(decode) == 0, "a table that measured nothing predicted");
    require(t.observe(decode, 10) && t.predict(decode) == 10, "a decode pass alone was not priced as measured");
    t.observe(decode, 14);
    require(std::fabs(t.decode.at({1, 2}).ms - (0.75 * 10 + 0.25 * 14)) < 1e-9 && t.decode.at({1, 2}).last_ms == 14 && t.decode.at({1, 2}).samples == 2, "a cell did not keep the forgetting mean and the last measure");
    server::PassWork three;
    three.decode(3, 5000);
    require(t.predict(three) == t.decode.at({1, 2}).ms, "three decode rows did not take the nearest measured rows of their band");
    const auto chunk = [](size_t cls, size_t rows, size_t depth) {
        server::PassWork w;
        w.chunk(cls, rows, depth);
        return w;
    };
    require(t.observe(chunk(7, 30, 0), 40) && t.chunks.count({7, 0, 32}), "a slice alone was not priced at its rows' step");
    server::PassWork both = chunk(7, 30, 0);
    both.decode(1, 5000);
    require(!t.observe(both, 99) && std::fabs(t.predict(both) - (40 + t.decode.at({1, 2}).ms)) < 1e-9, "a pass of two parts priced a cell, or was not predicted as its parts");
    require(t.predict(chunk(7, 64, 0)) == 80 && t.predict(chunk(7, 16, 0)) == 40 && t.predict(chunk(7, 16, 4096)) == 0 && t.predict(chunk(8, 16, 0)) == 0,
            "a slice's rows not measured were not priced from their class and band's nearest measure");
    // The search: one step at a class's first slice, doubling while within, the bracket's middle once over, half the smallest over where nothing is within, and a new band from the band below.
    server::PriceTable s;
    require(s.chunk_rows(3, 0, 50, 512) == 16, "the class's first slice was not one step");
    s.observe(chunk(3, 16, 0), 20);
    require(s.chunk_rows(3, 0, 50, 512) == 32, "a slice within the target did not double");
    s.observe(chunk(3, 32, 0), 30);
    s.observe(chunk(3, 64, 0), 45);
    s.observe(chunk(3, 128, 0), 75);
    require(s.chunk_rows(3, 0, 50, 512) == 96, "the bracket's middle was not taken");
    s.observe(chunk(3, 96, 0), 60);
    require(s.chunk_rows(3, 0, 50, 512) == 80, "the bracket did not narrow");
    s.observe(chunk(3, 80, 0), 52);
    require(s.chunk_rows(3, 0, 50, 512) == 64, "a bracket of one step did not settle on its slice within");
    s.observe(chunk(3, 64, 0), 55);
    require(s.chunk_rows(3, 0, 50, 512) == 48, "an overrun did not bring the bracket down to the largest within below it");
    require(s.chunk_rows(3, 0, 50, 20) == 20 && s.chunk_rows(3, 0, 50, 8) == 8, "the slice passed what the prompt has left");
    s.observe(chunk(3, 16, 0), 70);
    s.observe(chunk(3, 32, 0), 70);
    require(s.chunk_rows(3, 0, 50, 512) == 16, "a band whose every slice is over did not halve to one step");
    server::PriceTable n;
    n.observe(chunk(3, 128, 0), 40);
    n.observe(chunk(3, 256, 0), 70);
    require(n.chunk_rows(3, 2, 50, 512) == 128 && n.chunk_rows(4, 2, 50, 512) == 16, "a band not measured did not start at the band below's slice within, or another class at one step");
    n.observe(chunk(3, 128, 8192), 60);
    require(n.chunk_rows(3, 2, 50, 512) == 64, "an overrun in a new band did not halve its slice");
    require(server::service_gap(17, 3) == 51 && server::chunk_budget(51, 19) == 32 && server::chunk_budget(10, 19) == 0, "the gap or a chunk's budget was read wrongly");
    require(server::packed_passes(0, 16) == 0 && server::packed_passes(1, 16) == 1 && server::packed_passes(16, 16) == 1 && server::packed_passes(17, 16) == 2, "replies were not packed by the columns");

    // The share rules over three ready prompts in arrival order: an old long one, a newer short one and a newest middle one.
    std::vector<server::ReadyPrompt> p{{9000, 0, 500, 150000}, {300, 0, 0, 327}, {3000, 0, 0, 3334}};
    using R = server::Share::Rule;
    const auto pick = [&](server::Share s, const std::vector<double>& credit = {}) { return server::next_prompt(s, p, credit); };
    require(pick({R::oldest_first}) == 0, "oldest first did not take the oldest");
    server::Share today{R::today};
    today.short_rows = 128;
    require(pick(today) == 0, "today's rule read a prompt past its short rows ahead of the oldest");
    today.short_rows = 512;
    require(pick(today) == 1, "today's rule did not read the short prompt first");
    require(pick({R::least_remaining}) == 1, "least remaining did not take the least work left");
    server::Share aged{R::least_remaining};
    aged.age = 1;
    p[2].idle_ms = 2800;
    require(pick(aged) == 2, "least remaining with aging did not move a prompt that waited ahead");
    p[0].idle_ms = 10000;
    require(pick(aged) == 0, "a prompt idle for the bound did not go first");
    p[0].idle_ms = p[2].idle_ms = 0;
    require(pick({R::least_attained}) == 1, "least attained did not take the first that had none, the oldest of equals");
    // A rule that divides gives each its part over many chunks of equal time, a chunk at a time.
    const auto parts = [&](server::Share s) {
        std::vector<double> credit(p.size(), 0.0), got(p.size(), 0.0);
        for (int c = 0; c < 400; ++c) {
            const size_t i = server::next_prompt(s, p, credit);
            server::charge_share(s, p, credit, i, 1.0);
            got[i] += 1.0;
        }
        for (double& g : got) g /= 400.0;
        return got;
    };
    const std::vector<double> keeps = parts({R::oldest_keeps, 0.75}), equal = parts({R::equal}), floor = parts({R::least_remaining_floor, 0.25});
    require(std::fabs(keeps[0] - 0.75) < 0.01 && std::fabs(keeps[1] - 0.25) < 0.01 && keeps[2] == 0, "the oldest did not keep its part, the next in arrival order the rest");
    require(std::fabs(equal[0] - 1.0 / 3) < 0.01 && std::fabs(equal[2] - 1.0 / 3) < 0.01, "equal shares were not equal");
    require(std::fabs(floor[0] - 0.25) < 0.01 && std::fabs(floor[1] - 0.75) < 0.01 && floor[2] == 0, "the oldest's floor was not kept, the least remaining of the others taking the rest");
    require(server::share_parts({R::oldest_keeps, 0.75}, {p[2]}) == std::vector<double>{1.0} && server::share_parts({R::least_remaining}, p).empty(), "a lone prompt did not take all the time, or an ordering rule divided it");
}

// A request of a load: when it arrives, or the request it follows and the client's time after that one ends, its prompt, the history its first admission forks and the tokens it generates.
struct LoadRequest {
    double arrive_ms = 0;
    size_t prompt = 0, base = 0, gen = 1;
    long pred = -1;
    double think_ms = 0;
};
struct Load {
    std::string name;
    std::vector<LoadRequest> requests;
};
// The placement and pool a replay serves on: stages, passes in flight, requests at once, the ubatch and the pool's tokens.
struct Bench {
    size_t stages = 2, passes = 3, max_seqs = 8, ubatch = 512, pool_tokens = 262144;
};

// One replay of a load under a rule: the requests as they ended, every gap between two tokens of a request, and the requests skipped as larger than a request may hold.
struct Outcome {
    std::vector<Sim::Req> done;
    std::vector<double> gaps;
    double span_ms = 0;
    size_t chunk_passes = 0, chunk_rows = 0, skipped = 0;
};
Outcome replay(const Load& load, const Bench& b, const Service& rule) {
    Sim sim(1, b.stages, b.passes);
    sim.max_seqs = b.max_seqs;
    sim.ubatch = b.ubatch;
    sim.fail_per_mille = 0;
    sim.pools.block_tokens = {256, 512};
    sim.pools.blocks = {b.pool_tokens / 256, b.pool_tokens / 512};
    sim.growth.tokens = 256;
    sim.rows = server::logit_rows(b.passes, b.max_seqs);
    Service v = rule;
    v.tables.assign(b.stages, server::PriceTable{});
    v.credit.clear();
    v.gaps.clear();
    sim.service = &v;
    Outcome out;
    const size_t n = load.requests.size();
    const double never = std::numeric_limits<double>::infinity();
    std::vector<double> when(n, never);
    std::vector<char> sent(n, 0);
    std::vector<std::vector<size_t>> followers(n);
    for (size_t i = 0; i < n; ++i) {
        const LoadRequest& q = load.requests[i];
        if (q.pred < 0) when[i] = q.arrive_ms;
        else followers[(size_t)q.pred].push_back(i);
    }
    size_t submitted = 0, seen = 0;
    for (uint64_t rounds = 0;; ++rounds) {
        require(rounds < 200000000, load.name + ": the replay ran past its bound of rounds");
        for (; seen < sim.done.size(); ++seen)
            for (size_t j : followers[sim.done[seen].tag]) when[j] = sim.done[seen].ended_at + load.requests[j].think_ms;
        double next = never;
        for (size_t i = 0; i < n; ++i) {
            if (sent[i]) continue;
            if (when[i] > sim.clock) {
                next = std::min(next, when[i]);
                continue;
            }
            sent[i] = 1;
            const LoadRequest& q = load.requests[i];
            if (q.prompt + 1 >= sim.limit()) {
                ++out.skipped;
                // A skipped request's followers arrive as if it had ended at once.
                for (size_t j : followers[i]) when[j] = sim.clock + load.requests[j].think_ms;
                next = sim.clock;
                continue;
            }
            Sim::Req r;
            r.id = sim.next_id++;
            r.line = ++sim.admissions;
            r.tag = i;
            r.prompt = q.prompt;
            r.base = std::min(q.base, q.prompt - 1) / 256 * 256;
            r.max_tokens = std::max<size_t>(1, std::min(q.gen, sim.limit() - q.prompt));
            r.arrived = when[i];
            sim.queue.push_back(r);
            ++submitted;
        }
        if (sim.ended == submitted && next == never && std::all_of(sent.begin(), sent.end(), [](char c) { return c != 0; })) break;
        if (sim.active.empty() && sim.queue.empty() && sim.paused.empty() && !sim.in_flight()) {
            require(next != never, load.name + ": requests wait for a request that never ends");
            sim.clock = std::max(sim.clock, next);
            continue;
        }
        sim.round();
    }
    out.done = std::move(sim.done);
    out.gaps = std::move(v.gaps);
    out.span_ms = sim.clock;
    out.chunk_passes = v.chunk_passes;
    out.chunk_rows = v.chunk_rows;
    return out;
}

// The nearest-rank percentile of `v`, 0 for none.
double percentile(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, (size_t)std::ceil(q * (double)v.size()) - 1)];
}

// The candidate share rules A0 to A7 as the replay names them (docs/SERVER.md, Prompt shares).
struct Candidate {
    std::string name;
    server::Share share;
};
std::vector<Candidate> candidates(size_t ubatch) {
    using R = server::Share::Rule;
    std::vector<Candidate> c;
    server::Share today{R::today};
    today.short_rows = ubatch / server::kShortShare;
    c.push_back({"A0", today});
    c.push_back({"A1", {R::oldest_first}});
    c.push_back({"A2", {R::oldest_keeps, 0.75}});
    c.push_back({"A3", {R::oldest_keeps, 0.5}});
    c.push_back({"A4", {R::equal}});
    for (const double k : {1.0, 2.0, 4.0, 0.0})
        for (const double wait_s : {5.0, 10.0, 30.0}) {
            server::Share s{R::least_remaining};
            s.age = k > 0 ? 1.0 / k : 0.0;
            s.wait_ms = wait_s * 1000;
            c.push_back({"A5(k=" + (k > 0 ? std::to_string((int)k) : std::string("inf")) + ",W=" + std::to_string((int)wait_s) + "s)", s});
        }
    c.push_back({"A6(1/4)", {R::least_remaining_floor, 0.25}});
    c.push_back({"A6(1/2)", {R::least_remaining_floor, 0.5}});
    c.push_back({"A7", {R::least_attained}});
    return c;
}

// The figures of one load under one rule (docs/SERVER.md, Prompt shares), the score being the mean over the size classes present of the p95 slowdown against the prompt alone.
struct Figures {
    double score = 0, worst_wait_s = 0, longest_s = 0, longest_alone_s = 0, gap_p50_ms = 0, gap_p99_ms = 0, prompt_rate = 0, output_rate = 0, chunks_per_k = 0;
    size_t longest = 0, skipped = 0;
    std::string classes;
};
Figures figures(const Outcome& o, const std::function<double(size_t, size_t)>& alone_ms) {
    Figures f;
    static const size_t edges[] = {1000, 8000, 32000};
    static const char* const names[] = {"<1k", "1k-8k", "8k-32k", ">=32k"};
    std::vector<double> ttft[4], slow[4];
    size_t prompt = 0, generated = 0;
    for (const Sim::Req& r : o.done) {
        const size_t read = r.prompt - r.base;
        const size_t c = (size_t)(std::upper_bound(std::begin(edges), std::end(edges), read - 1) - std::begin(edges));
        const double t = r.first_at - r.arrived;
        ttft[c].push_back(t / 1000);
        slow[c].push_back(t / std::max(1.0, alone_ms(r.prompt, r.base)));
        f.worst_wait_s = std::max(f.worst_wait_s, r.worst_wait / 1000);
        prompt += read;
        generated += r.gen;
        if (read > f.longest) {
            f.longest = read;
            f.longest_s = t / 1000;
            f.longest_alone_s = alone_ms(r.prompt, r.base) / 1000;
        }
    }
    size_t present = 0;
    char line[256];
    for (size_t c = 0; c < 4; ++c) {
        if (ttft[c].empty()) continue;
        ++present;
        f.score += percentile(slow[c], 0.95);
        std::snprintf(line, sizeof line, " | %s n=%zu ttft p50/p95/p99 %.1f/%.1f/%.1f s, slowdown p50/p95/max %.2f/%.2f/%.2f", names[c], ttft[c].size(), percentile(ttft[c], 0.5),
                      percentile(ttft[c], 0.95), percentile(ttft[c], 0.99), percentile(slow[c], 0.5), percentile(slow[c], 0.95), percentile(slow[c], 1.0));
        f.classes += line;
    }
    f.score = present ? f.score / (double)present : 0;
    f.gap_p50_ms = percentile(o.gaps, 0.5);
    f.gap_p99_ms = percentile(o.gaps, 0.99);
    f.prompt_rate = o.span_ms > 0 ? 1000.0 * (double)prompt / o.span_ms : 0;
    f.output_rate = o.span_ms > 0 ? 1000.0 * (double)generated / o.span_ms : 0;
    f.chunks_per_k = o.chunk_rows ? 1000.0 * (double)o.chunk_passes / (double)o.chunk_rows : 0;
    f.skipped = o.skipped;
    return f;
}

// The UTC time `2026-10-27T11:03:12.972Z` in milliseconds since 1970, or a negative value where the text is not one.
double utc_ms(const std::string& t) {
    int y, mo, d, h, mi;
    double sec;
    if (std::sscanf(t.c_str(), "%d-%d-%dT%d:%d:%lfZ", &y, &mo, &d, &h, &mi, &sec) != 6) return -1;
    y -= mo <= 2;
    const long era = (y >= 0 ? y : y - 399) / 400;
    const long yoe = y - era * 400, doy = (153 * (mo + (mo > 2 ? -3 : 9)) + 2) / 5 + d - 1, doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    const double days = (double)(era * 146097 + doe - 719468);
    return ((days * 24 + h) * 60 + mi) * 60000.0 + sec * 1000.0;
}

// Production's requests with an arrival time from a trace of one JSON object a line (docs/SERVER.md, Prompt shares), a server's smoke request of 17 prompt and 151 generated tokens left out.
// A request follows the one it reuses at least half of, or whose end it arrives within 3 s of, unless the trace gives `pred` and `think_ms`.
Load trace(const std::string& path) {
    std::ifstream in(path);
    require((bool)in, "the trace " + path + " cannot be read");
    struct Row {
        LoadRequest q;
        double arrive = 0, end = -1;
        long given = -2;
    };
    std::vector<Row> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        const jmini::Value v = jmini::parse(line);
        const auto num = [&](const char* k, double fallback) {
            const jmini::Value* x = v.get(k);
            return x && x->isNumber() ? x->asNumber() : fallback;
        };
        const auto text = [&](const char* k) {
            const jmini::Value* x = v.get(k);
            return x && x->isString() ? x->asString() : std::string();
        };
        double arrive = utc_ms(text("arrival_utc_poller"));
        if (arrive < 0) arrive = utc_ms(text("arrival_utc_derived"));
        if (arrive < 0) continue;
        Row r;
        r.q.prompt = (size_t)num("prompt_tokens", 0);
        r.q.base = (size_t)num("reused_tokens", 0);
        r.q.gen = (size_t)std::max(1.0, num("generated_tokens", 1));
        if (!r.q.prompt || (num("seq", -1) == 0 && r.q.prompt == 17 && r.q.gen == 151)) continue;
        r.arrive = arrive;
        r.end = utc_ms(text("finish_utc"));
        r.given = (long)num("pred", -2);
        r.q.think_ms = num("think_ms", 0);
        rows.push_back(r);
    }
    require(!rows.empty(), "the trace " + path + " holds no request with an arrival time");
    std::stable_sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.arrive < b.arrive; });
    Load load{"trace", {}};
    const double t0 = rows.front().arrive;
    for (size_t i = 0; i < rows.size(); ++i) {
        LoadRequest q = rows[i].q;
        q.arrive_ms = rows[i].arrive - t0;
        if (rows[i].given >= -1) {
            q.pred = rows[i].given;
        } else {
            double best_key = 1e300;
            for (size_t j = 0; j < i; ++j) {
                const Row& y = rows[j];
                if (y.end < 0 || y.end > rows[i].arrive + 1000) continue;
                const double gap = rows[i].arrive - y.end;
                const bool reuse = q.base > 0 && y.q.prompt <= q.prompt && 2 * q.base >= y.q.prompt;
                if (!reuse && gap > 3000) continue;
                const double key = (reuse ? 0 : 1e12) + gap;
                if (key < best_key) {
                    best_key = key;
                    q.pred = (long)j;
                    q.think_ms = std::max(0.0, gap);
                }
            }
        }
        load.requests.push_back(q);
    }
    return load;
}

// The synthetic loads S1 to S7 (docs/SERVER.md, Prompt shares), `hours` long where they run on, their turns drawn from `pool` where it has any.
std::vector<Load> synthetic(double hours, const std::vector<LoadRequest>& pool, const Prices& prices, size_t ubatch) {
    std::mt19937 rng(1);
    std::vector<Load> loads;
    const auto turn = [&](double at) {
        LoadRequest q;
        if (!pool.empty()) {
            q = pool[rng() % pool.size()];
            q.gen = std::min<size_t>(q.gen, 4000);
            q.pred = -1;
        } else {
            q.prompt = 1000 + rng() % 7000;
            q.gen = 64 + rng() % 448;
        }
        q.arrive_ms = at;
        return q;
    };
    // A prompt's stage time alone, by which a stream's rate is set to a load.
    const auto cost = [&](const LoadRequest& q) {
        const double rows = (double)(q.prompt - std::min(q.base, q.prompt));
        return std::ceil(rows / (double)ubatch) * prices.chunk_ms + rows * (prices.row_ms + prices.row_depth_ms * ((double)q.base + rows / 2));
    };
    const auto stream = [&](Load& l, double load, double from, double until) {
        double mean = 0;
        for (int i = 0; i < 64; ++i) mean += cost(turn(0)) / 64;
        for (double t = from + std::exponential_distribution<double>(load / mean)(rng); t < until; t += std::exponential_distribution<double>(load / mean)(rng)) l.requests.push_back(turn(t));
    };
    const LoadRequest reply{0, 1000, 0, 4000};
    loads.push_back({"S1", {reply, {500, 150000, 0, 800}, {1500, 327, 0, 100}, {1500, 3334, 0, 160}}});
    loads.push_back({"S2", {reply, {0, 150000, 0, 800}, {10000, 2000, 0, 300}, {30000, 2000, 0, 300}, {50000, 2000, 0, 300}, {70000, 2000, 0, 300}}});
    loads.push_back({"S3", {{0, 150000, 0, 600}, {1000, 150000, 0, 600}}});
    for (const size_t burst : {8, 16, 32}) {
        Load l{"S4-" + std::to_string(burst), {reply, reply, reply, reply}};
        for (size_t i = 0; i < burst; ++i) l.requests.push_back({5000, 1000 + rng() % 7000, 0, 64 + rng() % 448});
        loads.push_back(l);
    }
    for (const double load : {0.7, 0.9}) {
        Load l{"S5-" + std::to_string((int)(load * 100)), {{60000, 150000, 0, 800}}};
        stream(l, load, 0, hours * 3600000);
        loads.push_back(l);
    }
    Load six{"S6", {{300000, 150000, 0, 800}}};
    stream(six, 0.5, 0, hours * 3600000);
    loads.push_back(six);
    for (const size_t users : {1, 4, 16, 64}) {
        Load l{"S7-" + std::to_string(users), {}};
        for (size_t u = 0; u < users; ++u)
            for (size_t k = 0; k < 8; ++k) l.requests.push_back({0, 1024, 0, 64, k ? (long)(l.requests.size() - 1) : -1, 0});
        loads.push_back(l);
    }
    return loads;
}

// The replay of loads under rules, a line a load and rule (llmx-server-passes-test --replay FILE or --load NAME): see docs/SERVER.md, Prompt shares.
int replay_main(int argc, char** argv) {
    Bench b;
    Service rule;
    std::string trace_path, only_load, only_rules;
    double hours = 1, row_scale = 1, fixed_scale = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            return argv[++i];
        };
        if (a == "--replay") trace_path = value();
        else if (a == "--load") only_load = value();
        else if (a == "--rules") only_rules = value();
        else if (a == "--hours") hours = std::stod(value());
        else if (a == "--stages") b.stages = std::stoul(value());
        else if (a == "--passes") b.passes = std::stoul(value());
        else if (a == "--max-seqs") b.max_seqs = std::stoul(value());
        else if (a == "--ubatch") b.ubatch = std::stoul(value());
        else if (a == "--pool") b.pool_tokens = std::stoul(value());
        else if (a == "--mix") rule.mix = std::stod(value());
        else if (a == "--columns") rule.columns = std::stoul(value());
        else if (a == "--row-scale") row_scale = std::stod(value());
        else if (a == "--fixed-scale") fixed_scale = std::stod(value());
        else if (a == "--class-ms") rule.prices.class_ms = std::stod(value());
        else throw std::runtime_error("unknown argument " + a);
    }
    rule.prices.row_ms *= row_scale;
    rule.prices.row_depth_ms *= row_scale;
    rule.prices.chunk_ms *= fixed_scale;
    rule.prices.reply_ms *= fixed_scale;
    std::vector<Load> loads;
    std::vector<LoadRequest> pool;
    if (!trace_path.empty()) {
        loads.push_back(trace(trace_path));
        pool = loads.back().requests;
    }
    for (Load& l : synthetic(hours, pool, rule.prices, b.ubatch)) loads.push_back(std::move(l));
    std::printf("replay: %zu stages, %zu passes, %zu requests at once, ubatch %zu, pool %zu tokens, mix %.2f, %zu columns; prices: reply %.2f + %.2f a row ms, chunk %.2f ms, row %.3f + %.3g a token of depth ms, class %.2f ms\n",
                b.stages, b.passes, b.max_seqs, b.ubatch, b.pool_tokens, rule.mix, rule.columns, rule.prices.reply_ms, rule.prices.reply_row_ms, rule.prices.chunk_ms, rule.prices.row_ms,
                rule.prices.row_depth_ms, rule.prices.class_ms);
    for (const Load& load : loads) {
        if (!only_load.empty() && load.name.rfind(only_load, 0) != 0) continue;
        // Each prompt alone on the same placement, once, for the slowdowns.
        std::map<std::pair<size_t, size_t>, double> alone;
        const auto alone_ms = [&](size_t prompt, size_t base) {
            const auto key = std::make_pair(prompt, base);
            const auto it = alone.find(key);
            if (it != alone.end()) return it->second;
            Service one = rule;
            one.share = server::Share{};
            const Outcome o = replay(Load{"alone", {{0, prompt, base, 1}}}, b, one);
            const double ms = o.done.empty() ? 0.0 : o.done[0].first_at - o.done[0].arrived;
            alone[key] = ms;
            return ms;
        };
        std::printf("load %s: %zu requests\n", load.name.c_str(), load.requests.size());
        for (const Candidate& c : candidates(b.ubatch)) {
            if (!only_rules.empty() && ("," + only_rules + ",").find("," + c.name.substr(0, 2) + ",") == std::string::npos && ("," + only_rules + ",").find("," + c.name + ",") == std::string::npos) continue;
            Service r = rule;
            r.share = c.share;
            const Outcome o = replay(load, b, r);
            const Figures f = figures(o, alone_ms);
            std::printf("  %s %s: score %.2f, worst wait %.1f s, longest %zu rows %.1f s (alone %.1f s), gaps p50/p99 %.1f/%.1f ms, %.0f prompt and %.1f output tok/s, %.2f chunk passes a 1k rows%s%s\n", load.name.c_str(), c.name.c_str(),
                        f.score, f.worst_wait_s, f.longest, f.longest_s, f.longest_alone_s, f.gap_p50_ms, f.gap_p99_ms, f.prompt_rate, f.output_rate, f.chunks_per_k, f.classes.c_str(),
                        f.skipped ? (", " + std::to_string(f.skipped) + " skipped as too long").c_str() : "");
            std::fflush(stdout);
        }
    }
    return 0;
}

// The owner's case at 20k tokens under three rules: every request ends, and least remaining reads the short prompts sooner than the oldest first.
void replay_smoke() {
    Bench b;
    b.max_seqs = 4;
    b.pool_tokens = 65536;
    Service rule;
    const Load load{"smoke", {{0, 1000, 0, 400}, {500, 20000, 0, 50}, {1500, 327, 0, 20}, {1500, 3334, 0, 20}}};
    double ttft[3] = {0, 0, 0};
    const server::Share rules[] = {{server::Share::Rule::today}, {server::Share::Rule::oldest_first}, {server::Share::Rule::least_remaining}};
    for (size_t i = 0; i < 3; ++i) {
        Service r = rule;
        r.share = rules[i];
        r.share.short_rows = b.ubatch / server::kShortShare;
        const Outcome o = replay(load, b, r);
        require(o.done.size() == load.requests.size(), "a replay left requests that never ended");
        for (const Sim::Req& q : o.done)
            if (q.prompt == 327 || q.prompt == 3334) ttft[i] = std::max(ttft[i], q.first_at - q.arrived);
        require(!o.gaps.empty() && o.chunk_passes > 0, "a replay measured no gap or formed no chunk pass");
    }
    require(ttft[2] < ttft[1], "least remaining did not read the short prompts before the long one: " + std::to_string(ttft[2]) + " ms against " + std::to_string(ttft[1]) + " ms oldest first");
    std::printf("server-passes: replay of the owner's case at 20k tokens, the later prompts' first token at %.0f ms today, %.0f ms oldest first and %.0f ms least remaining\n", ttft[0], ttft[1], ttft[2]);
}

} // namespace

// Why a turn reused little (reuse_miss): the closest held history by tokens, the message the difference falls in, and what limited the reuse below the agreement.
void reuse_miss_by_hand() {
    using server::ReuseMiss;
    const size_t block = 8;
    std::vector<uint32_t> prompt(60);
    for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = uint32_t(100 + i);
    const server::MessageSpans spans{{10, 30, 50, 60}, {"system", "user", "assistant", "user"}};
    const auto full = [](size_t n) { return n; };
    // The first message differs: the history agrees for 2 tokens, inside the first block, in message 1, the system message.
    std::vector<uint32_t> head = prompt;
    head[2] = 7;
    std::vector<uint32_t> other(40, 9);
    ReuseMiss m = server::reuse_miss(prompt, 0, {{&other, full, nullptr}, {&head, full, nullptr}}, block, spans);
    require(m.prompt == 60 && m.reused == 0 && m.agreed == 2 && m.limit == ReuseMiss::Limit::tokens && m.at == 2 && m.messages == 4 && m.message == 0 && m.role == "system",
            "a first message that differs is not named as the history agreeing for 2 tokens in message 1, the system message");
    // The last message differs at 45, in message 3, the assistant's; on a model that keeps a state the history's only state, at 48, lies past it, so nothing could be forked.
    std::vector<uint32_t> tail = prompt;
    tail[45] = 7;
    const auto state48 = [](size_t n) { return n >= 48 ? size_t(48) : size_t(0); };
    m = server::reuse_miss(prompt, 0, {{&tail, full, state48}}, block, spans);
    require(m.agreed == 45 && m.limit == ReuseMiss::Limit::state && m.at == 0 && m.message == 2 && m.role == "assistant",
            "a last message that differs past the history's only state is not named as the state limiting the reuse in message 3");
    // The same history with a state at 40: the 40 rows below it were forkable, so a turn that reused them is limited by the tokens, and one that reused none by the room.
    const auto state40 = [](size_t n) { return n >= 40 ? size_t(40) : size_t(0); };
    m = server::reuse_miss(prompt, 40, {{&tail, full, state40}}, block, spans);
    require(m.limit == ReuseMiss::Limit::tokens && m.at == 45, "a turn that forked every row below its difference is not named as limited by the tokens");
    m = server::reuse_miss(prompt, 0, {{&tail, full, state40}}, block, spans);
    require(m.limit == ReuseMiss::Limit::room && m.at == 40, "a history the turn did not fork though it held 40 rows is not named as limited by the room");
    // A completions prompt, no messages: the tokens agree whole, but the row classes only for 20, so no message is named and the classes limit it at 20.
    std::vector<uint32_t> same = prompt;
    m = server::reuse_miss(prompt, 16, {{&same, [](size_t n) { return std::min<size_t>(n, 20); }, nullptr}}, block, {});
    require(m.agreed == 60 && m.limit == ReuseMiss::Limit::classes && m.at == 20 && m.messages == 0 && m.role.empty(),
            "a completions prompt whose rows were computed in other classes from 20 is not named so, or names a message");
}

int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string(argv[1]).rfind("--", 0) == 0) return replay_main(argc, argv);
        const size_t schedules = argc > 1 ? (size_t)std::stoul(argv[1]) : 2000;
        rooms_by_hand();
        host_cache_by_hand();
        turns_line_by_hand();
        grow_step_by_hand();
        growth_by_hand();
        rounds_by_hand();
        rows_by_hand();
        automatic_by_hand();
        due_step_first();
        service_by_hand();
        replay_smoke();
        reuse_miss_by_hand();
        random_schedules(schedules);
        std::cout << "server-passes: " << checks << " checks pass\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "server-passes: " << e.what() << '\n';
        return 1;
    }
}
