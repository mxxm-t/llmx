// The cache and context plan's scenario (docs/CACHE-PLAN.md, section 5): one scripted conversation at production's shape through the scheduler, each turn's rows classified against a ledger the test keeps itself.
// Counts only, no clock; an expectation the plan has not closed yet is recorded as broken with its narrow witness, and one that improves unexpectedly fails until its entry is updated.
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <limits>
#include <iostream>
#include <mutex>
#include <thread>

#include "server/context_cut.hpp"
#include "server_harness.hpp"

namespace {

using Stats = server::Scheduler::Stats;
using Handle = std::shared_ptr<server::Request>;

std::vector<uint32_t> ids_of(const Reply& r) {
    std::vector<uint32_t> ids;
    for (const auto& t : r) ids.push_back(t.id);
    return ids;
}

// One turn's line of the report: its logical rows by kind, which add up to the prompt, and what the server counted while it ran.
struct Row {
    std::string leg;
    size_t turn = 0, prompt = 0, boundary = 0, expected = 0, reused = 0, new_work = 0, evicted = 0, failed_restore = 0, unexplained = 0;
    size_t gave_way = 0, handed = 0, consumed = 0;
};

// A loss the server reported: its kind, the rows [first, tokens) or the state at `tokens`, and the history they were of.
struct Loss {
    server::Move kind;
    size_t first, tokens;
    std::vector<uint32_t> history;
};

// The rows of a prompt's shortfall [reused, expected) that the losses reported from `since` on explain, one mark a row.
// A loss counts only where its history is the prompt's own over the rows they share, its rows or its state lie in the shortfall, and it came after the point was last prepared.
std::vector<char> lost_rows(const std::vector<uint32_t>& prompt, size_t reused, size_t expected, const std::vector<Loss>& losses, size_t since = 0) {
    std::vector<char> covered(expected > reused ? expected - reused : 0, 0);
    for (size_t i = since; i < losses.size(); ++i) {
        const Loss& l = losses[i];
        // A loss of a longer history is compared over the rows of it the prompt holds up to what was expected.
        const size_t n = std::min({l.tokens, l.history.size(), expected, prompt.size()});
        if (!std::equal(l.history.begin(), l.history.begin() + (std::ptrdiff_t)n, prompt.begin())) continue;
        const bool state = l.kind == server::Move::boundary_dropped || l.kind == server::Move::boundary_lost;
        // A state lost at a position leaves the rows up to it without a restore point, and needs its history known up to there; rows lost are the rows themselves, as far as their history is known.
        const size_t from = state ? reused : std::max(l.first, reused), to = std::min({l.tokens, expected, n});
        if (state && (l.tokens <= reused || l.tokens > expected || n < l.tokens)) continue;
        for (size_t k = from; k < to; ++k) covered[k - reused] = 1;
    }
    return covered;
}

// How many of those rows the losses explain.
size_t explained(const std::vector<uint32_t>& prompt, size_t reused, size_t expected, const std::vector<Loss>& losses, size_t since = 0) {
    const std::vector<char> covered = lost_rows(prompt, reused, expected, losses, since);
    return (size_t)std::count(covered.begin(), covered.end(), (char)1);
}

// The rows of a turn's shortfall [reused, expected) that its own read covered where that read reached its bound (Request::restore_late), up to the read's end, less the rows a loss already explains (`lost`, lost_rows).
size_t failed_rows(const std::vector<char>& lost, size_t reused, size_t expected, size_t late) {
    size_t n = 0;
    for (size_t k = reused; k < std::min(late, expected); ++k) n += !lost[k - reused];
    return n;
}

// The classifier held to its own rule on made-up losses: the right loss explains a shortfall, and another history's, other rows' or none does not.
void attribution() {
    // A failed restore is the turn's own read's: a timeout of another request's read explains none of its rows, and its own read explains only the rows it covered.
    const std::vector<uint32_t> mine = prompt_of(3, 1200, 64), theirs = prompt_of(4, 1200, 64);
    const Loss right{server::Move::job_lost, 0, 896, mine}, other{server::Move::job_lost, 0, 896, theirs}, elsewhere{server::Move::consumed_lost, 896, 1152, mine}, state{server::Move::boundary_dropped, 768, 768, mine};
    const std::vector<char> none(384, 0);
    require(failed_rows(none, 0, 384, 0) == 0, "attribution: another request's read that reached its bound explains a turn's rows");
    require(failed_rows(none, 0, 384, 128) == 128 && failed_rows(none, 0, 384, 384) == 384 && failed_rows(none, 256, 640, 512) == 256, "attribution: a read that reached its bound did not explain exactly the rows it covered");
    // Rows a loss explains are not the read's too: with [0, 128) lost and the read reaching its bound at 256, 128 rows are evicted, 128 a failed restore and 128 unexplained.
    const Loss head{server::Move::job_lost, 0, 128, mine};
    const std::vector<char> lost = lost_rows(mine, 0, 384, {head});
    const size_t evicted = explained(mine, 0, 384, {head}), failed = failed_rows(lost, 0, 384, 256);
    require(evicted == 128 && failed == 128 && 384 - evicted - failed == 128, "attribution: a read that reached its bound explained rows a loss already explains: " + std::to_string(evicted) + " evicted, " + std::to_string(failed) + " failed");
    require(failed_rows(lost_rows(mine, 0, 384, {Loss{server::Move::job_lost, 100, 384, mine}}), 0, 384, 384) == 100, "attribution: a read past the rows a later loss explains claimed them");
    require(explained(mine, 0, 896, {right}) == 896, "attribution: a history's own loss of the rows explains them");
    require(explained(mine, 0, 896, {other}) == 0, "attribution: another history's loss explains nothing");
    require(explained(mine, 0, 896, {elsewhere}) == 0, "attribution: a loss of other rows explains nothing");
    require(explained(mine, 0, 896, {}) == 0, "attribution: no loss explains nothing");
    require(explained(mine, 0, 896, {other, elsewhere, right}) == 896 && explained(theirs, 0, 896, {right, elsewhere}) == 0, "attribution: two histories in one span are told apart");
    require(explained(mine, 640, 768, {state}) == 128 && explained(mine, 768, 896, {state}) == 0 && explained(theirs, 640, 768, {state}) == 0, "attribution: a state lost at a position explains the rows up to it, of its history alone");
    // A request shorter than the history that was lost: its rows are in the lost range, so they are explained; one that parts from that history inside them is not.
    std::vector<uint32_t> shorter(mine.begin(), mine.begin() + 640), parted = shorter;
    parted[300] ^= 1;
    require(explained(shorter, 0, 512, {right}) == 512, "attribution: a longer history's loss explains a shorter request's rows inside it");
    require(explained(parted, 0, 512, {right}) == 0, "attribution: a request parting from the lost history inside its rows is not explained");
    // A loss whose history is known only in part explains only the rows it identifies, and a state no row of its position.
    const Loss truncated{server::Move::job_lost, 0, 896, std::vector<uint32_t>(mine.begin(), mine.begin() + 300)},
        state_short{server::Move::boundary_dropped, 768, 768, std::vector<uint32_t>(mine.begin(), mine.begin() + 300)};
    require(explained(mine, 0, 896, {truncated}) == 300 && explained(mine, 640, 768, {state_short}) == 0, "attribution: a loss explains no row past the history it carries");
    // A loss reported before the restore point was prepared again explains nothing of a later shortfall.
    require(explained(mine, 0, 896, {right}, 1) == 0 && explained(mine, 0, 896, {right, other}, 1) == 0 && explained(mine, 0, 896, {other, right}, 1) == 896, "attribution: a loss older than the point's last preparation explains nothing");
}

// A conversation as the test knows it: its history, and the longest restore point the script has seen prepared for its next turn.
struct Conversation {
    std::vector<uint32_t> history;
    size_t prepared = 0;
    // How many losses had been reported when that point was last prepared: only a later one can explain falling short of it.
    size_t since = 0;
    uint32_t seed = 0;
    Handle last;
    std::vector<std::pair<Req, Reply>> said;
};

// An expectation of the plan: its promise, whether it holds today, and what the scenario saw.
struct Expectation {
    std::string name, promise;
    bool held_today, seen;
};

// A turn adds one block, its reply crossing the block's end, so every read-ahead job has rows to read and a pass to be met at.
// What a placement shows today where the placements differ: what the two edits reused.
struct Witness {
    size_t edit_reused, older_reused, after_edit;
    // Whether a state is kept at a message boundary at all: over two devices the host tier's whole slabs, one a device, hold none, so below a history's newest point nothing was ever prepared there.
    bool boundaries;
};

constexpr size_t kPool = 2048, kUser = 88, kReply = 40, kOther = 300, kOtherReply = 30;

// The test's view of a running scheduler: the handles it gave out, the arrivals it submits from the scheduler's thread at a job's pass or at another request's, and every loss reported with its history.
struct Watch {
    server::Scheduler& sched;
    std::mutex handles;
    std::vector<Handle> known;
    std::function<void()> at_job_pass, at_that_pass;
    const server::Request* at_pass_of = nullptr;
    std::mutex losing;
    std::vector<Loss> losses;
    explicit Watch(server::Scheduler& s) : sched(s) {
        // A pass holding a request no handle of the test's names is a job's: the script's arrivals are submitted from here, on the scheduler's thread.
        sched.on_retire = [this](const server::Scheduler::Retired& t) {
            std::function<void()> act;
            {
                std::lock_guard<std::mutex> lk(handles);
                for (const server::Request* r : t.requests)
                    if (at_job_pass && std::none_of(known.begin(), known.end(), [&](const Handle& h) { return h.get() == r; })) act = std::move(at_job_pass);
                if (act) at_job_pass = nullptr;
            }
            if (act) {
                act();
                return;
            }
            {
                std::lock_guard<std::mutex> lk(handles);
                if (at_that_pass && std::find(t.requests.begin(), t.requests.end(), at_pass_of) != t.requests.end()) act = std::move(at_that_pass);
                if (act) at_that_pass = nullptr;
            }
            if (act) act();
        };
        sched.on_move = [this](const server::Scheduler::Moved& m) {
            std::lock_guard<std::mutex> lk(losing);
            losses.push_back(Loss{m.kind, m.first, m.tokens, std::vector<uint32_t>(m.history->begin(), m.history->begin() + (std::ptrdiff_t)std::min(m.tokens, m.history->size()))});
        };
    }
    // A cut request names the uncut prompt through its first dropped row (`cut`), as the routes do (Api::fit).
    Handle submit(const Req& r, size_t message = std::numeric_limits<size_t>::max(), std::vector<uint32_t> cut = {}) {
        std::lock_guard<std::mutex> lk(handles);
        known.push_back(sched.submit(r.prompt, params_of(r), std::numeric_limits<size_t>::max(), message, server::Request::Clock::now(), std::move(cut)));
        return known.back();
    }
    size_t lost_so_far() {
        std::lock_guard<std::mutex> lk(losing);
        return losses.size();
    }
};

// The report a script keeps: each turn's row, and the uncut conversation's message ends and which of them open a user's message, the leading 400 rows its system message; `dropped` is what the cuts so far took.
struct Ledger {
    std::string on_what;
    bool state = false;
    std::vector<Row> report;
    std::vector<size_t> ends{400};
    std::vector<char> starts{0};
    size_t dropped = 0;

    // A turn: its row of the report against what the script saw prepared, the chat's message ends where it is the chat's, and the conversation's history and its own donor as the next turn's restore point.
    void classify(Watch& w, Conversation& c, bool chat, const Req& r, const Handle& h, const Reply& reply, const std::string& leg, const Stats& before) {
        const Stats after = w.sched.stats();
        Row row;
        row.leg = leg;
        row.turn = report.size() + 1;
        row.prompt = r.prompt.size();
        row.boundary = c.history.size();
        row.expected = c.prepared;
        row.reused = h->reused();
        row.gave_way = after.reprefill_cancels - before.reprefill_cancels;
        row.handed = after.reprefill_handed - before.reprefill_handed;
        row.consumed = after.moves[(size_t)server::Move::consumed] - before.moves[(size_t)server::Move::consumed];
        const size_t shortfall = row.expected > row.reused ? row.expected - row.reused : 0;
        std::vector<char> lost;
        {
            std::lock_guard<std::mutex> lk(w.losing);
            lost = lost_rows(r.prompt, row.reused, row.expected, w.losses, c.since);
        }
        row.evicted = (size_t)std::count(lost.begin(), lost.end(), (char)1);
        // The turn's own read that reached its bound before it ended leaves the rows it covered to be computed, a failed restore (P2), as a slow runner meets where the tiny model computes faster than the disk reads.
        row.failed_restore = failed_rows(lost, row.reused, row.expected, h->restore_late());
        row.unexplained = shortfall - row.evicted - row.failed_restore;
        row.new_work = row.prompt - row.reused - shortfall;
        c.said.push_back({r, reply});
        if (chat) {
            ends.push_back(dropped + r.prompt.size());
            starts.push_back(1);
            ends.push_back(dropped + r.prompt.size() + reply.size());
            starts.push_back(0);
        }
        c.history = r.prompt;
        for (uint32_t id : ids_of(reply)) c.history.push_back(id);
        // The request's own donor: its prompt's whole blocks, or on a model that keeps a state the block its state was kept at.
        c.prepared = (state ? r.prompt.size() - 1 : r.prompt.size()) / kBlock * kBlock;
        c.since = w.lost_so_far();
        c.last = h;
        report.push_back(row);
        std::cout << "cache-scenario" << on_what << ": turn " << row.turn << " (" << leg << "): prompt " << row.prompt << ", boundary " << row.boundary << ", expected " << row.expected << ", reused " << row.reused
                  << ", new work " << row.new_work << ", evicted " << row.evicted << ", failed restore " << row.failed_restore << ", unexplained " << row.unexplained << "; jobs given way " << row.gave_way << ", handed "
                  << row.handed << ", donors consumed " << row.consumed << "\n";
    }

    // A prompt of the chat that passes `room`, cut by the product's rule at the message ends the script knows: the system message kept and the rows from the first kept message on; the row of the uncut conversation the kept part starts at.
    size_t cut(Req& r, size_t room) {
        const size_t n = dropped + r.prompt.size();
        std::vector<size_t> e = ends;
        std::vector<char> st = starts;
        e.push_back(n);
        st.push_back(1);
        server::MessageEnds cut_ends(e);
        const size_t k = server::first_kept(cut_ends, st, 1, n, room, kPool / server::kCutTo);
        require(k < e.size(), "cache-scenario" + on_what + ": a prompt of " + std::to_string(n) + " rows has no cut");
        const size_t from = e[k - 1] - dropped;
        r.prompt.erase(r.prompt.begin() + 400, r.prompt.begin() + (std::ptrdiff_t)from);
        dropped = e[k - 1] - 400;
        return e[k - 1];
    }
};

void scenario(const Make& make, const bpe::Tokenizer& tok, uint32_t v, const std::string& on_what, const Witness& w) {
    const size_t bt = kBlock;
    auto model = make(kPool, 64);
    const bool state = model->keeps_state();
    server::Scheduler sched(*model, tok, 4, 64, 0, false, (size_t)1 << 30);
    Watch watch(sched);
    auto& handles = watch.handles;
    auto& known = watch.known;
    auto& at_job_pass = watch.at_job_pass;
    auto& at_that_pass = watch.at_that_pass;
    auto& at_pass_of = watch.at_pass_of;
    std::thread runner([&] { sched.run(); });
    Ledger book;
    book.on_what = on_what;
    book.state = state;
    std::vector<Row>& report = book.report;
    Conversation chat;
    chat.seed = 7;
    chat.history = prompt_of(5, 400, v);
    uint32_t other_seed = 33;
    size_t cuts = 0, since1 = 0, since3 = 0;
    std::vector<uint32_t> after3, after1;
    std::vector<Expectation> table;
    const auto submit = [&](const Req& r, size_t message, std::vector<uint32_t> cut = {}) { return watch.submit(r, message, std::move(cut)); };
    const auto lost_so_far = [&] { return watch.lost_so_far(); };
    const auto classify = [&](Conversation& c, const Req& r, const Handle& h, const Reply& reply, const std::string& leg, const Stats& before) { book.classify(watch, c, &c == &chat, r, h, reply, leg, before); };
    const auto next_of = [&](Conversation& c) {
        Req r{c.history, (int)kReply};
        const std::vector<uint32_t> user = prompt_of(c.seed++ % v, kUser, v);
        r.prompt.insert(r.prompt.end(), user.begin(), user.end());
        return r;
    };
    try {
        // Leg A, below half the pool: each turn, its read-ahead job, and another conversation's request that fits beside the job.
        for (size_t k = 0; k < 4; ++k) {
            const Stats before = sched.stats();
            const Req r = next_of(chat);
            const Handle h = submit(r, chat.history.size());
            const Reply reply = drain(*h);
            classify(chat, r, h, reply, "A", before);
            if (k == 2) after3 = chat.history;
            if (k == 0) after1 = chat.history;
            const size_t jobs = sched.stats().reprefills;
            Handle other;
            const Req o{prompt_of(other_seed++ % v, kOther, v), (int)kOtherReply};
            {
                std::lock_guard<std::mutex> lk(handles);
                at_job_pass = [&] {
                    std::lock_guard<std::mutex> lk2(handles);
                    other = sched.submit(o.prompt, params_of(o));
                };
            }
            sched.follow(h, chat.history, true);
            within_a_minute([&] { return sched.stats().reprefills == jobs + 1; }, "leg A: the job done");
            {
                std::lock_guard<std::mutex> lk(handles);
                require(other != nullptr, "leg A: the other request arrived while the job ran");
                known.push_back(other);
            }
            drain(*other);
            chat.prepared = chat.history.size() / bt * bt;
            chat.since = lost_so_far();
            // The points the two edits will need are prepared here, by the jobs after turns 1 and 3.
            if (k == 0) since1 = chat.since;
            if (k == 2) since3 = chat.since;
        }
        // Below half the pool no turn lost a row, but jobs gave way to requests that fit once a donor is evicted, and read their rows again.
        const size_t a_gave_way = sched.stats().reprefill_cancels;
        table.push_back({"a job gives way only to a request that cannot be admitted beside it", "P7", true, !a_gave_way});
        // Leg B, past half the pool: the turn's donor and its job cannot both be reserved, so the job holds the chat's rows alone.
        // Another conversation's request that does not fit beside the job arrives at the job's pass, and the chat's next turn while that request runs.
        Stats mark = sched.stats();
        Handle h = submit(next_of(chat), chat.history.size());
        for (size_t k = 0; k < 2; ++k) {
            Req r{chat.history, (int)kReply};
            const std::vector<uint32_t> user = prompt_of((chat.seed - 1) % v, kUser, v);
            r.prompt.insert(r.prompt.end(), user.begin(), user.end());
            const Reply reply = drain(*h);
            classify(chat, r, h, reply, "B", mark);
            if (k == 1) break;
            mark = sched.stats();
            Handle other, after;
            const Req o{prompt_of(other_seed++ % v, 1100, v), (int)kOtherReply}, t = next_of(chat);
            const size_t message = chat.history.size();
            {
                std::lock_guard<std::mutex> lk(handles);
                at_job_pass = [&, o] {
                    std::lock_guard<std::mutex> lk2(handles);
                    other = sched.submit(o.prompt, params_of(o));
                    known.push_back(other);
                    at_pass_of = other.get();
                    at_that_pass = [&, t, message] {
                        std::lock_guard<std::mutex> lk3(handles);
                        after = sched.submit(t.prompt, params_of(t), std::numeric_limits<size_t>::max(), message);
                        known.push_back(after);
                    };
                };
            }
            sched.follow(h, chat.history, true);
            within_a_minute([&] { std::lock_guard<std::mutex> lk(handles); return after != nullptr; }, "leg B: the arrivals at the job's pass and at the other request's");
            drain(*other);
            h = after;
        }
        // Quiescence: the job that gave way has finished or been dropped as stale, so what follows meets no job; then one plain turn.
        within_a_minute([&] { return !sched.stats().jobs; }, "leg B: no job left");
        if (chat.history.size() / bt * bt - bt > chat.prepared) {
            chat.prepared = chat.history.size() / bt * bt - bt;
            chat.since = lost_so_far();
        }
        {
            const Stats before = sched.stats();
            const Req r = next_of(chat);
            const Handle plain = submit(r, chat.history.size());
            const Reply reply = drain(*plain);
            classify(chat, r, plain, reply, "B, settled", before);
        }
        const size_t b_evicted = report[5].evicted, b_gave_way = report[5].gave_way;
        table.push_back({"past half the pool a turn's donor stays beside its job", "P3", true, !report[5].consumed});
        table.push_back({"a job that gives way to another conversation's request loses no row a later turn needs", "P1, P7", true, !b_evicted});
        // The job's rows kept as a donor when it gives way (S8), so the next turn forks them while the other request runs.
        require(b_gave_way == 1 && b_evicted == 0 && report[5].reused == 896 && !sched.stats().moves[(size_t)server::Move::job_lost],
                "leg B's witness: " + std::to_string(b_gave_way) + " jobs gave way, " + std::to_string(report[5].reused) + " rows reused, " + std::to_string(b_evicted) + " evicted and " +
                std::to_string(sched.stats().moves[(size_t)server::Move::job_lost]) + " job losses, against 1, 896, 0 and 0");
        // Leg C, an edit: turn 4's user message replaced, so the prompt agrees with the chat through turn 3's reply.
        {
            Conversation edit;
            edit.history = after3;
            edit.seed = 60;
            edit.prepared = w.boundaries ? after3.size() / bt * bt : 0;
            edit.since = since3;
            const Stats before = sched.stats();
            const Req r = next_of(edit);
            const Handle e = submit(r, after3.size());
            const Reply reply = drain(*e);
            classify(edit, r, e, reply, "C", before);
            chat.said.push_back(edit.said.back());
            // On a model that keeps a state, the state at that message was thinned out of host memory (a fixed number of boundaries a conversation, with no disk tier), and the next one down serves.
            table.push_back({"an edit reuses up to the last unchanged message", "P10", w.boundaries, report.back().reused == after3.size() / bt * bt});
            require(report.back().reused == w.edit_reused && report.back().reused + report.back().evicted == (w.boundaries ? 768u : 0u), "leg C's witness: reused " + std::to_string(report.back().reused) + " and evicted " + std::to_string(report.back().evicted) + ", against " + std::to_string(w.edit_reused) + " and the rest of what was prepared");
        }
        // The same at an older message, turn 2's user message replaced.
        {
            Conversation edit;
            edit.history = after1;
            edit.seed = 61;
            edit.prepared = w.boundaries ? after1.size() / bt * bt : 0;
            edit.since = since1;
            const Stats before = sched.stats();
            const Req r = next_of(edit);
            const Handle e = submit(r, after1.size());
            const Reply reply = drain(*e);
            classify(edit, r, e, reply, "C, older", before);
            chat.said.push_back(edit.said.back());
            require(report.back().reused == w.older_reused, "leg C's older edit: reused " + std::to_string(report.back().reused) + ", against " + std::to_string(w.older_reused));
        }
        // Leg E, two cuts: the conversation grows past what a prompt may take and is cut by the product's rule at the message ends the script knows, twice, with two turns after each.
        const size_t room = sched.prompt_room(params_of(Req{{}, (int)kReply})), where[] = {1424, 2448};
        bool same_place = true, reused_after = true;
        size_t reread = 0;
        const size_t first_after_edit = report.size();
        for (size_t after_cut = 0; cuts < 2 || after_cut < 2;) {
            const Stats before = sched.stats();
            Req r = next_of(chat);
            bool cut = false;
            std::vector<uint32_t> dropped;
            if (r.prompt.size() > room) {
                const std::vector<uint32_t> uncut = r.prompt;
                const size_t at = book.cut(r, room);
                size_t same = 0;
                while (same < r.prompt.size() && r.prompt[same] == uncut[same]) ++same;
                dropped.assign(uncut.begin(), uncut.begin() + (std::ptrdiff_t)same + 1);
                require(cuts < 2 && at == where[cuts], "leg E: cut " + std::to_string(cuts + 1) + " falls at row " + std::to_string(at) + " of the conversation, against " + std::to_string(cuts < 2 ? where[cuts] : 0));
                chat.prepared = w.boundaries ? 400 / bt * bt : 0;
                chat.since = 0;
                ++cuts;
                after_cut = 0;
                cut = true;
            } else if (cuts) {
                // The turn after a cut is cut at the same place: its prompt is the one before it and more.
                same_place = same_place && book.dropped == where[cuts - 1] - 400;
                ++after_cut;
            }
            const Handle h = submit(r, r.prompt.size() - kUser, dropped);
            const Reply reply = drain(*h);
            classify(chat, r, h, reply, cut ? "E, cut" : "E", before);
            if (cut) reread += report.back().new_work;
            else if (cuts) reused_after = reused_after && report.back().reused == report.back().expected;
        }
        require(cuts == 2, "leg E: " + std::to_string(cuts) + " cuts, against 2");
        // The chat's turn after the edit: the edit forked the chat's donor below its restore point and consumed it, so the rows past the fork are computed again.
        const Row& back = report[first_after_edit];
        // Where the edit reused nothing it consumed nothing either, and the chat's donor is whole.
        table.push_back({"a donor consumed by its fork loses no row past the fork point", "P1, P3", w.after_edit == 1152, !back.evicted});
        require(back.expected == 1152 && back.reused == w.after_edit && back.reused + back.evicted == 1152,
                "the witness after the edit: expected " + std::to_string(back.expected) + ", reused " + std::to_string(back.reused) + ", evicted " + std::to_string(back.evicted) + ", against 1152, " + std::to_string(w.after_edit) + " and the rest");
        table.push_back({"the cut falls at the same message turn after turn", "P5", true, same_place});
        table.push_back({"the turns after a cut reuse their history", "P5", true, reused_after});
        // A cut prompt is 1000 rows, its system message's 384 whole-block rows reused where a state is kept at a boundary: 616 rows read again a cut, and all 1000 where none is.
        table.push_back({"a cut reuses its leading message's rows", "P5", w.boundaries, reread == 2 * 616});
        require(reread == 2 * (w.boundaries ? 616u : 1000u), "leg E: the two cuts read " + std::to_string(reread) + " rows again, against " + (w.boundaries ? "616" : "1000") + " each");
    } catch (...) {
        sched.stop();
        runner.join();
        throw;
    }
    sched.stop();
    runner.join();
    // Identity: every turn's reply is its reply on a fresh model.
    for (size_t i = 0; i < chat.said.size(); ++i) {
        auto fresh = make(kPool, 64);
        same(serve(*fresh, tok, 4, {{chat.said[i].first}})[0], chat.said[i].second, "cache-scenario" + on_what + ": turn " + std::to_string(i + 1) + "'s reply");
    }
    size_t unexplained = 0;
    for (const Row& r : report) {
        require(r.reused + r.new_work + r.evicted + r.failed_restore + r.unexplained == r.prompt, "cache-scenario" + on_what + ": turn " + std::to_string(r.turn) + "'s kinds do not add up to its prompt");
        unexplained += r.unexplained;
    }
    require(!unexplained, "cache-scenario" + on_what + ": " + std::to_string(unexplained) + " rows computed that a restore point should have held and no move the server reported explains (P8)");
    // Each expectation is what the plan records for today: one that fails, or improves without its entry being updated, fails here.
    for (const Expectation& e : table) {
        std::cout << "cache-scenario" << on_what << ": " << (e.seen ? "held" : "broken") << ": " << e.name << " (" << e.promise << ")\n";
        require(e.seen == e.held_today, "cache-scenario" + on_what + ": \"" + e.name + "\" (" + e.promise + ") is " + (e.seen ? "held" : "broken") + ", and the plan records it as " + (e.held_today ? "held" : "broken") + " today");
    }
}

// Production's proportions (docs/CACHE-PLAN.md, Production since 2026-10-11): a chat past half the pool, after every reply an unrelated request that does not fit beside the reply's job, and the next turn while it runs.
// No history fits its host tier, its disk tier is never idle, the chat is cut and the server restarted; with `late` the request comes once the job has read the reply (the 27B's shape), else at its pass.
void production(const Make& make, const bpe::Tokenizer& tok, uint32_t v, const std::string& name, bool late) {
    const std::string on_what = name + (late ? ", the request after the job" : ", the request at the job's pass");
    const size_t bt = kBlock;
    DiskRoot disk("cache-scenario");
    server::DiskOptions options = disk.options(uint64_t(1) << 30);
    options.keep = true;
    options.idle = std::chrono::hours(1);
    // A read is waited for whatever this runner computes a row in, so a slow runner computes no row a read gives; the bound itself is disk_read_bound's case in server-resume.
    options.read_wait = std::chrono::seconds(60);
    auto model = make(kPool, 64);
    const bool state = model->keeps_state();
    // A host tier a byte short of the copy of any history, which host memory takes in whole slabs: the disk is every donor's only save, as it is a long chat's on production.
    const size_t host = model->host_bytes(kBlock) - 1;
    auto sched = std::make_unique<server::Scheduler>(*model, tok, 4, 64, 0, false, host, nullptr, 0, true, options);
    auto watch = std::make_unique<Watch>(*sched);
    std::thread runner([&] { sched->run(); });
    Ledger book;
    book.on_what = on_what + ", production's proportions";
    book.state = state;
    Conversation chat;
    chat.seed = 70;
    chat.history = prompt_of(5, 400, v);
    std::vector<Expectation> table;
    std::vector<size_t> cut_turns, restart_turns;
    size_t released = 0, job_lost = 0, consumed_lost = 0, not_to_host = 0, host_lost = 0, lost_before_written = 0, written = 0, read = 0, tasks = 0;
    uint32_t other_seed = 90;
    const auto tally = [&] {
        const Stats s = sched->stats();
        released += s.reprefill_released;
        job_lost += s.moves[(size_t)server::Move::job_lost];
        consumed_lost += s.moves[(size_t)server::Move::consumed_lost];
        not_to_host += s.moves[(size_t)server::Move::not_to_host];
        host_lost += s.moves[(size_t)server::Move::host_lost];
        lost_before_written += s.lost_before_written;
        written += s.moves[(size_t)server::Move::write_whole];
        read += s.moves[(size_t)server::Move::read_done];
    };
    try {
        within_a_minute([&] { return sched->stats().disk_ready; }, "production: the disk tier made");
        // The first turn's user message takes the chat to 1256 rows, past half the pool, its reply crossing a block's end as every later turn's, each adding a block.
        Req r{chat.history, (int)kReply};
        const std::vector<uint32_t> first = prompt_of(chat.seed++ % v, 856, v);
        r.prompt.insert(r.prompt.end(), first.begin(), first.end());
        Handle h = watch->submit(r, chat.history.size());
        Stats mark = sched->stats();
        std::string leg = "P";
        constexpr size_t kTurns = 10, kRestartAfter = 7;
        const size_t room = sched->prompt_room(params_of(Req{{}, (int)kReply}));
        for (size_t k = 0;; ++k) {
            const Reply reply = drain(*h);
            book.classify(*watch, chat, true, r, h, reply, leg, mark);
            if (k + 1 == kTurns) break;
            // The next turn, cut where it passes what a prompt may take, and its restore point the system message's whole blocks after a cut.
            Req next{chat.history, (int)kReply};
            const std::vector<uint32_t> user = prompt_of(chat.seed++ % v, kUser, v);
            next.prompt.insert(next.prompt.end(), user.begin(), user.end());
            const size_t message = next.prompt.size() - kUser;
            bool cut = false;
            if (next.prompt.size() > room) {
                book.cut(next, room);
                cut = true;
            }
            // The unrelated request at the job's pass, and the next turn at that request's pass, but before the restart, where it waits for the new server.
            Handle other, after;
            const Req o{prompt_of(other_seed++ % v, 700, v), (int)kOtherReply};
            const bool restart = k + 1 == kRestartAfter;
            // The unrelated request and, at its pass, the chat's next turn, unless the server restarts first.
            const auto arrive = [&, o, next, message, restart] {
                other = sched->submit(o.prompt, params_of(o));
                watch->known.push_back(other);
                if (restart) return;
                watch->at_pass_of = other.get();
                watch->at_that_pass = [&, next, message] {
                    std::lock_guard<std::mutex> lk3(watch->handles);
                    after = sched->submit(next.prompt, params_of(next), std::numeric_limits<size_t>::max(), message);
                    watch->known.push_back(after);
                };
            };
            mark = sched->stats();
            const size_t jobs = mark.reprefills;
            if (late) {
                // The request after the job has read the reply, so its room takes the job's donor rather than the job.
                sched->follow(h, chat.history, true);
                within_a_minute([&] { return sched->stats().reprefills > jobs; }, "production: the job done");
                std::lock_guard<std::mutex> lk(watch->handles);
                arrive();
            } else {
                {
                    std::lock_guard<std::mutex> lk(watch->handles);
                    watch->at_job_pass = [&, arrive] {
                        std::lock_guard<std::mutex> lk2(watch->handles);
                        arrive();
                    };
                }
                sched->follow(h, chat.history, true);
            }
            within_a_minute([&] { std::lock_guard<std::mutex> lk(watch->handles); return other != nullptr && (restart || after != nullptr); }, "production: the arrivals of the other request and of the next turn");
            drain(*other);
            ++tasks;
            if (cut) {
                // A model that keeps a state reuses the system message only through the state at its end, which a host tier short of one slab cannot hold.
                chat.prepared = state ? 0 : 400 / bt * bt;
                chat.since = 0;
                cut_turns.push_back(book.report.size() + 1);
            }
            leg = cut ? "P, cut" : "P";
            if (restart) {
                // Once the job has read the reply, a clean stop under keep writes what memory holds; a new server on a fresh model adopts the files, and the next turn reads them.
                within_a_minute([&] { return sched->stats().reprefills > jobs; }, "production: the job before the restart done");
                tally();
                sched->stop();
                runner.join();
                std::vector<Loss> losses = std::move(watch->losses);
                watch.reset();
                sched.reset();
                model = make(kPool, 64);
                sched = std::make_unique<server::Scheduler>(*model, tok, 4, 64, 0, false, host, nullptr, 0, true, options);
                watch = std::make_unique<Watch>(*sched);
                watch->losses = std::move(losses);
                runner = std::thread([&] { sched->run(); });
                // The turn is sent as the new server starts, before its disk tier has taken in what the stop kept, which the turn waits for.
                mark = sched->stats();
                after = watch->submit(next, message);
                restart_turns.push_back(book.report.size() + 1);
                leg = "P, after a restart";
            }
            h = after;
            r = next;
        }
        tally();
        require(cut_turns.size() == 1 && restart_turns.size() == 1, "production: " + std::to_string(cut_turns.size()) + " cuts and " + std::to_string(restart_turns.size()) + " restarts, against 1 and 1");
        size_t evicted = 0, unexplained = 0;
        for (const Row& row : book.report) {
            if (row.turn != cut_turns.front() && row.turn != restart_turns.front()) evicted += row.evicted;
            unexplained += row.unexplained;
        }
        const Row &cut = book.report[cut_turns.front() - 1], &restarted = book.report[restart_turns.front() - 1];
        std::cout << "cache-scenario" << on_what << ", production's proportions: " << tasks << " unrelated requests; rows evicted " << evicted << " on the plain turns, " << cut.evicted << " on the cut turn and " << restarted.evicted
                  << " after the restart, unexplained " << unexplained << "; moves: job_lost " << job_lost << ", consumed_lost " << consumed_lost << ", not_to_host " << not_to_host << ", host_lost " << host_lost << ", lost before written " << lost_before_written
                  << ", rows jobs released " << released << ", files written whole " << written << ", reads " << read << "\n";
        size_t short_turns = 0;
        for (const Row& row : book.report) short_turns += row.turn > 1 && row.reused < row.expected;
        table.push_back({"past half the pool, with a host tier smaller than a history, a turn after an unrelated request loses no row its history held", "P1, P3", true, !evicted});
        table.push_back({"a job that gives way loses no row it read", "P7", true, !job_lost});
        table.push_back({"a cut turn reuses its leading message's rows beside an unrelated request", "P5", true, !cut.evicted && cut.reused >= cut.expected});
        table.push_back({"a restart over kept files of this build costs the next turn a read, not a recompute", "P4", true, read && restarted.reused >= restarted.expected});
        table.push_back({"every turn computes only the rows its history did not hold", "P1", true, !short_turns});
        table.push_back({"every row a turn falls short by is a loss the server reported with its history", "P8", true, !unexplained});
    } catch (...) {
        sched->stop();
        runner.join();
        throw;
    }
    sched->stop();
    runner.join();
    for (size_t i = 0; i < chat.said.size(); ++i) {
        auto fresh = make(kPool, 64);
        same(serve(*fresh, tok, 4, {{chat.said[i].first}})[0], chat.said[i].second, "cache-scenario" + on_what + ", production's proportions: turn " + std::to_string(i + 1) + "'s reply");
    }
    for (const Row& row : book.report) require(row.reused + row.new_work + row.evicted + row.failed_restore + row.unexplained == row.prompt, "cache-scenario" + on_what + ", production's proportions: turn " + std::to_string(row.turn) + "'s kinds do not add up to its prompt");
    for (const Expectation& e : table) {
        std::cout << "cache-scenario" << on_what << ", production's proportions: " << (e.seen ? "held" : "broken") << ": " << e.name << " (" << e.promise << ")\n";
        require(e.seen == e.held_today, "cache-scenario" + on_what + ", production's proportions: \"" + e.name + "\" (" + e.promise + ") is " + (e.seen ? "held" : "broken") + ", and the plan records it as " + (e.held_today ? "held" : "broken") + " today");
    }
}

// Two conversations under one system prompt of more than two blocks, each read again while two other requests hold both seats: the second's job must not drop the first's, which shares only that start (Scheduler::stale), so both wait.
void shared_system_prompt(const Make& make, const bpe::Tokenizer& tok, uint32_t v) {
    auto model = make(kPool, 64);
    server::Scheduler sched(*model, tok, 2, 64, 0, false, (size_t)1 << 30);
    const std::vector<uint32_t> system = prompt_of(5, 300, v);
    std::vector<Req> turns;
    for (const uint32_t seed : {11u, 12u}) {
        Req r{system, (int)kReply};
        const std::vector<uint32_t> user = prompt_of(seed, 100, v);
        r.prompt.insert(r.prompt.end(), user.begin(), user.end());
        turns.push_back(r);
    }
    const Req other{prompt_of(20, 200, v), (int)kReply}, another{prompt_of(21, 200, v), (int)kReply};
    std::mutex handles;
    std::vector<Handle> hs;
    std::vector<std::vector<uint32_t>> histories;
    Handle busy, also;
    size_t busy_passes = 0, waiting = 0;
    // At the third request's first pass both conversations' next turns are given, and at its second, the round between having taken them in, both jobs wait for the seat.
    sched.on_retire = [&](const server::Scheduler::Retired& t) {
        std::lock_guard<std::mutex> lk(handles);
        if (!busy || std::find(t.requests.begin(), t.requests.end(), busy.get()) == t.requests.end()) return;
        if (++busy_passes == 1)
            for (size_t i = 0; i < hs.size(); ++i) sched.follow(hs[i], histories[i], true);
        if (busy_passes == 2) waiting = sched.stats().jobs;
    };
    std::thread runner([&] { sched.run(); });
    try {
        for (const Req& r : turns) {
            Handle h = sched.submit(r.prompt, params_of(r));
            std::vector<uint32_t> history = r.prompt;
            for (uint32_t id : ids_of(drain(*h))) history.push_back(id);
            std::lock_guard<std::mutex> lk(handles);
            hs.push_back(h);
            histories.push_back(history);
        }
        {
            std::lock_guard<std::mutex> lk(handles);
            also = sched.submit(another.prompt, params_of(another));
            busy = sched.submit(other.prompt, params_of(other));
        }
        drain(*busy);
        drain(*also);
        std::lock_guard<std::mutex> lk(handles);
        require(busy_passes >= 2 && waiting == 2, "two conversations under one system prompt: " + std::to_string(waiting) + " of their jobs waiting, against 2");
    } catch (...) {
        sched.stop();
        runner.join();
        throw;
    }
    sched.stop();
    runner.join();
}

// A restore from disk the turn then reuses (production, 2026-10-16 17:56): a conversation kept across a restart is read back into a host tier of one copy while a conversation that came back holds the devices.
// Promoting it evicts that donor, whose copy wants the host tier's one room: it must not take the copy being promoted, or the read is lost and the turn computes all its rows.
void restore_beside_donor(const Make& make, const bpe::Tokenizer& tok, uint32_t v) {
    DiskRoot disk("restore-beside");
    server::DiskOptions options = disk.options(uint64_t(1) << 30);
    options.keep = true;
    options.idle = std::chrono::hours(1);
    // A read is waited for whatever this runner computes a row in, so a slow runner computes no row a read gives; the bound itself is disk_read_bound's case in server-resume.
    options.read_wait = std::chrono::seconds(60);
    const Req first{prompt_of(5, 1100, v), (int)kReply}, other{prompt_of(9, 900, v), (int)kOtherReply};
    std::vector<uint32_t> next = first.prompt;
    size_t reused = 0;
    Reply got;
    for (const bool restarted : {false, true}) {
        auto model = make(kPool, 64);
        const size_t host = model->host_bytes(kBlock);
        server::Scheduler sched(*model, tok, 2, 64, 0, false, host, nullptr, 0, true, options);
        std::thread runner([&] { sched.run(); });
        try {
            if (!restarted) {
                for (uint32_t id : ids_of(drain(*sched.submit(first.prompt, params_of(first))))) next.push_back(id);
                const std::vector<uint32_t> user = prompt_of(30, kUser, v);
                next.insert(next.end(), user.begin(), user.end());
            } else {
                within_a_minute([&] { return sched.stats().disk_ready && sched.stats().disk_entries > 0; }, "restore beside a donor: the kept files adopted");
                // The other conversation comes back, so its donor is one the host tier keeps first.
                std::vector<uint32_t> again = other.prompt;
                for (uint32_t id : ids_of(drain(*sched.submit(other.prompt, params_of(other))))) again.push_back(id);
                const std::vector<uint32_t> more = prompt_of(31, kUser, v);
                again.insert(again.end(), more.begin(), more.end());
                drain(*sched.submit(again, params_of(Req{again, (int)kOtherReply})));
                const Req turn{next, (int)kReply};
                const auto h = sched.submit(turn.prompt, params_of(turn));
                got = drain(*h);
                reused = h->reused();
            }
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    require(reused == 1024, "a turn restored from disk beside another conversation's donor reused " + std::to_string(reused) + " tokens, against the 1024 its kept history holds");
    auto fresh = make(kPool, 64);
    same(serve(*fresh, tok, 4, {{Req{next, (int)kReply}}})[0], got, "restore beside a donor, the turn's reply");
}

// A donor's write slower than the wait's constant (production, 2026-10-17 run D), each 4 MiB held five seconds, whose room another request needs once the write is under way.
// The wait lets the next turn read it back; `capped` at 1 us a row of recomputing, each piece held five minutes, it reaches its bound and is counted (AGENTS.md, Tests).
void slow_write_wait(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok, uint32_t v, bool capped) {
    const std::string what = capped ? "a write slower than a wait capped by recomputing" : "a write slower than the wait's constant";
    DiskRoot disk(capped ? "slow-write-capped" : "slow-write");
    server::DiskOptions options = disk.options(uint64_t(1) << 30, capped ? std::chrono::milliseconds(std::chrono::minutes(5)) : std::chrono::milliseconds(5000));
    options.idle = std::chrono::hours(1);
    options.read_wait = std::chrono::seconds(60);
    options.wait = std::chrono::milliseconds(100);
    // The cap by recomputing is the test's, so this runner's speed cannot reach it, or reaches it at once.
    options.recompute_row_ms = capped ? 0.001 : 50.0;
    const auto slow = hooked(1);
    slow[0]->hook = [] { std::this_thread::sleep_for(std::chrono::milliseconds(30)); };
    const Make make = on(weights, [slow] { return std::vector<backend::BackendPtr>{slow[0]}; }), plain = on(weights, [] { return cpus(1); });
    auto model = make(kPool, 64);
    server::Scheduler sched(*model, tok, 2, 64, 0, false, model->host_bytes(kBlock) - 1, nullptr, 0, true, options);
    std::thread runner([&] { sched.run(); });
    const Req first{prompt_of(5, 1100, v), (int)kReply}, other{prompt_of(9, 1000, v), (int)kOtherReply};
    size_t reused = 0;
    Stats st;
    std::vector<uint32_t> next;
    Reply got;
    try {
        within(std::chrono::minutes(10), [&] { return sched.stats().disk_ready; }, "slow write: the disk tier made", "within ten minutes");
        next = first.prompt;
        for (uint32_t id : ids_of(drain(*sched.submit(first.prompt, params_of(first))))) next.push_back(id);
        // A write not yet begun gives the wait only its constant, so the other request comes once the history's write is under way, whenever this runner starts it.
        within(std::chrono::minutes(10), [&] { return sched.stats().disk_in_flight > 0; }, "slow write: the history's write under way", "within ten minutes");
        drain(*sched.submit(other.prompt, params_of(other)));
        const std::vector<uint32_t> user = prompt_of(30, kUser, v);
        next.insert(next.end(), user.begin(), user.end());
        const auto h = sched.submit(next, params_of(Req{next, (int)kReply}));
        got = drain(*h);
        reused = h->reused();
        st = sched.stats();
    } catch (...) {
        sched.stop();
        runner.join();
        throw;
    }
    sched.stop();
    runner.join();
    const std::string seen = std::to_string(st.room_waits) + " waits, " + std::to_string(st.room_wait_timeouts) + " at their bound, the next turn reusing " + std::to_string(reused);
    if (capped) require(st.room_waits >= 1 && st.room_wait_timeouts >= 1, what + ": " + seen + ", against some and one at least");
    else require(st.room_waits >= 1 && !st.room_wait_timeouts && reused == 1024, what + ": " + seen + ", against some, none and 1024");
    auto fresh = plain(kPool, 64);
    same(serve(*fresh, tok, 4, {{Req{next, (int)kReply}}})[0], got, what + ", the next turn's reply");
}

// A server's first write, with no rate measured before it (production, 2026-10-17 run D at 90e20b26d): the store's probe fails, so the wait has only its constant, here 1500 ms, until the write in flight gives its own rate.
// Each history's 2 MiB go in one 4 MiB piece the store holds 3 s, written or read, longer than the constant and than 2 s past what a probe's rate would give; recomputing it is capped at 50 ms a row, about a minute.
void first_write_wait(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok, uint32_t v) {
    DiskRoot disk("first-write");
    const auto slow = hooked(1);
    auto slowed = std::make_shared<std::atomic<bool>>(false);
    slow[0]->hook = [slowed] {
        if (slowed->load()) std::this_thread::sleep_for(std::chrono::milliseconds(300));
    };
    const Make make = on(weights, [slow] { return std::vector<backend::BackendPtr>{slow[0]}; }), plain = on(weights, [] { return cpus(1); });
    auto model = make(kPool, 64);
    const Req first{prompt_of(5, 1100, v), 4}, other{prompt_of(9, 1000, v), 4};
    server::DiskOptions options = disk.options(uint64_t(1) << 30, std::chrono::milliseconds(3000));
    options.idle = std::chrono::hours(1);
    options.read_wait = std::chrono::seconds(60);
    options.wait = std::chrono::milliseconds(1500);
    options.recompute_row_ms = 50.0;
    options.fault = [](const char* step) { return std::string(step) == "probe"; };
    server::Scheduler sched(*model, tok, 2, 64, 0, false, model->host_bytes(kBlock) - 1, nullptr, 0, true, options);
    std::thread runner([&] { sched.run(); });
    size_t reused = 0;
    Stats st;
    std::vector<uint32_t> next;
    Reply got;
    try {
        // The store holds each 4 MiB 3 s, so its writes are waited for by their event with a bound only a hang reaches: a slow runner's queue of them passed a minute.
        within(std::chrono::minutes(10), [&] { return sched.stats().disk_ready; }, "first write: the disk tier made", "within ten minutes");
        next = first.prompt;
        slowed->store(true);
        for (uint32_t id : ids_of(drain(*sched.submit(first.prompt, params_of(first))))) next.push_back(id);
        drain(*sched.submit(other.prompt, params_of(other)));
        slowed->store(false);
        // The other request's history lands too, a file of its own, so the next turn's read waits on no write of it.
        // The first history is whole on disk, since the other request waited for it, but in one write or two as the budget fell, so the files are counted and not its bytes.
        const size_t files = sched.stats().disk_entries;
        within(std::chrono::minutes(10), [&] { const Stats now = sched.stats(); return now.disk_entries > files && !now.disk_in_flight; }, "first write: the other request's history written",
               "within ten minutes");
        const std::vector<uint32_t> user = prompt_of(30, kUser, v);
        next.insert(next.end(), user.begin(), user.end());
        const auto h = sched.submit(next, params_of(Req{next, (int)kReply}));
        got = drain(*h);
        reused = h->reused();
        st = sched.stats();
    } catch (...) {
        slowed->store(false);
        sched.stop();
        runner.join();
        throw;
    }
    sched.stop();
    runner.join();
    require(st.room_waits >= 1 && !st.room_wait_timeouts && reused == 1024,
            "a first write with no rate before it: " + std::to_string(st.room_waits) + " waits of " + std::to_string((long long)st.room_wait_ms) + " ms, " + std::to_string(st.room_wait_timeouts) +
                " at their bound, the next turn reusing " + std::to_string(reused) + ", against some, none and 1024");
    auto fresh = plain(kPool, 64);
    same(serve(*fresh, tok, 4, {{Req{next, (int)kReply}}})[0], got, "first write, the next turn's reply");
}

// A pool sized apart from a request's limit (docs/CACHE-PLAN.md, section 7): a limit below the context beside a larger pool holds requests to it with unchanged replies, and a pool below the limit holds them to the pool.
// A history kept at a pool of one context is read back at a pool of two, since a kept entry's identity reads the model's own limit.
void pool_past_context(const Make& make, const bpe::Tokenizer& tok, uint32_t v) {
    {
        auto model = make(2 * kPool, 64);
        server::Scheduler sched(*model, tok, 2, 64, 0, false, 0, nullptr, 0, true, {}, {}, 512);
        require(sched.token_limit() == 512, "a request limit of 512 beside a pool of " + std::to_string(model->kv_tokens_total()) + " gave requests " + std::to_string(sched.token_limit()));
        bool refused = false;
        try {
            sched.submit(prompt_of(3, 600, v), params_of(Req{prompt_of(3, 600, v), 8}));
        } catch (const server::TooLong&) {
            refused = true;
        }
        require(refused, "a 600-token prompt was taken under a request limit of 512");
        const Req fits{prompt_of(4, 300, v), (int)kReply};
        std::thread runner([&] { sched.run(); });
        Reply got;
        try {
            got = drain(*sched.submit(fits.prompt, params_of(fits)));
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
        auto one = make(kPool, 64);
        same(serve(*one, tok, 2, {{fits}})[0], got, "a request under a limit below the context beside a larger pool");
    }
    {
        auto model = make(2 * kBlock, 64);
        server::Scheduler sched(*model, tok, 2, 64, 0, false, 0, nullptr, 0, true, {}, {}, 512);
        require(sched.token_limit() == 2 * kBlock, "a pool of 256 tokens below a request limit of 512 gave requests " + std::to_string(sched.token_limit()));
    }
    DiskRoot disk("pool-past-context");
    server::DiskOptions options = disk.options(uint64_t(1) << 30);
    options.keep = true;
    options.idle = std::chrono::hours(1);
    // A read is waited for whatever this runner computes a row in, so a slow runner computes no row a read gives; the bound itself is disk_read_bound's case in server-resume.
    options.read_wait = std::chrono::seconds(60);
    const Req first{prompt_of(5, 1100, v), (int)kReply};
    std::vector<uint32_t> next = first.prompt;
    size_t reused = 0;
    Reply got;
    for (const size_t pool : {kPool, 2 * kPool}) {
        auto model = make(pool, 64);
        server::Scheduler sched(*model, tok, 2, 64, 0, false, model->host_bytes(kBlock), nullptr, 0, true, options);
        std::thread runner([&] { sched.run(); });
        try {
            if (pool == kPool) {
                for (uint32_t id : ids_of(drain(*sched.submit(first.prompt, params_of(first))))) next.push_back(id);
                const std::vector<uint32_t> user = prompt_of(30, kUser, v);
                next.insert(next.end(), user.begin(), user.end());
            } else {
                within_a_minute([&] { return sched.stats().disk_ready && sched.stats().disk_entries > 0; }, "a pool past the context: the kept files adopted");
                const auto h = sched.submit(next, params_of(Req{next, (int)kReply}));
                got = drain(*h);
                reused = h->reused();
            }
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    require(reused == 1024, "a history kept at a pool of one context and read back at a pool of two reused " + std::to_string(reused) + " tokens, against 1024");
    auto fresh = make(kPool, 64);
    same(serve(*fresh, tok, 4, {{Req{next, (int)kReply}}})[0], got, "a pool past the context, the turn after the restart");
}

// A scheduler running for one leg, its losses kept for the classification, stopped as it goes out of scope; a stop under keep writes what memory holds to disk.
struct Run {
    std::unique_ptr<infer::Model> model;
    std::unique_ptr<server::Scheduler> sched;
    std::mutex losing, submitting;
    std::vector<Loss> losses;
    std::thread runner;
    Run(std::unique_ptr<infer::Model> m, const bpe::Tokenizer& tok, size_t seats, size_t host, const server::DiskOptions& disk = {}) : model(std::move(m)) {
        sched = std::make_unique<server::Scheduler>(*model, tok, seats, 64, 0, false, host, nullptr, 0, true, disk);
        sched->on_move = [this](const server::Scheduler::Moved& mv) {
            std::lock_guard<std::mutex> lk(losing);
            losses.push_back(Loss{mv.kind, mv.first, mv.tokens, std::vector<uint32_t>(mv.history->begin(), mv.history->begin() + (std::ptrdiff_t)std::min(mv.tokens, mv.history->size()))});
        };
        // No pass retires while the test submits several requests, so they are queued together.
        sched->on_retire = [this](const server::Scheduler::Retired&) { std::lock_guard<std::mutex> lk(submitting); };
        runner = std::thread([this] { sched->run(); });
    }
    ~Run() {
        sched->stop();
        runner.join();
    }
    size_t lost() {
        std::lock_guard<std::mutex> lk(losing);
        return losses.size();
    }
    // The rows of a shortfall [reused, expected) the losses reported from `since` on explain, and those the turn's own read covered past its bound (failed_rows).
    std::pair<size_t, size_t> classified(const std::vector<uint32_t>& prompt, size_t reused, size_t expected, size_t since, size_t late) {
        std::lock_guard<std::mutex> lk(losing);
        const std::vector<char> lost = lost_rows(prompt, reused, expected, losses, since);
        return {(size_t)std::count(lost.begin(), lost.end(), (char)1), failed_rows(lost, reused, expected, late)};
    }
    size_t count(server::Move k) { return sched->stats().moves[(size_t)k]; }
};

std::vector<uint32_t> joined(std::vector<uint32_t> a, const std::vector<uint32_t>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

// Each expectation is what the plan records for today: one that fails, or improves without its entry being updated, fails here.
void check(const std::vector<Expectation>& table, const std::string& on_what) {
    for (const Expectation& e : table) {
        std::cout << "cache-scenario" << on_what << ": " << (e.seen ? "held" : "broken") << ": " << e.name << " (" << e.promise << ")\n";
        require(e.seen == e.held_today, "cache-scenario" + on_what + ": \"" + e.name + "\" (" + e.promise + ") is " + (e.seen ? "held" : "broken") + ", and the plan records it as " + (e.held_today ? "held" : "broken") + " today");
    }
}

// A restart over kept files: two conversations each leave a turn and its read-ahead job on disk, and a new server opens the directory.
// Two requests of the first history arrive together, the second conversation's turn is sent again with its job after it, whose rows are on disk; with `leave` the first follower leaves while both wait for the one read the test holds.
void restart(const Make& make, const bpe::Tokenizer& tok, uint32_t v, bool leave) {
    const std::string on_what = std::string(", a restart over kept files") + (leave ? ", one follower leaving the read" : "");
    // A pool of two of the scenario's, so each job finds the spare room it starts only in beside both conversations' donors.
    DiskRoot disk(leave ? "scenario-restart-leave" : "scenario-restart");
    server::DiskOptions options = disk.options(uint64_t(1) << 30);
    options.keep = true;
    // The reads' chunks wait while `held`.
    struct Hold {
        std::mutex m;
        std::condition_variable cv;
        bool held = false;
    };
    auto hold = std::make_shared<Hold>();
    options.fault = [hold](const char* step) {
        std::unique_lock<std::mutex> lk(hold->m);
        if (std::string(step) == "read") hold->cv.wait(lk, [&] { return !hold->held; });
        return false;
    };
    const auto held = [&](bool on) {
        {
            std::lock_guard<std::mutex> lk(hold->m);
            hold->held = on;
        }
        hold->cv.notify_all();
    };
    // A read is waited for whatever this runner computes a row in, so the leg counts reads and not a race with the bound.
    options.read_wait = std::chrono::seconds(60);
    const Req one{prompt_of(21, 600, v), (int)kReply}, two{prompt_of(31, 600, v), (int)kReply};
    std::vector<uint32_t> ids_one, ids_two;
    {
        Run before(make(2 * kPool, 64), tok, 4, (size_t)1 << 30, options);
        size_t jobs = 0;
        for (const auto& c : {std::pair<const Req*, std::vector<uint32_t>*>{&one, &ids_one}, {&two, &ids_two}}) {
            const Handle h = before.sched->submit(c.first->prompt, params_of(*c.first));
            *c.second = joined(joined(c.first->prompt, ids_of(drain(*h))), prompt_of(22, 200, v));
            before.sched->follow(h, *c.second, true);
            within_a_minute([&] { return before.sched->stats().reprefills == jobs + 1; }, "the restart leg: a job done before the stop");
            ++jobs;
        }
    }
    Run after(make(2 * kPool, 64), tok, 4, (size_t)1 << 30, options);
    within_a_minute([&] { return after.sched->stats().disk_ready; }, "the restart leg: the kept files adopted");
    const size_t point = ids_one.size() / kBlock * kBlock;
    // Two followers of the first history, which is on disk alone.
    const Req next{joined(ids_one, prompt_of(23, 60, v)), (int)kReply};
    Handle a, b;
    held(leave);
    {
        std::lock_guard<std::mutex> lk(after.submitting);
        a = after.sched->submit(next.prompt, params_of(next));
        b = after.sched->submit(next.prompt, params_of(next));
    }
    if (leave) {
        try {
            within_a_minute([&] { return after.sched->stats().disk_waits >= 2; }, "the restart leg: both followers waiting for the read");
            a->cancel();
            collect(*a);
            require(a->finish() == "cancel", "cache-scenario" + on_what + ": the follower that left ended with " + a->finish());
        } catch (...) {
            held(false);
            throw;
        }
        held(false);
        const Reply rb = drain(*b);
        const size_t reads = after.count(server::Move::read_started);
        std::cout << "cache-scenario" << on_what << ": the other follower reused " << b->reused() << " of " << point << " with " << reads << " reads from disk\n";
        require(reads == 1 && b->reused() == point, "cache-scenario" + on_what + ": " + std::to_string(reads) + " reads and the other follower reusing " + std::to_string(b->reused()) + ", against 1 and " + std::to_string(point));
        auto fresh = make(kPool, 64);
        same(serve(*fresh, tok, 4, {{next}})[0], rb, "cache-scenario" + on_what + ": the other follower's reply");
        return;
    }
    const Reply ra = drain(*a), rb = drain(*b);
    const size_t reads = after.count(server::Move::read_started);
    // The second conversation's turn again, and its job: the rows the job is to read are in the files the earlier job's donor left.
    const Handle q = after.sched->submit(two.prompt, params_of(two));
    const Reply rq = drain(*q);
    const size_t q_reads = after.count(server::Move::read_started), rows = after.sched->stats().reprefill_rows;
    after.sched->follow(q, ids_two, true);
    within_a_minute([&] { return after.sched->stats().reprefills == 1; }, "the restart leg: the job after the restart done");
    const size_t job_rows = after.sched->stats().reprefill_rows - rows, job_reads = after.count(server::Move::read_started) - q_reads;
    std::cout << "cache-scenario" << on_what << ": the two followers reused " << a->reused() << " and " << b->reused() << " of " << point << " with " << reads << " reads from disk; the turn sent again reused " << q->reused()
              << "; its job read " << job_rows << " rows with " << job_reads << " reads from disk\n";
    std::vector<Expectation> table;
    // A follower whose read ends after the other's prompt has passed the history forks that request instead, so it may reuse more.
    table.push_back({"after a restart a turn reads its history and does not recompute", "P4", true, a->reused() >= point && b->reused() >= point});
    table.push_back({"two followers of one history on disk share one read", "P2", true, reads == 1});
    table.push_back({"after a restart a job reads and does not recompute", "P4", true, job_rows == 0});
    require(reads == 1 && q->reused() == 512 && job_rows == 0 && job_reads == 1,
            "the restart leg's witness: " + std::to_string(reads) + " reads for the two followers, the turn sent again reusing " + std::to_string(q->reused()) + ", its job reading " + std::to_string(job_rows) + " rows with " +
                std::to_string(job_reads) + " reads, against 1, 512, 0 and 1");
    check(table, on_what);
    auto fresh = make(kPool, 64);
    const std::vector<Reply> alone = serve(*fresh, tok, 4, {{next}, {two}});
    same(alone[0], ra, "cache-scenario" + on_what + ": the first follower's reply");
    same(alone[0], rb, "cache-scenario" + on_what + ": the second follower's reply");
    same(alone[1], rq, "cache-scenario" + on_what + ": the turn sent again");
}

// A history the host tier cannot keep: another conversation's request takes its room at once, or after its files are whole on disk.
void beyond_the_host_tier(const Make& make, const bpe::Tokenizer& tok, uint32_t v, bool written) {
    const std::string on_what = std::string(", a history the host tier cannot keep") + (written ? ", written before it is taken" : ", taken at once");
    DiskRoot disk(written ? "scenario-beyond-written" : "scenario-beyond");
    server::DiskOptions options = disk.options(uint64_t(1) << 30);
    options.keep = true;
    options.idle = written ? std::chrono::milliseconds(0) : std::chrono::milliseconds(std::chrono::hours(1));
    options.read_wait = std::chrono::seconds(60);
    const Req turn{prompt_of(41, 700, v), 20}, other{prompt_of(42, 500, v), 20};
    auto model = make(1024, 64);
    // The tier is a byte short of one slab, the least a copy takes, so no history has a place in it.
    const size_t cap = infer::Model::host_slab_bytes() - 1, point = turn.prompt.size() / kBlock * kBlock;
    require(model->host_bytes(point) > cap, "the fixture's history fits the host tier");
    Run run(std::move(model), tok, 2, cap, options);
    const Handle h = run.sched->submit(turn.prompt, params_of(turn));
    const Req next{joined(joined(turn.prompt, ids_of(drain(*h))), prompt_of(43, 60, v)), 20};
    if (written) within_a_minute([&] { return run.count(server::Move::write_whole) >= 1 && !run.sched->stats().disk_in_flight; }, "the leg beyond the host tier: its files whole on disk");
    const size_t since = run.lost();
    drain(*run.sched->submit(other.prompt, params_of(other)));
    const Handle f = run.sched->submit(next.prompt, params_of(next));
    const Reply reply = drain(*f);
    const size_t reused = f->reused();
    const auto [evicted, failed] = run.classified(next.prompt, reused, point, since, f->restore_late());
    const size_t unexplained = point - std::min(point, reused) - evicted - failed;
    std::cout << "cache-scenario" << on_what << ": reused " << reused << ", evicted " << evicted << ", failed restore " << failed << " of " << point << "\n";
    require(!unexplained, "cache-scenario" + on_what + ": " + std::to_string(unexplained) + " of " + std::to_string(point) + " rows computed that no reported loss or failed restore explains (P8)");
    std::vector<Expectation> table;
    table.push_back({"a history the host tier cannot keep is whole on disk before it can be taken", "P3", true, !evicted});
    require(reused == point && !failed, "the leg's witness: reused " + std::to_string(reused) + " and failed restore " + std::to_string(failed) + ", against " + std::to_string(point) + " and none");
    check(table, on_what);
    auto fresh = make(1024, 64);
    same(serve(*fresh, tok, 2, {{next}})[0], reply, "cache-scenario" + on_what + ": the follow-up's reply");
}

// Many users at the cap, on a pool and a host tier that hold them all (`room`), or a pool for three and a host tier for one more: six conversations taking turns, three each (`room` and `cycle`),
// or two conversations that come back every round beside four one-off requests a round (`returning`), the production shape the one ranking is for (docs/CACHE-PLAN.md, section 4).
enum class Users { room, cycle, returning };
void many_users(const Make& make, const bpe::Tokenizer& tok, uint32_t v, Users mode) {
    const bool room = mode == Users::room;
    const std::string on_what = mode == Users::returning ? ", two conversations coming back beside four one-off requests a round, room for four"
                                                         : std::string(", six conversations taking turns") + (room ? ", room for all" : ", room for four");
    auto model = make(room ? 8192 : kPool, 64);
    const size_t host = room ? (size_t)1 << 30 : model->host_bytes(1024);
    Run run(std::move(model), tok, 4, host);
    struct User {
        std::vector<uint32_t> history;
        size_t prepared = 0, since = 0, evicted = 0, reused = 0;
    };
    size_t shortfall = 0, unexplained = 0, ended = 0;
    const auto turn = [&](User& c, uint32_t seed) {
        const Req r{joined(c.history, prompt_of(seed % v, kUser, v)), (int)kReply};
        const Handle h = run.sched->submit(r.prompt, params_of(r));
        const Reply reply = drain(*h);
        ++ended;
        const size_t reused = h->reused(), missing = c.prepared > reused ? c.prepared - reused : 0;
        const auto [evicted, failed] = run.classified(r.prompt, reused, c.prepared, c.since, h->restore_late());
        shortfall += missing;
        unexplained += missing - evicted - failed;
        c.evicted += evicted;
        c.reused += reused;
        c.history = joined(r.prompt, ids_of(reply));
        c.prepared = r.prompt.size() / kBlock * kBlock;
        c.since = run.lost();
    };
    std::vector<User> users(mode == Users::returning ? 2 : 6);
    std::vector<Expectation> table;
    if (mode == Users::returning) {
        // The two take a turn each and come back once before the one-offs start, so they have come back when the tiers first fill.
        for (User& c : users) c.history = prompt_of((uint32_t)(50 + (&c - users.data())), 200, v);
        for (uint32_t warm = 0; warm < 2; ++warm)
            for (size_t u = 0; u < users.size(); ++u) turn(users[u], warm * 7 + (uint32_t)u);
        const size_t before = users[0].evicted + users[1].evicted;
        for (uint32_t round = 0; round < 3; ++round) {
            for (size_t u = 0; u < users.size(); ++u) turn(users[u], 20 + round * 7 + (uint32_t)u);
            for (uint32_t k = 0; k < 4; ++k) {
                User once;
                once.history = prompt_of((uint32_t)((10 + round * 4 + k) % v), 300, v);
                turn(once, 30 + round * 4 + k);
            }
        }
        std::cout << "cache-scenario" << on_what << ": " << ended << " turns ended, " << shortfall << " rows short of what was prepared, " << unexplained << " unexplained; the two lost " << users[0].evicted << " and "
                  << users[1].evicted << " and reused " << users[0].reused << " and " << users[1].reused << "\n";
        require(ended == 22, "cache-scenario" + on_what + ": not every turn ended");
        table.push_back({"at the cap the conversations that came back keep their rows over those that did not", "P3", true, users[0].evicted + users[1].evicted == before});
    } else {
        for (size_t u = 0; u < users.size(); ++u) users[u].history = prompt_of((uint32_t)(50 + u), 500, v);
        for (uint32_t t = 0; t < 3; ++t)
            for (size_t u = 0; u < users.size(); ++u) turn(users[u], t * 7 + (uint32_t)u);
        std::cout << "cache-scenario" << on_what << ": " << ended << " turns ended, " << shortfall << " rows short of what was prepared, " << unexplained << " of them unexplained; evicted by conversation:";
        for (const User& c : users) std::cout << " " << c.evicted;
        std::cout << "\n";
        require(ended == 18, "cache-scenario" + on_what + ": not every turn ended");
        if (room) table.push_back({"with room for every conversation no turn computes a row it held", "P1, P2", true, !shortfall});
        // A pure cycle past the tiers' room, which no recency order wins, is a recorded count: each conversation loses its 512 and then 640 prepared rows, each loss reported.
        else
            for (size_t u = 0; u < users.size(); ++u)
                require(users[u].evicted == 1152, "the many-users cycle: conversation " + std::to_string(u) + " lost " + std::to_string(users[u].evicted) + " rows, against 1152");
    }
    require(!unexplained, "cache-scenario" + on_what + ": " + std::to_string(unexplained) + " rows computed that a restore point should have held and no reported loss explains (P8)");
    check(table, on_what);
}

// A21: a read-ahead job with one block left when another conversation's request arrives that does not fit beside it, the turn's donor beyond the host tier.
// `held`: the donor's write is held, so the request waits for a write and the job reads its block first, `costly` alike with steps past the allowance; `ended`: the write ends while the job is spared; `whole`: written before.
enum class LastBlock { held, costly, ended, whole };
void last_block(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok, uint32_t v, LastBlock mode) {
    const std::string on_what = std::string(", a job with its last block left") +
                                (mode == LastBlock::held     ? ", the donor's write held"
                                 : mode == LastBlock::costly ? ", the donor's write held, each step costing more than the allowance"
                                 : mode == LastBlock::ended  ? ", the write ending while the job is spared"
                                                             : ", the donor whole on disk");
    DiskRoot disk(mode == LastBlock::held ? "scenario-last-held" : mode == LastBlock::costly ? "scenario-last-costly" : mode == LastBlock::ended ? "scenario-last-ended" : "scenario-last-whole");
    const bool held = mode == LastBlock::held || mode == LastBlock::costly;
    // A write's chunks wait while `writes` is held, and a pass's retirement while `retires` is 0, so the test acts between two rounds.
    struct Gate {
        std::mutex m;
        std::condition_variable cv;
        bool writes = false, waiting = false;
        size_t retires = std::numeric_limits<size_t>::max();
    };
    auto gate = std::make_shared<Gate>();
    gate->writes = mode != LastBlock::whole;
    const auto set = [&](bool writes, size_t retires) {
        {
            std::lock_guard<std::mutex> lk(gate->m);
            gate->writes = writes;
            gate->retires = retires;
        }
        gate->cv.notify_all();
    };
    // Waits until a pass waits to retire, with none let through since.
    const auto at_retire = [&](const std::string& what) {
        std::unique_lock<std::mutex> lk(gate->m);
        require(gate->cv.wait_for(lk, std::chrono::minutes(1), [&] { return gate->waiting && !gate->retires; }), "cache-scenario" + on_what + ": " + what);
    };
    server::DiskOptions options = disk.options(uint64_t(1) << 30);
    options.keep = true;
    options.idle = std::chrono::hours(1);
    options.read_wait = std::chrono::seconds(60);
    options.fault = [gate](const char* step) {
        std::unique_lock<std::mutex> lk(gate->m);
        if (std::string(step) == "write") gate->cv.wait(lk, [&] { return !gate->writes; });
        return false;
    };
    const Req turn{prompt_of(44, 700, v), 20}, other{prompt_of(45, 500, v), 20};
    const auto slow = hooked(1);
    auto slowed = std::make_shared<std::atomic<bool>>(false), armed = std::make_shared<std::atomic<bool>>(false);
    const Make make = on(weights, [slow] { return std::vector<backend::BackendPtr>{slow[0]}; }), plain = on(weights, [] { return cpus(1); });
    auto model = make(1024, 16);
    // The tier is a byte short of one slab, so the disk is the turn's donor's only save.
    const size_t cap = infer::Model::host_slab_bytes() - 1;
    server::Scheduler sched(*model, tok, 2, 64, 0, false, cap, nullptr, 0, true, options);
    Handle b;
    std::atomic<bool> sent{false};
    // Under `held` the other request is sent as the job's first pass retires, which no hold lengthens, since a held pass's time becomes the job's measured cost.
    sched.on_retire = [&, gate, armed](const server::Scheduler::Retired&) {
        if (armed->exchange(false)) {
            b = sched.submit(other.prompt, params_of(other));
            sent = true;
        }
        std::unique_lock<std::mutex> lk(gate->m);
        gate->waiting = true;
        gate->cv.notify_all();
        gate->cv.wait(lk, [&] { return gate->retires > 0; });
        if (gate->retires != std::numeric_limits<size_t>::max()) --gate->retires;
        gate->waiting = false;
    };
    // Under `held` a step of the job costs more than half of a gap's background allowance and under `costly` more than the whole, spun, not slept, since sleeps round up to the timer's tick.
    const auto spin = std::chrono::milliseconds(mode == LastBlock::costly ? 30 : 8);
    slow[0]->hook = [slowed, spin] {
        for (const auto until = std::chrono::steady_clock::now() + spin; slowed->load() && std::chrono::steady_clock::now() < until;) {}
    };
    std::thread runner([&] { sched.run(); });
    const auto count = [&](server::Move k) { return sched.stats().moves[(size_t)k]; };
    size_t done = 0, gave = 0, reused = 0;
    Reply rb, rc;
    Req next;
    try {
        within_a_minute([&] { return sched.stats().disk_ready; }, "the last-block leg: the disk tier made");
        const Handle h = sched.submit(turn.prompt, params_of(turn));
        // The job forks the turn's 640 prompt rows and reads to 768, one block, its ids the turn, its reply and 52 more.
        const std::vector<uint32_t> ids = joined(joined(turn.prompt, ids_of(drain(*h))), prompt_of(46, 52, v));
        slowed->store(held);
        if (mode == LastBlock::whole) within_a_minute([&] { return count(server::Move::write_whole) >= 1; }, "the last-block leg: the turn's files whole on disk");
        set(mode != LastBlock::whole, held ? std::numeric_limits<size_t>::max() : 0);
        armed->store(held);
        sched.follow(h, ids, true);
        if (held) {
            within_a_minute([&] { return sent.load(); }, "the last-block leg: the other request sent as the job's first pass retired");
        } else {
            at_retire("the job's first pass held");
            b = sched.submit(other.prompt, params_of(other));
        }
        if (mode == LastBlock::ended) {
            // One round with the request waiting: the job is spared, and its next pass waits to retire.
            set(true, 1);
            at_retire("the spared job's next pass held");
            require(!count(server::Move::job_gave_way), "cache-scenario" + on_what + ": the job gave way while the write was held");
            set(false, 0);
            within_a_minute([&] { return !disk.files(".kv").empty(); }, "the last-block leg: the turn's files written");
        }
        set(held, std::numeric_limits<size_t>::max());
        within_a_minute([&] { return count(server::Move::job_done) + count(server::Move::job_gave_way) >= 1; }, "the last-block leg: the job ended or gave way");
        done = count(server::Move::job_done);
        gave = count(server::Move::job_gave_way);
        set(false, std::numeric_limits<size_t>::max());
        slowed->store(false);
        rb = drain(*b);
        next = Req{joined(ids, prompt_of(47, 30, v)), 20};
        const Handle c = sched.submit(next.prompt, params_of(next));
        rc = drain(*c);
        reused = c->reused();
    } catch (...) {
        set(false, std::numeric_limits<size_t>::max());
        slowed->store(false);
        sched.stop();
        runner.join();
        throw;
    }
    sched.stop();
    runner.join();
    std::cout << "cache-scenario" << on_what << ": " << done << " jobs done, " << gave << " gave way, the next turn reused " << reused << "\n";
    if (held)
        require(done == 1 && !gave && reused == 768, "cache-scenario" + on_what + ": " + std::to_string(done) + " jobs done and " + std::to_string(gave) + " gave way, the next turn reusing " +
                                                          std::to_string(reused) + ", against 1, 0 and the job's 768");
    else
        require(!done && gave == 1, "cache-scenario" + on_what + ": " + std::to_string(done) + " jobs done and " + std::to_string(gave) + " gave way, against 0 and 1");
    auto fresh = plain(1024, 16);
    const std::vector<Reply> alone = serve(*fresh, tok, 2, {{other}, {next}});
    same(alone[0], rb, "cache-scenario" + on_what + ": the other request");
    same(alone[1], rc, "cache-scenario" + on_what + ": the next turn");
}

} // namespace

// `legs` runs the scripted conversation and its cases, `production` the production-shaped script; with no argument, both.
int main(int argc, char** argv) {
    const std::string only = argc > 1 ? argv[1] : "";
    if (!only.empty() && only != "legs" && only != "production") {
        std::cerr << "usage: llmx-cache-scenario-test [legs|production]\n";
        return 2;
    }
    const bool legs = only != "production", shaped = only != "legs";
    try {
        if (legs) attribution();
        const gguf::GGUFModel weights = served(kCpu), deep = served(kSplit), mixed = served_hybrid(kHybrid);
        const Make one = on(weights, [] { return cpus(1); }), split = on(deep, [] { return cpus(2); });
        const Make hybrid = on(mixed, [] { return cpus(1); }, 4, 6), hybrid_split = on(mixed, [] { return cpus(2); }, 4, 6);
        const bpe::Tokenizer tok(weights), deep_tok(deep), mixed_tok(mixed);
        if (legs) {
            scenario(one, tok, (uint32_t)kCpu.vocab, "", {768, 512, 1152, true});
            shared_system_prompt(one, tok, (uint32_t)kCpu.vocab);
            restore_beside_donor(one, tok, (uint32_t)kCpu.vocab);
            for (const bool capped : {false, true}) slow_write_wait(weights, tok, (uint32_t)kCpu.vocab, capped);
            first_write_wait(weights, tok, (uint32_t)kCpu.vocab);
            pool_past_context(one, tok, (uint32_t)kCpu.vocab);
            for (const bool leave : {false, true}) restart(one, tok, (uint32_t)kCpu.vocab, leave);
            for (const bool written : {false, true}) beyond_the_host_tier(one, tok, (uint32_t)kCpu.vocab, written);
            for (const Users mode : {Users::room, Users::cycle, Users::returning}) many_users(one, tok, (uint32_t)kCpu.vocab, mode);
            for (const LastBlock mode : {LastBlock::held, LastBlock::costly, LastBlock::ended, LastBlock::whole}) last_block(weights, tok, (uint32_t)kCpu.vocab, mode);
            scenario(split, deep_tok, (uint32_t)kSplit.vocab, ", a split over two CPUs", {768, 512, 1152, true});
            scenario(hybrid, mixed_tok, (uint32_t)kHybrid.vocab, ", the hybrid model", {768, 512, 1152, true});
            scenario(hybrid_split, mixed_tok, (uint32_t)kHybrid.vocab, ", the hybrid model over two CPUs", {768, 512, 1152, true});
        }
        // The request after the job takes the same paths on a split as on one CPU, so it runs on one CPU of each model alone.
        if (shaped)
            for (const bool late : {false, true}) {
                production(one, tok, (uint32_t)kCpu.vocab, "", late);
                production(hybrid, mixed_tok, (uint32_t)kHybrid.vocab, ", the hybrid model", late);
                if (late) continue;
                production(split, deep_tok, (uint32_t)kSplit.vocab, ", a split over two CPUs", late);
                production(hybrid_split, mixed_tok, (uint32_t)kHybrid.vocab, ", the hybrid model over two CPUs", late);
            }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    std::cout << "cache-scenario: " << checks << " checks passed\n";
    return 0;
}
