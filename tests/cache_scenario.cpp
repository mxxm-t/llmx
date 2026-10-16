// The cache and context plan's scenario (docs/CACHE-PLAN.md, section 5): one scripted conversation at production's shape through the scheduler, each turn's rows classified against a ledger the test keeps itself.
// Counts only, no clock; an expectation the plan has not closed yet is recorded as broken with its narrow witness, and one that improves unexpectedly fails until its entry is updated.
#include <algorithm>
#include <atomic>
#include <iostream>
#include <mutex>

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

// The rows of a prompt's shortfall [reused, expected) that the losses reported from `since` on explain: a loss counts only where its history is the prompt's own over the rows they share, and its rows or its state lie in the shortfall.
// So another history's loss, one of other rows, or one reported before the restore point was last prepared explains nothing.
size_t explained(const std::vector<uint32_t>& prompt, size_t reused, size_t expected, const std::vector<Loss>& losses, size_t since = 0) {
    std::vector<char> covered(expected > reused ? expected - reused : 0, 0);
    for (size_t i = since; i < losses.size(); ++i) {
        const Loss& l = losses[i];
        // A loss of a longer history is compared over the rows of it the prompt holds up to what was expected.
        const size_t n = std::min({l.tokens, l.history.size(), expected, prompt.size()});
        if (!std::equal(l.history.begin(), l.history.begin() + (std::ptrdiff_t)n, prompt.begin())) continue;
        const bool state = l.kind == server::Move::boundary_dropped || l.kind == server::Move::boundary_lost;
        // A state lost at a position leaves the rows up to it without a restore point; rows lost are the rows themselves.
        const size_t from = state ? reused : std::max(l.first, reused), to = std::min(l.tokens, expected);
        if (state && (l.tokens <= reused || l.tokens > expected)) continue;
        for (size_t k = from; k < to; ++k) covered[k - reused] = 1;
    }
    return (size_t)std::count(covered.begin(), covered.end(), (char)1);
}

// The classifier held to its own rule on made-up losses: the right loss explains a shortfall, and another history's, other rows' or none does not.
void attribution() {
    const std::vector<uint32_t> mine = prompt_of(3, 1200, 64), theirs = prompt_of(4, 1200, 64);
    const Loss right{server::Move::job_lost, 0, 896, mine}, other{server::Move::job_lost, 0, 896, theirs}, elsewhere{server::Move::consumed_lost, 896, 1152, mine}, state{server::Move::boundary_dropped, 768, 768, mine};
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
// What a placement shows today where the placements differ: the jobs that gave way below half the pool, and what the two edits reused.
struct Witness {
    size_t gave_way, edit_reused, older_reused, after_edit;
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
    Handle submit(const Req& r, size_t message = std::numeric_limits<size_t>::max()) {
        std::lock_guard<std::mutex> lk(handles);
        known.push_back(sched.submit(r.prompt, params_of(r), std::numeric_limits<size_t>::max(), message));
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
        {
            std::lock_guard<std::mutex> lk(w.losing);
            row.evicted = explained(r.prompt, row.reused, row.expected, w.losses, c.since);
        }
        row.unexplained = shortfall - row.evicted;
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
    Ledger book{on_what, state};
    std::vector<Row>& report = book.report;
    Conversation chat;
    chat.seed = 7;
    chat.history = prompt_of(5, 400, v);
    uint32_t other_seed = 33;
    size_t cuts = 0, follows = 0, since1 = 0, since3 = 0;
    std::vector<uint32_t> after3, after1;
    std::vector<Expectation> table;
    const auto submit = [&](const Req& r, size_t message) { return watch.submit(r, message); };
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
            ++follows;
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
        table.push_back({"a job gives way only to a request that cannot be admitted beside it", "P7", false, !a_gave_way});
        require(a_gave_way == w.gave_way, "leg A's witness: " + std::to_string(a_gave_way) + " jobs gave way below half the pool, against " + std::to_string(w.gave_way));
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
            ++follows;
            within_a_minute([&] { std::lock_guard<std::mutex> lk(handles); return after != nullptr; }, "leg B: the arrivals at the job's pass and at the other request's");
            drain(*other);
            h = after;
        }
        // Quiescence: the job that gave way has started again and finished, so what follows meets no job; then one plain turn.
        within_a_minute([&] { return sched.stats().reprefills == follows; }, "leg B: the job that gave way finished");
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
        table.push_back({"past half the pool a turn's donor stays beside its job", "P3", false, !report[5].consumed});
        table.push_back({"a job that gives way to another conversation's request loses no row a later turn needs", "P1, P7", false, !b_evicted});
        require(b_gave_way == 1 && b_evicted == 896 && !report[5].reused, "leg B's witness: " + std::to_string(b_gave_way) + " jobs gave way, " + std::to_string(report[5].reused) + " rows reused and " + std::to_string(b_evicted) + " evicted, against 1, 0 and 896");
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
            table.push_back({"an edit reuses up to the last unchanged message", "P10", !state, report.back().reused == after3.size() / bt * bt});
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
            if (r.prompt.size() > room) {
                const size_t at = book.cut(r, room);
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
            const Handle h = submit(r, r.prompt.size() - kUser);
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
// The host tier holds less than one history and a disk tier keeps files, with no idle moment for its idle writes; the chat passes the context and is cut, and the server restarts over its kept files.
void production(const Make& make, const bpe::Tokenizer& tok, uint32_t v, const std::string& on_what, const Witness& w) {
    const size_t bt = kBlock;
    DiskRoot disk("cache-scenario");
    server::DiskOptions options = disk.options(uint64_t(1) << 30);
    options.keep = true;
    options.idle = std::chrono::hours(1);
    auto model = make(kPool, 64);
    const bool state = model->keeps_state();
    // A host tier of 1024 rows: the chat's history passes it from the first turn, the unrelated request's fits.
    const size_t host = model->host_bytes(1024);
    auto sched = std::make_unique<server::Scheduler>(*model, tok, 4, 64, 0, false, host, nullptr, 0, true, options);
    auto watch = std::make_unique<Watch>(*sched);
    std::thread runner([&] { sched->run(); });
    Ledger book{on_what + ", production's proportions", state};
    Conversation chat;
    chat.seed = 70;
    chat.history = prompt_of(5, 400, v);
    std::vector<Expectation> table;
    std::vector<size_t> cut_turns, restart_turns;
    size_t released = 0, job_lost = 0, consumed_lost = 0, not_to_host = 0, host_lost = 0, written = 0, read = 0, tasks = 0;
    uint32_t other_seed = 90;
    const auto tally = [&] {
        const Stats s = sched->stats();
        released += s.reprefill_released;
        job_lost += s.moves[(size_t)server::Move::job_lost];
        consumed_lost += s.moves[(size_t)server::Move::consumed_lost];
        not_to_host += s.moves[(size_t)server::Move::not_to_host];
        host_lost += s.moves[(size_t)server::Move::host_lost];
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
            {
                std::lock_guard<std::mutex> lk(watch->handles);
                watch->at_job_pass = [&, o, next, message, restart] {
                    std::lock_guard<std::mutex> lk2(watch->handles);
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
            }
            mark = sched->stats();
            const size_t jobs = mark.reprefills;
            sched->follow(h, chat.history, true);
            within_a_minute([&] { std::lock_guard<std::mutex> lk(watch->handles); return other != nullptr && (restart || after != nullptr); }, "production: the arrivals at the job's pass and at the other request's");
            drain(*other);
            ++tasks;
            if (cut) {
                chat.prepared = w.boundaries ? 400 / bt * bt : 0;
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
                within_a_minute([&] { return sched->stats().disk_ready; }, "production: the disk tier made again");
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
                  << " after the restart, unexplained " << unexplained << "; moves: job_lost " << job_lost << ", consumed_lost " << consumed_lost << ", not_to_host " << not_to_host << ", host_lost " << host_lost
                  << ", rows jobs released " << released << ", files written whole " << written << ", reads " << read << "\n";
        table.push_back({"past half the pool, with a host tier smaller than a history, a turn after an unrelated request loses no row its history held", "P1, P3", false, !evicted});
        table.push_back({"a job that gives way releases no row it read", "P7", false, !released});
        // Over two devices no state is kept at the system message (Witness::boundaries), so a cut turn has nothing to reuse and loses nothing.
        table.push_back({"a cut turn reuses its leading message's rows beside an unrelated request", "P5", !w.boundaries, !cut.evicted});
        // What the turn after the restart falls short by was lost before the stop; everything else it reads back.
        table.push_back({"a restart over kept files of this build costs the next turn a read, not a recompute", "P4", true, read && restarted.reused + restarted.evicted == restarted.expected});
        table.push_back({"every row a turn falls short by is a loss the server reported with its history", "P8", true, !unexplained});
        // Today each job gives way to the unrelated request and releases the chat's rows, which it alone held (S8): every plain turn from the third reads its history again, and so do the cut turn and the turn after the restart.
        const size_t plain = state ? 7296 : 5888, at_cut = w.boundaries ? 384 : 0;
        require(evicted == plain && cut.evicted == at_cut && restarted.evicted == 512 && job_lost == (state ? 7u : 6u),
                "production's witness: " + std::to_string(evicted) + ", " + std::to_string(cut.evicted) + " and " + std::to_string(restarted.evicted) + " rows evicted, " + std::to_string(job_lost) + " job losses, against " +
                    std::to_string(plain) + ", " + std::to_string(at_cut) + ", 512 and " + std::to_string(state ? 7 : 6));
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

} // namespace

int main() {
    try {
        attribution();
        const gguf::GGUFModel weights = served(kCpu);
        scenario(on(weights, [] { return cpus(1); }), bpe::Tokenizer(weights), (uint32_t)kCpu.vocab, "", {3, 768, 512, 768, true});
        production(on(weights, [] { return cpus(1); }), bpe::Tokenizer(weights), (uint32_t)kCpu.vocab, "", {3, 768, 512, 768, true});
        const gguf::GGUFModel deep = served(kSplit);
        scenario(on(deep, [] { return cpus(2); }), bpe::Tokenizer(deep), (uint32_t)kSplit.vocab, ", a split over two CPUs", {3, 768, 512, 768, true});
        production(on(deep, [] { return cpus(2); }), bpe::Tokenizer(deep), (uint32_t)kSplit.vocab, ", a split over two CPUs", {3, 768, 512, 768, true});
        const gguf::GGUFModel mixed = served_hybrid(kHybrid);
        scenario(on(mixed, [] { return cpus(1); }, 4, 6), bpe::Tokenizer(mixed), (uint32_t)kHybrid.vocab, ", the hybrid model", {1, 640, 384, 768, true});
        production(on(mixed, [] { return cpus(1); }, 4, 6), bpe::Tokenizer(mixed), (uint32_t)kHybrid.vocab, ", the hybrid model", {1, 640, 384, 768, true});
        scenario(on(mixed, [] { return cpus(2); }, 4, 6), bpe::Tokenizer(mixed), (uint32_t)kHybrid.vocab, ", the hybrid model over two CPUs", {1, 0, 0, 1152, false});
        production(on(mixed, [] { return cpus(2); }, 4, 6), bpe::Tokenizer(mixed), (uint32_t)kHybrid.vocab, ", the hybrid model over two CPUs", {1, 0, 0, 1152, false});
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    std::cout << "cache-scenario: " << checks << " checks passed\n";
    return 0;
}
