// The cut of a prompt that does not fit its context, alone (server/context_cut.hpp): the index of cut points and the rows or messages a cut drops, with no model, tokenizer or template.
#include <algorithm>
#include <cstdint>
#include <optional>
#include <random>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "server/context_cut.hpp"

namespace {

size_t checks = 0;
void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}

// first_kept over ends already known.
size_t kept(const std::vector<size_t>& at, const std::vector<char>& starts, size_t lead, size_t n, size_t fit, size_t step) {
    server::MessageEnds ends(at);
    return server::first_kept(ends, starts, lead, n, fit, step);
}

// The rule first_kept keeps, written out as a walk over every message's end: the first turn start a whole number of steps past the leading messages that leaves at most `fit` rows, else the first that leaves at most `fit`.
size_t walked(const std::vector<size_t>& ends, const std::vector<char>& starts, size_t lead, size_t n, size_t fit, size_t step) {
    const size_t head = lead ? ends[lead - 1] : 0;
    for (size_t j = 1; step && head + j * step < n; ++j) {
        size_t k = lead + 1;
        while (k < ends.size() && !(starts[k] && ends[k - 1] >= head + j * step)) ++k;
        if (k == ends.size()) break;
        if (head + (n - ends[k - 1]) <= fit) return k;
    }
    for (size_t k = lead + 1; k < ends.size(); ++k)
        if (starts[k] && head + (n - ends[k - 1]) <= fit) return k;
    return ends.size();
}

// The most ends first_kept may ask for over `count` messages: the head, two searches by halving and the turn start before the first that fits.
size_t ask_bound(size_t count) {
    size_t halvings = 0;
    while ((size_t(1) << halvings) < count + 1) ++halvings;
    return 2 * (halvings + 1) + 2;
}

}   // namespace

int main() {
    try {
        using server::RowCut;
        const auto ends = [](uint32_t id) { return id == 9; };
        // Five turns of 10, 30, 30, 30 and 20 rows, each ending in the end token, and 5 rows of an open turn.
        std::vector<uint32_t> ids;
        for (size_t turn : {10, 30, 30, 30, 20}) {
            ids.insert(ids.end(), turn - 1, 1);
            ids.push_back(9);
        }
        ids.insert(ids.end(), 5, 2);
        const std::vector<size_t> points = server::cut_points(ids.data(), ids.size(), ends);
        require(points == std::vector<size_t>({10, 40, 70, 100, 120}), "the cut points are not the rows after the five end tokens");
        require(server::cut_points(ids.data(), 9, ends).empty(), "a prompt without an end token has a cut point");

        // A prompt that fits is not cut.
        RowCut c = server::cut_rows(points, ids.size(), 125, 60);
        require(c.from == c.lead && c.lead == 0 && !c.at_marker, "a prompt that fits was cut");
        // A limit of 120 rows, steps of 60: the leading 10 stay and the first cut point a step or more past them is 70, which leaves 10 + 55.
        c = server::cut_rows(points, ids.size(), 119, 60);
        require(c.lead == 10 && c.from == 70 && c.at_marker, "a cut by one step did not keep the leading turn and drop to the first turn end a step past it");
        // The same prompt five rows longer is cut at the same place: the cut does not move with every turn.
        std::vector<uint32_t> longer = ids;
        longer.insert(longer.end(), 5, 2);
        c = server::cut_rows(server::cut_points(longer.data(), longer.size(), ends), longer.size(), 119, 60);
        require(c.lead == 10 && c.from == 70 && c.at_marker, "a prompt a few rows longer was cut somewhere else");
        // Room for 40 rows only: one step leaves 65 and a second would pass the prompt's end, so no step serves, and the first turn end after the lead that fits is taken, 100, which leaves 10 + 25.
        c = server::cut_rows(points, ids.size(), 40, 60);
        require(c.at_marker && c.lead == 10 && c.from == 100, "a prompt whose one step leaves too much was not cut at its first turn end that fits");
        // Room for 14: the last turn end leaves 10 + 5, so nothing fits, by markers and by rows alike, and the cut says so.
        c = server::cut_rows(points, ids.size(), 14, 60);
        require(!c.at_marker && c.from >= ids.size(), "a prompt whose newest rows alone do not fit was reported as cut to fit");
        // Room for 66 with steps of 50 and no cut point a step past the lead that is not the end: by rows, the first 25 and a step dropped.
        c = server::cut_rows({10, 20}, 125, 80, 50);
        require(!c.at_marker && c.lead == 25 && c.from == 75 && c.lead + (125 - c.from) == 75, "a cut no marker allows was not made by a whole step of rows after the fixed lead");
        c = server::cut_rows(points, ids.size(), 70, 30);
        require(c.at_marker && c.lead == 10 && c.from == 70, "steps of 30 rows did not go on to the second step, the first leaving 95 rows");
        // No end token at all: the fixed lead and whole steps after it.
        c = server::cut_rows({}, 1000, 599, 300);
        require(c.lead == server::kCutLead && c.from == server::kCutLead + 600 && !c.at_marker, "a prompt without cut points was not cut by two steps after its first 64 rows");
        // A leading block longer than a step is not kept whole.
        c = server::cut_rows({400}, 1000, 800, 300);
        require(!c.at_marker && c.lead == server::kCutLead && c.from == server::kCutLead + 300, "a leading block past a step was kept");

        // Messages: a system message, then user, assistant with a tool call, tool, assistant, user, assistant, user; the rows each ends at.
        const std::vector<size_t> at{10, 40, 70, 100, 120, 150, 180, 200};
        const std::vector<char> starts{0, 1, 0, 0, 0, 1, 0, 1};
        // Steps of 60 rows: the first message a window may start at whose predecessor ends 60 rows or more past the system message is the second user message, past the tool call and its result.
        require(kept(at, starts, 1, 205, 150, 60) == 5, "the first window a step past the system message does not start at the second user message");
        require(kept(at, starts, 1, 205, 95, 60) == 5 && kept(at, starts, 1, 205, 94, 60) == 7, "a window of exactly the room was not taken, or one a row over it was");
        require(kept(at, starts, 1, 215, 150, 60) == 5, "a conversation ten rows longer was cut at another message");
        require(kept(at, starts, 1, 205, 34, 60) == at.size(), "a window past the room was taken");
        require(kept(at, starts, 0, 205, 150, 60) == 5, "a conversation without a system message kept its first message as one");

        // A reply's room: at a limit of 4096 an uncapped prompt may take 3584 rows, and a cut drops in steps of 2048.
        {
            const size_t limit = 4096, room = limit - limit / 8, step = limit / server::kCutTo;
            // A system message of 100 rows and user and assistant turns of 100 rows each after it.
            const auto chat = [](size_t head, size_t n) {
                std::pair<std::vector<size_t>, std::vector<char>> c{{head}, {0}};
                for (size_t at = head + 100, k = 1; at <= n; at += 100, ++k) {
                    c.first.push_back(at);
                    c.second.push_back(k % 2);
                }
                return c;
            };
            const auto small = chat(100, 3700);
            // The first user start a step or more past the head: its predecessor ends at 2300, so the window is 1400 rows and the head.
            const size_t k = kept(small.first, small.second, 1, 3700, room, step);
            require(k < small.first.size() && small.second[k] && small.first[k - 1] == 2300, "a prompt past its room was not cut at the first user start a step past the head");
            // A head within a step of the prompt's end leaves no whole step: the cut is the first user start after it that leaves at most the room, not a refusal.
            const auto large = chat(1600, 3700);
            const size_t j = kept(large.first, large.second, 1, 3700, room, step);
            require(j < large.first.size() && large.second[j] && 1600 + (3700 - large.first[j - 1]) <= room && large.first[j - 1] == 1800,
                    "a system message within a step of the prompt's end was refused, or cut later than its first user start that fits");
            // A step with a user start whose window does not fit, and the next step past the prompt's end: the first user start after it that fits is taken, not a refusal.
            const auto late = chat(2048, 5848);
            const size_t l = kept(late.first, late.second, 1, 5848, room, step);
            require(l == 25 && 2048 + (5848 - late.first[24]) == 3448, "a prompt whose one step leaves too much was refused, or cut elsewhere than its first user start that fits: " + std::to_string(l));
            // A tool call and its result between two user starts: the first message that would fit is the tool result, and the window starts at the user message after it.
            {
                const std::vector<size_t> at{1600, 1700, 1800, 1900, 2000, 2100, 3700};
                const std::vector<char> turn{0, 1, 0, 0, 0, 1, 0};
                // From the tool result at index 3 the window would be 1600 + 1900 = 3500 rows, which fits, but it is no turn start; index 5 is.
                require(1600 + (3700 - at[2]) <= room && kept(at, turn, 1, 3700, room, step) == 5, "a window was started inside a tool call's messages");
            }
            // A head that leaves the newest turn no room is still refused.
            const auto full = chat(3600, 3900);
            require(kept(full.first, full.second, 1, 3900, room, step) == full.first.size(), "a system message past the room was cut");
            // The text route: a leading block of 1600 rows, cut points every 100 after it; the rows after it go up to the first cut point that leaves at most the room, at a marker.
            std::vector<size_t> points;
            for (size_t at = 1600; at < 3700; at += 100) points.push_back(at);
            const RowCut c = server::cut_rows(points, 3700, room, step);
            require(c.at_marker && c.lead == 1600 && c.from == 1800, "a text prompt whose leading block is within a step of its end lost that block");
        }
        // Random conversations: first_kept gives the walk's message, asking for few ends, each once.
        {
            std::mt19937 rng(7);
            for (int trial = 0; trial < 20000; ++trial) {
                const size_t count = 2 + rng() % 60;
                std::vector<size_t> row_at;
                std::vector<char> turn_at;
                for (size_t m = 0, row = 0; m < count; ++m) {
                    row += 1 + rng() % 40;
                    row_at.push_back(row);
                    turn_at.push_back(m && rng() % 3 == 0);
                }
                const size_t lead = rng() % 2, n = row_at.back() + rng() % 10, fit = rng() % (n + 20), step = rng() % 4 ? 1 + rng() % (n / 2 + 1) : 0;
                size_t asked = 0;
                server::MessageEnds lazy(count, n, {}, [&](size_t k) -> std::optional<size_t> { ++asked; return row_at[k - 1]; });
                const size_t got = server::first_kept(lazy, turn_at, lead, n, fit, step), want = walked(row_at, turn_at, lead, n, fit, step);
                require(got == want, "trial " + std::to_string(trial) + ": first_kept keeps from message " + std::to_string(got) + " where the walk keeps from " + std::to_string(want));
                require(asked == lazy.renders() && asked <= ask_bound(count), "trial " + std::to_string(trial) + ": first_kept asked for " + std::to_string(asked) + " ends of " + std::to_string(count) + " messages");
            }
        }
        // A conversation of 4096 messages cut by many small steps asks for row_at most the bound, not one end a message.
        {
            const size_t count = 4096;
            std::vector<char> turn_at(count);
            for (size_t m = 1; m < count; m += 2) turn_at[m] = 1;
            size_t asked = 0;
            server::MessageEnds lazy(count, count * 100, {}, [&](size_t k) -> std::optional<size_t> { ++asked; return k * 100; });
            const size_t k = server::first_kept(lazy, turn_at, 1, count * 100, 1000, 64);
            require(k < count && turn_at[k] && count * 100 - k * 100 + 100 <= 1000 && asked <= ask_bound(count),
                    "a cut of 4096 messages kept from " + std::to_string(k) + " after asking for " + std::to_string(asked) + " ends");
        }
        // A part the template refuses, the system message alone, ends at the last end token before the next part's end, or with that part where there is none.
        {
            size_t asked = 0;
            const auto rows = [&](size_t k) -> std::optional<size_t> { ++asked; if (k == 1) return std::nullopt; return k * 50; };
            server::MessageEnds with(4, 200, {12, 30, 100, 150, 200}, rows);
            require(with(0) == 30 && with(1) == 100 && asked == 2 && with(0) == 30 && asked == 2, "the system message alone, refused, did not end at the end token before the next part's end, or was rendered twice");
            server::MessageEnds without(4, 200, {}, rows);
            require(without(0) == 100, "a refused part without an end token before the next part's end did not end with it");
        }
        std::cout << "context-cut: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "context-cut: FAIL: " << e.what() << "\n";
        return 1;
    }
}
