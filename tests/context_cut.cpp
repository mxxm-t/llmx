// The cut of a prompt that does not fit its context, alone (server/context_cut.hpp): the index of cut points and the rows or messages a cut drops, with no model, tokenizer or template.
#include <cstdint>
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
        // Room for 40 rows only: one step leaves 65 and a second would pass the prompt's end, by markers and by rows alike, so nothing fits and the cut says so.
        c = server::cut_rows(points, ids.size(), 40, 60);
        require(!c.at_marker && c.from >= ids.size(), "a prompt whose last step does not fit was reported as cut to fit");
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
        require(server::first_kept(at, starts, 1, 205, 150, 60) == 5, "the first window a step past the system message does not start at the second user message");
        require(server::first_kept(at, starts, 1, 205, 95, 60) == 5 && server::first_kept(at, starts, 1, 205, 94, 60) == 7, "a window of exactly the room was not taken, or one a row over it was");
        require(server::first_kept(at, starts, 1, 215, 150, 60) == 5, "a conversation ten rows longer was cut at another message");
        require(server::first_kept(at, starts, 1, 205, 34, 60) == at.size(), "a window past the room was taken");
        require(server::first_kept(at, starts, 0, 205, 150, 60) == 5, "a conversation without a system message kept its first message as one");
        std::cout << "context-cut: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "context-cut: FAIL: " << e.what() << "\n";
        return 1;
    }
}
