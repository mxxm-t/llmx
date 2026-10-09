#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace server {

// A prompt that does not fit its context, cut (docs/SERVER.md, A prompt larger than the context): the index of where a prompt may begin again, and which rows a cut drops.
// A cut point is a row offset, the row after a turn's end, never a text offset, so a part that is not text takes rows as any other.

// A cut drops rows in steps of the context over kCutTo, one half: the first cut point at least one step past the leading block, or two steps, and so on, the fewest steps after which the prompt fits.
// The steps are counted from the prompt's start, so a conversation that goes on is cut at the same place turn after turn; a step of f of a context C re-reads up to (1 - f) C rows to buy f C new ones, about one per new row at one half.
constexpr size_t kCutTo = 2;

// The rows a text prompt keeps at its start when no turn end marks a leading block: a fixed count, the same on every device and placement.
constexpr size_t kCutLead = 64;

// The cut points of `n` ids: the offset after each id `ends` names, the tokens the tokenizer's own metadata marks as ends.
inline std::vector<size_t> cut_points(const uint32_t* ids, size_t n, const std::function<bool(uint32_t)>& ends) {
    std::vector<size_t> points;
    for (size_t i = 0; i < n; ++i)
        if (ends(ids[i])) points.push_back(i + 1);
    return points;
}

// A cut of rows: the prompt keeps its first `lead` rows and those from `from` on; `at_marker` says both are cut points.
struct RowCut {
    size_t lead = 0, from = 0;
    bool at_marker = false;
};

// The cut of a text prompt of `n` rows to at most `fit`, in steps of `step` rows: its leading block, up to the first cut point, stays, and the rows after it go up to the first cut point a whole number of steps on, the fewest that fit.
// Where no cut point serves, the first kCutLead rows stay (fewer where a step is shorter) and whole steps of the rows after them go, the cut then not at a marker; no cut where the prompt fits, `from` past `n` where nothing fits.
inline RowCut cut_rows(const std::vector<size_t>& points, size_t n, size_t fit, size_t step) {
    RowCut c;
    if (n <= fit || !step) return c;
    if (!points.empty() && points.front() < step) {
        c.lead = points.front();
        for (size_t j = 1; c.lead + j * step < n; ++j) {
            const auto p = std::find_if(points.begin(), points.end(), [&](size_t at) { return at >= c.lead + j * step; });
            if (p == points.end() || *p >= n) break;
            if (c.lead + (n - *p) <= fit) {
                c.from = *p;
                c.at_marker = true;
                return c;
            }
        }
    }
    c.lead = std::min(kCutLead, step / 2);
    size_t j = 1;
    while (c.lead + j * step < n && c.lead + (n - (c.lead + j * step)) > fit) ++j;
    c.from = c.lead + j * step;
    return c;
}

// The first message a conversation keeps after its leading `lead` messages, for a prompt of `n` rows whose message k ends at row ends[k].
// It is the first turn start (`starts[k]`, so a tool call stays with its results) a whole number of steps or more past them, the fewest steps that leave at most `fit` rows, or the message count where none does.
inline size_t first_kept(const std::vector<size_t>& ends, const std::vector<char>& starts, size_t lead, size_t n, size_t fit, size_t step) {
    const size_t head = lead ? ends[lead - 1] : 0;
    for (size_t j = 1; step && head + j * step < n; ++j) {
        size_t k = lead + 1;
        while (k < ends.size() && !(starts[k] && ends[k - 1] >= head + j * step)) ++k;
        if (k == ends.size()) break;
        if (head + (n - ends[k - 1]) <= fit) return k;
    }
    return ends.size();
}

} // namespace server
