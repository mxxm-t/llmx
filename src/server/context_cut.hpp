#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
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
    // No whole step serves, none having a cut point or none leaving a window that fits: the rows after the leading block go up to the first cut point that leaves at most the room.
    if (!points.empty() && points.front() < n)
        for (size_t i = 1; i < points.size() && points[i] < n; ++i)
            if (points.front() + (n - points[i]) <= fit) return {points.front(), points[i], true};
    c.lead = std::min(kCutLead, step / 2);
    size_t j = 1;
    while (c.lead + j * step < n && c.lead + (n - (c.lead + j * step)) > fit) ++j;
    c.from = c.lead + j * step;
    return c;
}

// Where the messages of a conversation end in its prompt's rows, each found only when a cut asks for it and once (docs/SERVER.md, A prompt larger than the context).
// `rows(k)` gives the rows the first k messages take, or nothing where the template refuses them; such a part ends at the last of `points` before the next part's end, or with the prompt.
class MessageEnds {
public:
    using Rows = std::function<std::optional<size_t>(size_t)>;
    // Ends already known, as the end tokens give them.
    explicit MessageEnds(std::vector<size_t> known) : ends_(std::move(known)) {}
    MessageEnds(size_t messages, size_t prompt, std::vector<size_t> points, Rows rows)
        : ends_(messages, kUnknown), points_(std::move(points)), prompt_(prompt), rows_(std::move(rows)) {}

    // The row where message m ends.
    size_t operator()(size_t m) {
        if (ends_[m] != kUnknown) return ends_[m];
        ++renders_;
        if (const std::optional<size_t> r = rows_(m + 1)) return ends_[m] = *r;
        const size_t next = m + 1 < ends_.size() ? (*this)(m + 1) : prompt_;
        const auto before = std::lower_bound(points_.begin(), points_.end(), next);
        return ends_[m] = before == points_.begin() ? next : *(before - 1);
    }
    // How many parts were rendered.
    size_t renders() const { return renders_; }

private:
    static constexpr size_t kUnknown = ~size_t(0);
    std::vector<size_t> ends_, points_;
    size_t prompt_ = 0, renders_ = 0;
    Rows rows_;
};

// The first message a conversation of `n` rows keeps after its `lead` leading ones: the turn start (`starts`, so a tool call stays with its results) the fewest whole steps past them that leaves at most `fit` rows, or the count.
// The ends grow with the messages, so it asks `end` for a few found by halving, never one a message, since each it asks for may be a render.
inline size_t first_kept(MessageEnds& end, const std::vector<char>& starts, size_t lead, size_t n, size_t fit, size_t step) {
    const size_t count = starts.size(), head = lead ? end(lead - 1) : 0;
    // The first turn start after the leading messages whose predecessor ends at row `at` or later, or `count`.
    const auto first_start = [&](size_t at) {
        size_t lo = lead, hi = count;
        while (lo < hi) {
            const size_t mid = lo + (hi - lo) / 2;
            if (end(mid) >= at) hi = mid;
            else lo = mid + 1;
        }
        size_t k = std::max(lo + 1, lead + 1);
        while (k < count && !starts[k]) ++k;
        return std::min(k, count);
    };
    // A start leaves at most `fit` rows from `fits` on; the steps' first start that does is the first past the start before `fits`, if a step lands there before the prompt's end.
    const size_t need = head + n > fit ? head + n - fit : 0, fits = first_start(need);
    if (fits == count) return count;
    size_t before = count;
    for (size_t k = fits; k-- > lead + 1;)
        if (starts[k]) { before = k; break; }
    const size_t past = before < count ? end(before - 1) : 0, j = step && before < count && past >= head ? (past - head) / step + 1 : 1;
    if (step && head + j * step < n) {
        const size_t k = first_start(head + j * step);
        if (k < count) return k;
    }
    // No whole step serves, none having a turn start or none leaving a window that fits: the first turn start after the leading messages that leaves at most the room is kept from.
    return fits;
}

} // namespace server
