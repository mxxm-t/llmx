#pragma once
// How a prompt is cut into slices on a pipelined split: each slice the next of the equal cut of what remains that the stages' measured cost lines finish first (docs/src/model-slicing.md).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace infer {

// The depth band of rows read at history position `depth` (docs/SERVER.md, The price table): 0 below kBandTokens, then one band a doubling, so 4k to 8k tokens is band 1 and 128k to 256k band 6.
inline constexpr size_t kBandTokens = 4096;
inline size_t depth_band(size_t depth) {
    size_t b = 0;
    for (size_t d = depth / kBandTokens; d; d >>= 1) ++b;
    return b;
}

// A stage's time for one prompt slice of `rows` rows in a depth band, as a measured line: fixed_ms + rows * row_ms.
struct StageLine {
    double fixed_ms = 0, row_ms = 0;
    double at(size_t rows) const { return fixed_ms + (double)rows * row_ms; }
};

// One measured time for fit_line: the rows of the `slices` slices it took, its time and how many measures it stands for.
struct LinePoint {
    double rows = 0, ms = 0, weight = 1, slices = 1;
};
// The least-squares line ms = fixed * slices + row * rows through `points`, each weighted by its measures, each column scaled to unit size so a singular system is told by its pivot.
// None before two rows a slice are measured, or without a positive fixed cost and cost a row; a point without a positive time, weight or slice count is left out.
inline std::optional<StageLine> fit_line(const std::vector<LinePoint>& points) {
    double a = 0, b = 0, c = 0, u = 0, v = 0, fx = 0, rx = 0, first = -1;
    bool differ = false;
    for (const LinePoint& p : points)
        if (p.ms > 0 && p.weight > 0 && p.slices > 0) {
            const double per = p.rows / p.slices;
            differ = differ || (first >= 0 && per != first);
            if (first < 0) first = per;
            fx = std::max(fx, p.slices), rx = std::max(rx, std::fabs(p.rows));
        }
    if (!differ || !(rx > 0)) return std::nullopt;
    for (const LinePoint& p : points)
        if (p.ms > 0 && p.weight > 0 && p.slices > 0) {
            const double x0 = p.slices / fx, x1 = p.rows / rx;
            a += p.weight * x0 * x0, b += p.weight * x0 * x1, c += p.weight * x1 * x1, u += p.weight * x0 * p.ms, v += p.weight * x1 * p.ms;
        }
    const double det = a * c - b * b;
    if (!(det > 1e-12 * std::max(a, c) * std::max(a, c))) return std::nullopt;
    const StageLine l{(c * u - b * v) / det / fx, (a * v - b * u) / det / rx};
    if (!(l.fixed_ms > 0 && l.row_ms > 0)) return std::nullopt;
    return l;
}

// The least slice the 2S cut takes without prices, as the scheduler's lone prompts took it: a slice's fixed costs outweigh its rows below it.
inline constexpr size_t kSliceFloor = 128;

// Each stage's finish once one more slice of `rows` rows runs through the stages from `first` on: F[s] = max(F[s-1], F[s]) + t_s(rows).
inline void pipeline_add(std::vector<double>& finish, const std::vector<StageLine>& lines, size_t rows, size_t first = 0) {
    double prev = 0;
    for (size_t s = first; s < finish.size(); ++s) prev = finish[s] = std::max(prev, finish[s]) + lines[s].at(rows);
}
// The same step for a pass whose time at each stage is given, `ms[s]`.
inline void pipeline_run(std::vector<double>& finish, const std::vector<double>& ms, size_t first) {
    double prev = 0;
    for (size_t s = first; s < finish.size(); ++s) prev = finish[s] = std::max(prev, finish[s]) + ms[s];
}

// The rows of a prompt's next slice from history position `start`, at most `cap`: the next of the equal cut of the `rows` left that the stages' lines, `lines(band)`, finish first after each stage's `backlog`, else the 2S cut.
// A finite `increment_ms` bounds how much later than the backlog the next slice leaves the last stage, else a slice of kSliceFloor rows (docs/src/model-slicing.md).
template <class Lines>
size_t prompt_slice(size_t start, size_t rows, size_t stages, const Lines& lines, size_t cap, std::vector<double> backlog = {},
                    double increment_ms = std::numeric_limits<double>::infinity()) {
    cap = std::max<size_t>(1, cap);
    if (!rows) return 0;
    if (stages < 2) return std::min(rows, cap);
    const auto equal = [&](size_t n) { return (rows + n - 1) / n; };
    const size_t lo = depth_band(start);
    std::vector<std::vector<StageLine>> bands(depth_band(start + rows - 1) - lo + 1);
    for (size_t b = 0; b < bands.size(); ++b)
        if ((bands[b] = lines(lo + b)).size() != stages) bands[b].clear();
    const size_t mid = depth_band(start + rows / 2) - lo;
    // A slice starting in a band without a line of its own takes the cut without prices, not a deeper band's price; a band later in the cut without one takes the line below it.
    if (bands[0].empty() || bands[mid].empty()) return std::min({rows, cap, std::max(std::min(kSliceFloor, cap), equal(2 * stages))});
    for (size_t b = 1; b < bands.size(); ++b)
        if (bands[b].empty()) bands[b] = bands[b - 1];
    backlog.resize(stages, 0.0);
    // No cut of n slices ends before any stage has run each of them: its backlog, n of its least fixed costs and every row at its least cost a row.
    std::vector<double> fixed(stages), row(stages);
    for (size_t s = 0; s < stages; ++s) {
        fixed[s] = row[s] = std::numeric_limits<double>::infinity();
        for (const auto& b : bands) fixed[s] = std::min(fixed[s], b[s].fixed_ms), row[s] = std::min(row[s], b[s].row_ms);
    }
    const auto least = [&](size_t n) {
        double t = 0;
        for (size_t s = 0; s < stages; ++s) t = std::max(t, backlog[s] + double(n) * fixed[s] + row[s] * double(rows));
        return t;
    };
    const bool capped = increment_ms < std::numeric_limits<double>::infinity();
    const size_t floor = std::min({rows, cap, kSliceFloor});
    size_t best_r = 0, last = 0;
    double best = -1;
    for (size_t n = (rows + cap - 1) / cap; n <= rows; ++n) {
        const size_t r = equal(n);
        if (best >= 0 && least(equal(r)) >= best) break;
        if (capped && r < floor) break;
        if (r == last) continue;
        last = r;
        std::vector<double> finish = backlog;
        if (capped) {
            pipeline_add(finish, bands[0], r);
            if (finish.back() - backlog.back() > increment_ms) continue;
            finish = backlog;
        }
        for (size_t at = 0; at < rows; at += r) {
            const size_t k = std::min(r, rows - at);
            pipeline_add(finish, bands[depth_band(start + at) - lo], k);
        }
        if (best < 0 || finish.back() < best) best = finish.back(), best_r = r;
    }
    return best < 0 ? floor : std::min(rows, best_r);
}

// A pipelined prefill's own prices by the band its slices start in: a least-squares line through the times its slices left the last stage, its fixed cost calibrated by whole prefills.
class SlicePeriods {
public:
    // The points a band keeps, its latest: about sixteen prefills of sixteen slices, so the line follows the present costs and planning stays bounded however many prompts run.
    static constexpr size_t kBandPoints = 256;
    // A prefill's slices, each its history position and rows, and the time each left the last stage: every run of slices in one band is a series of points, each the slices since the run's origin, their rows and the time since then.
    // The origin is the end of the slice before the run, so the first slice, which fills the stages, is no point, nor is the last, which carries the head and the drain; a late time moves one point, not two periods.
    void observe(const std::vector<std::pair<size_t, size_t>>& slices, const std::vector<double>& done_ms) {
        const size_t n = std::min(slices.size(), done_ms.size());
        size_t band = 0, count = 0;
        double rows = 0, origin = 0;
        for (size_t c = 1; c + 1 < n; ++c) {
            const size_t b = depth_band(slices[c].first);
            if (!count || b != band) band = b, count = 0, rows = 0, origin = done_ms[c - 1];
            rows += (double)slices[c].second;
            ++count;
            if (done_ms[c] <= origin) continue;
            auto& points = points_[band];
            if (points.size() == kBandPoints) points.erase(points.begin());
            points.push_back({rows, done_ms[c] - origin, 1.0, (double)count});
        }
    }
    // A prefill of `bands.size()` slices, each starting in its band, took `residual_ms` beyond the predicted finish: each band's fixed cost becomes the running mean of what it was plus the residual a slice.
    void calibrate(const std::vector<size_t>& bands, double residual_ms) {
        if (bands.empty()) return;
        std::vector<size_t> seen(bands);
        std::sort(seen.begin(), seen.end());
        seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
        for (size_t b : seen) {
            auto& [mean, n] = extra_[b];
            const double sample = mean + residual_ms / (double)bands.size();
            mean += (sample - mean) / double(++n);
        }
    }
    // The period's line in `band`, else in the nearest band below that has one, with `band`'s calibration.
    std::optional<StageLine> line(size_t band) const {
        for (size_t b = band + 1; b-- > 0;)
            if (auto l = fit(b)) {
                const auto e = extra_.find(band);
                if (e != extra_.end() && l->fixed_ms + e->second.first > 0) l->fixed_ms += e->second.first;
                return l;
            }
        return std::nullopt;
    }

private:
    std::map<size_t, std::vector<LinePoint>> points_;     // each band's points
    std::map<size_t, std::pair<double, size_t>> extra_;   // a band's fixed-cost calibration: its running mean and samples

    std::optional<StageLine> fit(size_t b) const {
        const auto it = points_.find(b);
        return it == points_.end() ? std::nullopt : fit_line(it->second);
    }
};

} // namespace infer
