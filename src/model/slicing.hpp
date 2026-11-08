#pragma once
// How a prompt is cut into slices on a pipelined split: each slice the first of the cut of what remains that the stages' measured cost lines finish first (docs/src/model-slicing.md).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <iterator>
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

// The ratios between a cut's successive slices: 1 is the equal cut, below 1 slices shrink toward the drain and above 1 they grow out of the fill.
inline constexpr double kSliceRatios[] = {0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2};
// The first slices tried around the best equal cut's, a doubling apart in this many steps (docs/src/model-slicing.md).
inline constexpr size_t kSliceSteps = 16;

// A cut of a prompt: its first slice's rows and the ratio of each slice to the one before, at most the cap and the last the remainder; ratio 0 where the cut was not priced and each slice is chosen again.
struct SliceCut {
    size_t first = 0;
    double ratio = 0;
    // The slice after one of `rows` rows planned at `size` rows, given the `left` rows and the cap; `size` moves on by the ratio.
    size_t next(double& size, size_t left, size_t cap) const { return std::min({left, cap, std::max<size_t>(1, (size_t)(size *= ratio))}); }
};

// The cut of a prompt's `rows` rows from `start` that the stages' `lines(band)` finish first after their `backlog`, of the equal cuts and the kSliceRatios cuts near the best one, else the 2S cut's first slice.
// A finite `increment_ms` bounds how much later than the backlog a cut's first slice leaves the last stage, else a slice of kSliceFloor rows (docs/src/model-slicing.md).
template <class Lines>
SliceCut prompt_cut(size_t start, size_t rows, size_t stages, const Lines& lines, size_t cap, std::vector<double> backlog = {},
                    double increment_ms = std::numeric_limits<double>::infinity()) {
    cap = std::max<size_t>(1, cap);
    if (!rows) return {};
    if (stages < 2) return {std::min(rows, cap), 0};
    const auto equal = [&](size_t n) { return (rows + n - 1) / n; };
    const size_t lo = depth_band(start);
    std::vector<std::vector<StageLine>> bands(depth_band(start + rows - 1) - lo + 1);
    for (size_t b = 0; b < bands.size(); ++b)
        if ((bands[b] = lines(lo + b)).size() != stages) bands[b].clear();
    const size_t mid = depth_band(start + rows / 2) - lo;
    // A slice starting in a band without a line of its own takes the cut without prices, not a deeper band's price; a band later in the cut without one takes the line below it.
    if (bands[0].empty() || bands[mid].empty()) return {std::min({rows, cap, std::max(std::min(kSliceFloor, cap), equal(2 * stages))}), 0};
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
    // A cut's finish: first slice `r`, each after it `q` times the one before, at most the cap, the last the remainder; given up once it ends no sooner than `best`.
    // None when a finite increment_ms is passed by the first slice.
    size_t best_r = 0;
    double best = -1, best_q = 1;
    const auto price = [&](size_t r, double q) {
        std::vector<double> finish = backlog;
        double next = (double)r;
        for (size_t at = 0; at < rows; next *= q) {
            const size_t k = std::min({rows - at, cap, std::max<size_t>(1, (size_t)next)});
            pipeline_add(finish, bands[depth_band(start + at) - lo], k);
            if (!at && capped && finish.back() - backlog.back() > increment_ms) return;
            if (best >= 0 && finish.back() >= best) return;
            at += k;
        }
        best = finish.back(), best_r = r, best_q = q;
    };
    // The equal cuts first, from the fewest slices up until no cut of more can end sooner; then first slices from half to twice the best one's in kSliceSteps steps a doubling, at each ratio.
    size_t last = 0;
    for (size_t n = (rows + cap - 1) / cap; n <= rows; ++n) {
        const size_t r = equal(n);
        if (best >= 0 && least(equal(r)) >= best) break;
        if (capped && r < floor) break;
        if (r == last) continue;
        price(last = r, 1.0);
    }
    if (best < 0) return {floor, 0};
    const size_t centre = best_r, lowest = capped ? floor : 1;
    last = 0;
    for (int j = -int(kSliceSteps); j <= int(kSliceSteps); ++j) {
        const size_t r = std::clamp<size_t>((size_t)std::lround(double(centre) * std::exp2(double(j) / double(kSliceSteps))), lowest, std::min(rows, cap));
        if (r == last) continue;
        last = r;
        for (const double q : kSliceRatios) price(r, q);
    }
    return {std::min(rows, best_r), best_q};
}

// The first slice of prompt_cut's cut: the rows to read next.
template <class Lines>
size_t prompt_slice(size_t start, size_t rows, size_t stages, const Lines& lines, size_t cap, std::vector<double> backlog = {},
                    double increment_ms = std::numeric_limits<double>::infinity()) {
    return prompt_cut(start, rows, stages, lines, cap, std::move(backlog), increment_ms).first;
}

// A pipelined prefill's own prices by the band its slices start in: each stage's least-squares line through the device time each slice held it (Backend::busy_ms), the fixed costs calibrated by whole prefills.
class SlicePeriods {
public:
    // The points a stage keeps in a band, its latest: about sixteen prefills of sixteen slices, so the line follows the present costs and planning stays bounded however many prompts run.
    static constexpr size_t kBandPoints = 256;
    // The prefills a band's calibration averages over, and the share of the least fixed cost it may move a stage's, chosen by the simulation in docs/src/model-slicing.md.
    static constexpr size_t kCalibrationWindow = 4;
    static constexpr double kCalibrationShare = 0.5;

    // A prompt's slices of at most `cap` over `stages` stages, a slice crossing `stop` ending there: prompt_cut's cut at these lines, chosen again after a slice it did not price or that `stop` ended.
    // `finish` is left at each stage's predicted finish, or empty where a slice had no price; without prices the second slice is halved, so one prefill's points tell fixed cost from rows.
    std::vector<std::pair<size_t, size_t>> cut(size_t start, size_t rows, size_t stages, size_t cap, size_t stop, std::vector<double>& finish) const {
        std::vector<std::pair<size_t, size_t>> parts;
        const auto at = [&](size_t band) { return lines(band, stages); };
        finish.assign(stages, 0.0);
        bool priced = stages > 1;
        SliceCut plan;
        double size = 0;
        for (size_t i = 0; i < rows;) {
            size_t n = plan.ratio > 0 ? plan.next(size, rows - i, std::max<size_t>(1, cap)) : 0;
            if (!n) size = double(n = (plan = prompt_cut(start + i, rows - i, stages, at, cap, finish)).first);
            const auto l = stages > 1 ? at(depth_band(start + i)) : std::vector<StageLine>{};
            if (stages > 1 && l.empty() && parts.size() == 1) n = std::max<size_t>(1, n / 2);
            if (stop > i && stop < i + n) n = stop - i, plan.ratio = 0;
            if (!l.empty()) pipeline_add(finish, l, n);
            priced = priced && !l.empty();
            parts.push_back({i, n});
            i += n;
        }
        if (!priced) finish.clear();
        return parts;
    }

    // A prefill's slices, each stage's busy time on each (negative where unmeasured), and its wall time against `finish`, its cut's prediction, which calibrates the fixed costs the busy times leave out.
    // Each band adds the mean of its last kCalibrationWindow samples, each the residual over the finish's change for one more ms of every fixed cost, bounded by kCalibrationShare of the least (docs/src/model-slicing.md).
    void observe(const std::vector<std::pair<size_t, size_t>>& slices, const std::vector<std::vector<double>>& busy, double wall_ms, const std::vector<double>& finish) {
        const size_t n = std::min(slices.size(), busy.size());
        std::vector<size_t> seen;
        for (size_t c = 0; c < n; ++c) {
            const size_t band = depth_band(slices[c].first);
            seen.push_back(band);
            auto& stages = points_[band];
            if (stages.size() != busy[c].size()) stages.assign(busy[c].size(), {});
            for (size_t s = 0; s < stages.size(); ++s) {
                if (!(busy[c][s] > 0)) continue;
                if (stages[s].size() == kBandPoints) stages[s].erase(stages[s].begin());
                stages[s].push_back({(double)slices[c].second, busy[c][s], 1.0, 1.0});
            }
        }
        if (finish.empty() || !n) return;
        std::vector<double> at(finish.size(), 0.0), more = at;
        for (size_t c = 0; c < n; ++c) {
            auto l = lines(depth_band(slices[c].first), finish.size());
            if (l.empty()) return;
            pipeline_add(at, l, slices[c].second);
            for (StageLine& x : l) x.fixed_ms += 1;
            pipeline_add(more, l, slices[c].second);
        }
        const double cells = more.back() - at.back();
        if (!(cells > 0)) return;
        std::sort(seen.begin(), seen.end());
        seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
        for (size_t b : seen) {
            auto& [mean, samples] = extra_[b];
            samples.push_back(mean + (wall_ms - finish.back()) / cells);
            if (samples.size() > kCalibrationWindow) samples.erase(samples.begin());
            mean = 0;
            for (const double x : samples) mean += x / double(samples.size());
            const auto raw = fit(b, finish.size());
            double least = std::numeric_limits<double>::infinity();
            for (const StageLine& l : raw) least = std::min(least, l.fixed_ms);
            if (!raw.empty()) mean = std::clamp(mean, -kCalibrationShare * least, kCalibrationShare * least);
        }
    }

    // Each stage's line in `band`, else in the nearest band below where every stage has one, with `band`'s calibration; none until then.
    std::vector<StageLine> lines(size_t band, size_t stages) const {
        auto l = fit(band, stages);
        const auto e = extra_.find(band);
        if (e != extra_.end())
            for (StageLine& x : l) x.fixed_ms = std::max(0.0, x.fixed_ms + e->second.first);
        return l;
    }

private:
    std::map<size_t, std::vector<std::vector<LinePoint>>> points_;   // each band's points, a list a stage
    std::map<size_t, std::pair<double, std::vector<double>>> extra_;   // a band's fixed-cost calibration: its bounded mean and its last samples

    std::vector<StageLine> fit(size_t band, size_t stages) const {
        for (size_t b = band + 1; b-- > 0;) {
            const auto it = points_.find(b);
            if (it == points_.end() || it->second.size() != stages) continue;
            std::vector<StageLine> l;
            for (const auto& points : it->second) {
                const auto line = fit_line(points);
                if (!line) break;
                l.push_back(*line);
            }
            if (l.size() == stages) return l;
        }
        return {};
    }
};

} // namespace infer
