#pragma once
// How a prompt is cut into slices on a pipelined split: each slice the next of the equal cut of what remains that the stages' measured cost lines finish first (docs/src/model-slicing.md).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <tuple>
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

// A stage's time for one prompt slice of `rows` rows whose middle row sits at history position `depth`, as a measured line: fixed_ms + rows * (row_ms + depth_ms * depth).
struct StageLine {
    double fixed_ms = 0, row_ms = 0, depth_ms = 0;
    double at(size_t rows, size_t depth = 0) const { return fixed_ms + (double)rows * (row_ms + depth_ms * (double)depth); }
};

// One measured slice time for fit_line: its rows, its middle row's history position, its time and how many measures it stands for.
struct LinePoint {
    double rows = 0, depth = 0, ms = 0, weight = 1;
};
namespace detail {
// The first k coefficients' weighted least-squares solution, each column scaled to unit size so a singular system is told by its pivots.
inline std::optional<std::array<double, 3>> least_squares(const std::vector<std::array<double, 3>>& x, const std::vector<double>& y, const std::vector<double>& w, size_t k) {
    std::array<double, 3> scale{};
    for (const auto& p : x)
        for (size_t j = 0; j < k; ++j) scale[j] = std::max(scale[j], std::fabs(p[j]));
    double a[3][4] = {};
    for (size_t i = 0; i < x.size(); ++i)
        for (size_t j = 0; j < k; ++j) {
            if (!(scale[j] > 0)) return std::nullopt;
            for (size_t l = 0; l < k; ++l) a[j][l] += w[i] * (x[i][j] / scale[j] * x[i][l] / scale[l]);
            a[j][k] += w[i] * (x[i][j] / scale[j] * y[i]);
        }
    double top = 0;
    for (size_t j = 0; j < k; ++j) top = std::max(top, std::fabs(a[j][j]));
    for (size_t j = 0; j < k; ++j) {
        size_t p = j;
        for (size_t i = j + 1; i < k; ++i)
            if (std::fabs(a[i][j]) > std::fabs(a[p][j])) p = i;
        if (!(std::fabs(a[p][j]) > 1e-9 * top)) return std::nullopt;
        std::swap(a[p], a[j]);
        for (size_t i = 0; i < k; ++i)
            if (i != j) {
                const double f = a[i][j] / a[j][j];
                for (size_t l = j; l <= k; ++l) a[i][l] -= f * a[j][l];
            }
    }
    std::array<double, 3> c{};
    for (size_t j = 0; j < k; ++j) c[j] = a[j][k] / a[j][j] / scale[j];
    return c;
}
} // namespace detail
// The least-squares line ms = fixed + row * rows + depth_ms * rows * depth through `points`, each weighted by its measures, without the depth term where it is unknowable or negative.
// None before two row counts are measured, or without a positive fixed cost and cost a row; a point without a positive time or weight is left out.
inline std::optional<StageLine> fit_line(const std::vector<LinePoint>& points) {
    std::vector<std::array<double, 3>> x;
    std::vector<double> y, w;
    for (const LinePoint& p : points)
        if (p.ms > 0 && p.weight > 0) x.push_back({1.0, p.rows, p.rows * p.depth}), y.push_back(p.ms), w.push_back(p.weight);
    bool rows_differ = false;
    for (const auto& p : x) rows_differ = rows_differ || p[1] != x[0][1];
    if (!rows_differ) return std::nullopt;
    for (size_t k : {size_t(3), size_t(2)}) {
        const auto c = detail::least_squares(x, y, w, k);
        if (c && (k == 2 || (*c)[2] >= 0) && (*c)[0] > 0 && (*c)[1] > 0) return StageLine{(*c)[0], (*c)[1], k == 3 ? (*c)[2] : 0.0};
    }
    return std::nullopt;
}

// The least slice the 2S cut takes without prices, as the scheduler's lone prompts took it: a slice's fixed costs outweigh its rows below it.
inline constexpr size_t kSliceFloor = 128;

// Each stage's finish once one more slice of `rows` rows, its middle at `depth`, runs through the stages from `first` on: F[s] = max(F[s-1], F[s]) + t_s(rows).
inline void pipeline_add(std::vector<double>& finish, const std::vector<StageLine>& lines, size_t rows, size_t first = 0, size_t depth = 0) {
    double prev = 0;
    for (size_t s = first; s < finish.size(); ++s) prev = finish[s] = std::max(prev, finish[s]) + lines[s].at(rows, depth);
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
    if (bands[mid].empty()) return std::min({rows, cap, std::max(std::min(kSliceFloor, cap), equal(2 * stages))});
    for (size_t b = 0; b < bands.size(); ++b)
        if (bands[b].empty()) bands[b] = b ? bands[b - 1] : bands[mid];
    backlog.resize(stages, 0.0);
    // No cut of n slices ends before any stage has run each of them: its backlog, n of its least fixed costs and every row at its least cost a row.
    std::vector<double> fixed(stages), row(stages);
    for (size_t s = 0; s < stages; ++s) {
        fixed[s] = row[s] = std::numeric_limits<double>::infinity();
        for (const auto& b : bands) fixed[s] = std::min(fixed[s], b[s].fixed_ms), row[s] = std::min(row[s], b[s].row_ms + b[s].depth_ms * (double)start);
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
            pipeline_add(finish, bands[0], r, 0, start + r / 2);
            if (finish.back() - backlog.back() > increment_ms) continue;
            finish = backlog;
        }
        for (size_t at = 0; at < rows; at += r) {
            const size_t k = std::min(r, rows - at);
            pipeline_add(finish, bands[depth_band(start + at) - lo], k, 0, start + at + k / 2);
        }
        if (best < 0 || finish.back() < best) best = finish.back(), best_r = r;
    }
    return best < 0 ? floor : std::min(rows, best_r);
}

// A pipelined prefill's own prices by the band its slices start in: a least-squares line through their measured periods, its fixed cost calibrated by whole prefills.
class SlicePeriods {
public:
    // One slice's period: the band it starts in, its rows, its middle row's history position and the interval measured for it.
    void observe(size_t band, size_t rows, size_t depth, double ms) {
        if (rows && ms > 0) ms_[{band, rows, depth}] = ms;
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
    // The line through band `b`'s periods (fit_line), each one measure.
    std::optional<StageLine> fit(size_t b) const {
        std::vector<LinePoint> points;
        for (auto it = ms_.lower_bound({b, 0, 0}); it != ms_.end() && std::get<0>(it->first) == b; ++it)
            points.push_back({(double)std::get<1>(it->first), (double)std::get<2>(it->first), it->second});
        return fit_line(points);
    }

    std::map<std::tuple<size_t, size_t, size_t>, double> ms_;
    std::map<size_t, std::pair<double, size_t>> extra_;   // a band's fixed-cost calibration: its running mean and samples
};

} // namespace infer
