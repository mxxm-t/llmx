#pragma once
// How a prompt is cut into slices on a pipelined split: each slice the first of the cut of what remains that the stages' measured cost lines finish first (docs/src/model-slicing.md).

#include <algorithm>
#include <array>
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
// The fewest layer groups of a paced stage queued ahead on its devices (Model::pace_pass_stage): a replies' pass posted to the stage goes in behind at least that many, and the service rule caps a chunk's rows by them (pipeline_layout).
inline constexpr size_t kQueuedGroups = 2;
// The wait a reply's step within `gap_ms` leaves at each of `stages` stages past its `passes` replies' passes of `own_ms`: the device time a chunk's stage may hold queued ahead of a replies' pass there.
// The service rule cuts a chunk's groups by it (pipeline_layout) and a paced stage caps its queue by it (Model::pace_pass_stage), so the two price one wait (docs/src/model-slicing.md).
inline double stage_wait_ms(double gap_ms, size_t passes, double own_ms, size_t stages) {
    return std::max(0.0, (gap_ms - (double)passes * own_ms) / (double)std::max<size_t>(1, stages));
}
// The groups a paced stage keeps queued at `group_ms` a group: device time to cover the host's recording of a replies' stage there, `reply_ms`, and one group more, at least kQueuedGroups.
// Never more than `wait_ms` of device time, the stage's wait (stage_wait_ms), and never fewer than one group (docs/src/model-slicing.md).
inline size_t queued_groups(double reply_ms, double group_ms, double wait_ms) {
    if (!(group_ms > 0)) return kQueuedGroups;
    size_t groups = kQueuedGroups;
    if (reply_ms > 0) groups = std::max(groups, (size_t)std::min(1e6, std::ceil(reply_ms / group_ms) + 1));
    if (wait_ms < std::numeric_limits<double>::infinity()) groups = std::min(groups, (size_t)std::max(1.0, std::min(1e6, std::floor(wait_ms / group_ms))));
    return groups;
}
// The host's recording of a replies' stage a paced stage covers (Model::reply_host_ms): the median of the first `n` of the last recordings `ms`, so one recording held by a wait moves no queue.
inline double recording_median(std::array<double, 8> ms, size_t n) {
    n = std::min(n, ms.size());
    if (!n) return 0;
    std::nth_element(ms.begin(), ms.begin() + (std::ptrdiff_t)(n / 2), ms.begin() + (std::ptrdiff_t)n);
    return ms[n / 2];
}
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

// Each stage's finish once slices of `rows[c]` rows at lines `lines[c]` run from `finish`, where stage s records on device `device[s]` and a device runs several stages in turn (stage rounds).
// A stage takes its slices in order, each after the stage before it, and a device free takes its oldest ready slice (Model::run_chunks); with a device a stage it is pipeline_add slice by slice.
inline void pipeline_rounds(std::vector<double>& finish, const std::vector<size_t>& device, const std::vector<const std::vector<StageLine>*>& lines, const std::vector<size_t>& rows) {
    const size_t S = finish.size(), M = rows.size();
    if (!S || !M) return;
    std::vector<double> free(*std::max_element(device.begin(), device.end()) + 1, 0.0);
    for (size_t s = 0; s < S; ++s) free[device[s]] = std::max(free[device[s]], finish[s]);
    std::vector<size_t> next(S, 0);
    std::vector<std::vector<double>> done(S, std::vector<double>(M, 0.0));
    for (size_t left = M * S; left; --left) {
        // The slice that starts first, the oldest of those starting together.
        size_t pick = S, pc = 0;
        double first = 0;
        for (size_t s = 0; s < S; ++s) {
            const size_t c = next[s];
            if (c >= M || (s && next[s - 1] <= c)) continue;
            const double start = std::max({finish[s], s ? done[s - 1][c] : 0.0, free[device[s]]});
            if (pick == S || start < first || (start == first && c < pc)) pick = s, pc = c, first = start;
        }
        const double end = first + (*lines[pc])[pick].at(rows[pc]);
        done[pick][pc] = finish[pick] = free[device[pick]] = end;
        ++next[pick];
    }
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
                    double increment_ms = std::numeric_limits<double>::infinity(), const std::vector<size_t>& device = {}) {
    cap = std::max<size_t>(1, cap);
    if (!rows) return {};
    if (stages < 2) return {std::min(rows, cap), 0};
    const auto equal = [&](size_t n) { return (rows + n - 1) / n; };
    // Where a device runs several stages its slices interleave, so a cut is priced whole (pipeline_rounds); the stage rounds' prefill never passes a finite increment_ms.
    const bool rounds = !device.empty();
    const size_t lo = depth_band(start);
    std::vector<std::vector<StageLine>> bands(depth_band(start + rows - 1) - lo + 1);
    for (size_t b = 0; b < bands.size(); ++b)
        if ((bands[b] = lines(lo + b)).size() != stages) bands[b].clear();
    const size_t mid = depth_band(start + rows / 2) - lo;
    // A slice starting in a band without a line of its own takes the cut without prices, not a deeper band's price; a band later in the cut without one takes the line below it.
    const size_t paced = rounds ? *std::max_element(device.begin(), device.end()) + 1 : stages;   // what the pipeline's depth is on, its devices where they run several stages
    if (bands[0].empty() || bands[mid].empty()) return {std::min({rows, cap, std::max(std::min(kSliceFloor, cap), equal(2 * paced))}), 0};
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
        std::vector<const std::vector<StageLine>*> at_lines;
        std::vector<size_t> sizes;
        double next = (double)r;
        for (size_t at = 0; at < rows; next *= q) {
            const size_t k = std::min({rows - at, cap, std::max<size_t>(1, (size_t)next)});
            if (rounds) {
                at_lines.push_back(&bands[depth_band(start + at) - lo]);
                sizes.push_back(k);
                at += k;
                continue;
            }
            pipeline_add(finish, bands[depth_band(start + at) - lo], k);
            if (!at && capped && finish.back() - backlog.back() > increment_ms) return;
            if (best >= 0 && finish.back() >= best) return;
            at += k;
        }
        if (rounds) pipeline_rounds(finish, device, at_lines, sizes);
        if (best >= 0 && finish.back() >= best) return;
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
        std::vector<std::vector<StageLine>> held;
        for (size_t i = 0; i < rows;) {
            size_t n = plan.ratio > 0 ? plan.next(size, rows - i, std::max<size_t>(1, cap)) : 0;
            if (!n) {
                // Under rounds `finish` is priced whole at the end, so a later cut starts from the slices taken so far priced together, or from an empty pipeline where one had no price.
                std::vector<double> backlog = finish;
                if (rounds_ && held.size() == parts.size()) predict(backlog, held, parts);
                size = double(n = (plan = prompt_cut(start + i, rows - i, stages, at, cap, std::move(backlog), std::numeric_limits<double>::infinity(), device_)).first);
            }
            const auto l = stages > 1 ? at(depth_band(start + i)) : std::vector<StageLine>{};
            if (stages > 1 && l.empty() && parts.size() == 1) n = std::max<size_t>(1, n / 2);
            if (stop > i && stop < i + n) n = stop - i, plan.ratio = 0;
            if (!l.empty() && rounds_) held.push_back(l);
            else if (!l.empty()) pipeline_add(finish, l, n);
            priced = priced && !l.empty();
            parts.push_back({i, n});
            i += n;
        }
        if (!priced) finish.clear();
        else if (rounds_) predict(finish, held, parts);
        return parts;
    }
    // The stages' devices where a device runs several stages (Model::stage_devices), so a cut's slices are priced as they interleave there (pipeline_rounds).
    void set_devices(std::vector<size_t> device) {
        std::vector<size_t> sorted = device;
        std::sort(sorted.begin(), sorted.end());
        rounds_ = std::unique(sorted.begin(), sorted.end()) != sorted.end();
        device_ = rounds_ ? std::move(device) : std::vector<size_t>{};
    }

    // A prefill's slices, each stage's busy time on each (negative where unmeasured), and its wall time against `finish`, its cut's prediction, which calibrates the fixed costs the busy times leave out.
    // Each band adds the mean of its last kCalibrationWindow samples, each the residual over the finish's change for one more ms of every fixed cost, bounded by kCalibrationShare of the least (docs/src/model-slicing.md).
    void observe(const std::vector<std::pair<size_t, size_t>>& slices, const std::vector<std::vector<double>>& busy, double wall_ms, const std::vector<double>& finish) {
        const size_t n = std::min(slices.size(), busy.size()), prefill = prefills_++;
        std::vector<size_t> seen;
        for (size_t c = 0; c < n; ++c) {
            const size_t band = depth_band(slices[c].first);
            seen.push_back(band);
            auto& stages = points_[band];
            if (stages.size() != busy[c].size()) stages.assign(busy[c].size(), {});
            for (size_t s = 0; s < stages.size(); ++s) {
                if (!(busy[c][s] > 0)) continue;
                if (stages[s].size() == kBandPoints) stages[s].erase(stages[s].begin());
                stages[s].push_back({{(double)slices[c].second, busy[c][s], 1.0, 1.0}, prefill});
            }
        }
        if (!n) return;
        std::sort(seen.begin(), seen.end());
        seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
        if (shed(busy.front().size(), seen, prefill) || finish.empty()) return;
        std::vector<double> at(finish.size(), 0.0), more = at;
        std::vector<std::vector<StageLine>> l(n), plus(n);
        for (size_t c = 0; c < n; ++c) {
            l[c] = plus[c] = lines(depth_band(slices[c].first), finish.size());
            if (l[c].empty()) return;
            for (StageLine& x : plus[c]) x.fixed_ms += 1;
        }
        const std::vector<std::pair<size_t, size_t>> first(slices.begin(), slices.begin() + (std::ptrdiff_t)n);
        predict(at, l, first);
        predict(more, plus, first);
        const double cells = more.back() - at.back();
        if (!(cells > 0)) return;
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
    struct Point {
        LinePoint at;
        size_t prefill = 0;   // the prefill that measured it, by observe's count
    };
    std::map<size_t, std::vector<std::vector<Point>>> points_;   // each band's points, a list a stage
    std::vector<size_t> device_;   // each stage's device where a device runs several, else empty
    bool rounds_ = false;

    // Each stage's finish from `finish` after `parts`, slice c at lines `l[c]`.
    void predict(std::vector<double>& finish, const std::vector<std::vector<StageLine>>& l, const std::vector<std::pair<size_t, size_t>>& parts) const {
        if (!rounds_) {
            for (size_t c = 0; c < parts.size(); ++c) pipeline_add(finish, l[c], parts[c].second);
            return;
        }
        std::vector<const std::vector<StageLine>*> at;
        std::vector<size_t> rows;
        for (size_t c = 0; c < parts.size(); ++c) at.push_back(&l[c]), rows.push_back(parts[c].second);
        pipeline_rounds(finish, device_, at, rows);
    }
    std::map<size_t, std::pair<double, std::vector<double>>> extra_;   // a band's fixed-cost calibration: its bounded mean and its last samples
    size_t prefills_ = 0;

    // Each stage's line in `band`, else in the nearest band below where every stage has one, through the points of prefills `since` to before `until`.
    std::vector<StageLine> fit(size_t band, size_t stages, size_t since = 0, size_t until = std::numeric_limits<size_t>::max()) const {
        for (size_t b = band + 1; b-- > 0;) {
            auto l = band_fit(b, stages, since, until);
            if (!l.empty()) return l;
        }
        return {};
    }
    // Each stage's line through `band`'s own points of prefills `since` to before `until`; none unless every stage has one.
    std::vector<StageLine> band_fit(size_t band, size_t stages, size_t since, size_t until) const {
        const auto it = points_.find(band);
        if (it == points_.end() || it->second.size() != stages) return {};
        std::vector<StageLine> l;
        std::vector<LinePoint> kept;
        for (const auto& points : it->second) {
            kept.clear();
            for (const Point& p : points)
                if (p.prefill >= since && p.prefill < until) kept.push_back(p.at);
            const auto line = fit_line(kept);
            if (!line) return {};
            l.push_back(*line);
        }
        return l;
    }

    // Drops the points of the oldest prefills in the `seen` bands while they keep a band from a fit, or their own lines miss `prefill`'s points, as a cold first prefill's do.
    // A stage misses when its mean time on `prefill`'s slices is off by more than kCalibrationShare of the band's least fixed cost, more than the calibration absorbs; dropping restarts the bands' calibration (docs/src/model-slicing.md).
    bool shed(size_t stages, const std::vector<size_t>& seen, size_t prefill) {
        std::vector<size_t> ids;
        for (const size_t b : seen)
            for (const auto& points : points_[b])
                for (const Point& p : points) ids.push_back(p.prefill);
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        const auto fits = [&](size_t since) {
            return std::all_of(seen.begin(), seen.end(), [&](size_t b) { return !fit(b, stages, since).empty(); });
        };
        const auto meets = [&](size_t since) {
            for (const size_t b : seen) {
                const auto l = band_fit(b, stages, since, prefill);
                if (l.empty()) continue;
                double least = std::numeric_limits<double>::infinity();
                for (const StageLine& x : l) least = std::min(least, x.fixed_ms);
                for (size_t s = 0; s < stages; ++s) {
                    double off = 0, count = 0;
                    for (const Point& p : points_[b][s])
                        if (p.prefill == prefill) off += p.at.ms - l[s].fixed_ms * p.at.slices - l[s].row_ms * p.at.rows, count += p.at.slices;
                    if (count > 0 && std::fabs(off / count) > kCalibrationShare * least) return false;
                }
            }
            return true;
        };
        for (const size_t since : ids) {
            if (!fits(since) || (since < prefill && !meets(since))) continue;
            if (since == ids.front()) return false;
            for (const size_t b : seen) {
                for (auto& points : points_[b]) points.erase(std::remove_if(points.begin(), points.end(), [&](const Point& p) { return p.prefill < since; }), points.end());
                extra_.erase(b);
            }
            return true;
        }
        return false;
    }
};

} // namespace infer
