#pragma once
// The service record (docs/SERVER.md, The price table): each stage's measured time against its predicted one, the rows by kind and the requests' waits and gaps, for /v1/health.
// A thread of its own sees each stage retire, so neither a stage nor the round waits for a measure.
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <cmath>
#include <map>
#include <mutex>
#include <thread>
#include <vector>
#include "backends/backend.hpp"
#include "server/policy.hpp"

namespace server {

// The last kValues values of a figure and their count since start, for percentiles over the recent ones.
struct Recent {
    static constexpr size_t kValues = 1024;
    std::array<double, kValues> values{};
    uint64_t count = 0;
    void add(double v) { values[count++ % kValues] = v; }
    struct Summary {
        uint64_t count = 0;
        double p50 = 0, p95 = 0, p99 = 0, max = 0;
    };
    // The nearest-rank percentiles of the values kept, with the count since start.
    Summary summary() const {
        Summary s;
        s.count = count;
        const size_t n = (size_t)std::min<uint64_t>(count, kValues);
        if (!n) return s;
        std::vector<double> v(values.begin(), values.begin() + (std::ptrdiff_t)n);
        std::sort(v.begin(), v.end());
        const auto at = [&](double q) { return v[std::min(n - 1, (size_t)std::ceil(q * (double)n) - 1)]; };
        s.p50 = at(0.50);
        s.p95 = at(0.95);
        s.p99 = at(0.99);
        s.max = v.back();
        return s;
    }
};

// The rows passes carried since start by kind: decode entries' tokens, drafts their verifies fed, generated tokens a resume computed again, prompt rows, a job's rows read again, and the prompt and job rows by depth band.
struct RowKinds {
    uint64_t decode = 0, drafts = 0, recomputed = 0, prompt = 0, reread = 0;
    std::vector<uint64_t> by_band;
};

// One stage of one pass as the record is given it once recorded: the pass's formation order, its work and prediction, when its recording began and ended, its host waits for lookup rows and in sums, and the submission that ends it.
struct StageRun {
    uint64_t pass = 0;
    size_t stage = 0;
    PassWork work;
    double predicted_ms = 0;
    std::chrono::steady_clock::time_point begun, recorded;
    double lookup_ms = 0, sum_ms = 0;
    backend::Backend* backend = nullptr;
    backend::Ticket ticket = 0;
};

// What a stage's passes measured since start: passes, those predicted and the predictions' total and absolute error, those over their prediction, those fed in pieces, the longest stage time, the last kRecentPasses passes and the prices.
struct StageRecord {
    static constexpr size_t kRecentPasses = 8;
    uint64_t passes = 0, predicted = 0, over = 0;
    double measured_ms = 0, predicted_ms = 0, error_ms = 0, longest_ms = 0;
    // The measured time split: the host recording the stage, of it the waits for lookup rows and in tensor-group sums, and the device's time after the recording.
    double recording_ms = 0, lookup_ms = 0, sum_ms = 0, after_ms = 0;
    struct Pass {
        uint64_t pass = 0;
        double measured_ms = 0, predicted_ms = 0;
        PassWork work;
    };
    std::deque<Pass> recent;
    PriceTable prices;
};

// The record's owner: the scheduler posts each recorded stage and each request's times, the record's thread measures the stages, and the health reads a copy.
class ServiceRecord {
public:
    using Clock = std::chrono::steady_clock;
    explicit ServiceRecord(size_t stages) : stages_(stages), pending_(stages), last_done_(stages) {}
    ~ServiceRecord() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }
    ServiceRecord(const ServiceRecord&) = delete;
    ServiceRecord& operator=(const ServiceRecord&) = delete;

    // A stage recorded, on any thread that records; the record's thread starts with the first.
    void post(StageRun run) {
        {
            std::lock_guard<std::mutex> lk(m_);
            pending_[run.stage].push_back(std::move(run));
            if (!thread_.joinable()) thread_ = std::thread([this] { loop(); });
        }
        cv_.notify_all();
    }
    // The stage time stage s's prices predict for work `w`, 0 where they do not price it yet.
    double predict(size_t s, const PassWork& w) const {
        std::lock_guard<std::mutex> lk(m_);
        return stages_[s].prices.predict(w);
    }
    // Stage s's price of `rows` decode rows at depth band `band`, 0 before one is measured.
    double decode_ms(size_t s, size_t band, size_t rows) const {
        std::lock_guard<std::mutex> lk(m_);
        return stages_[s].prices.decode_ms(band, rows);
    }
    // Stage s's price of a prompt row in a slice of `rows` rows of class `cls` at band `band`, 0 before one is measured.
    double row_ms(size_t s, size_t cls, size_t band, size_t rows) const {
        std::lock_guard<std::mutex> lk(m_);
        return rows ? stages_[s].prices.chunk_ms({cls, band, rows}) / (double)rows : 0.0;
    }
    // Stage s's price of a slice of `rows` rows for the quick rule, 0 with `step` the rows to read next where it is not yet known (PriceTable::slice_ms).
    double slice_ms(size_t s, size_t cls, size_t band, size_t rows, size_t& step) const {
        std::lock_guard<std::mutex> lk(m_);
        return stages_[s].prices.slice_ms({cls, band, rows}, step);
    }
    // The rows of a slice of class `cls` at band `band` within `target_ms` of stage s, at most `most` (PriceTable::chunk_rows).
    size_t chunk_rows(size_t s, size_t cls, size_t band, double target_ms, size_t most) const {
        std::lock_guard<std::mutex> lk(m_);
        return stages_[s].prices.chunk_rows(cls, band, target_ms, most);
    }
    // A pass formed, its rows by kind.
    void formed(const RowKinds& k) {
        std::lock_guard<std::mutex> lk(m_);
        rows_.decode += k.decode;
        rows_.drafts += k.drafts;
        rows_.recomputed += k.recomputed;
        rows_.prompt += k.prompt;
        rows_.reread += k.reread;
        if (rows_.by_band.size() < k.by_band.size()) rows_.by_band.resize(k.by_band.size(), 0);
        for (size_t b = 0; b < k.by_band.size(); ++b) rows_.by_band[b] += k.by_band[b];
    }
    // A row class's least extent seen, which the health names the class by.
    void class_extent(size_t cls, size_t extent) {
        std::lock_guard<std::mutex> lk(m_);
        if (extents_.size() <= cls) extents_.resize(cls + 1, 0);
        if (!extents_[cls] || extent < extents_[cls]) extents_[cls] = extent;
    }
    // A request's first token, with its wait for admission and its time from reading to that token; and a later token's gap since the one before it.
    // A pass's reply objective while replies and prompt work both waited (docs/SERVER.md, The service rule): the gap it chose, whether the replies' pass left a chunk room and whether the step gap's maximum bound the gap.
    struct Objective {
        double gap_ms = 0;
        uint64_t met = 0, unmet = 0, capped = 0;
    };
    void objective(double gap_ms, bool met, bool capped) {
        std::lock_guard<std::mutex> lk(m_);
        objective_.gap_ms = gap_ms;
        ++(met ? objective_.met : objective_.unmet);
        objective_.capped += capped;
    }
    void first_token(double queued_ms, double first_ms) {
        std::lock_guard<std::mutex> lk(m_);
        queue_waits_.add(queued_ms);
        first_tokens_.add(first_ms);
    }
    void inter_token(double ms) {
        std::lock_guard<std::mutex> lk(m_);
        inter_tokens_.add(ms);
    }

    struct Snapshot {
        std::vector<StageRecord> stages;
        RowKinds rows;
        std::vector<size_t> extents;   // by row class, its least extent seen
        Recent::Summary queue_wait, first_token, inter_token;
        Objective objective;
    };
    Snapshot snapshot() const {
        std::lock_guard<std::mutex> lk(m_);
        return {stages_, rows_, extents_, queue_waits_.summary(), first_tokens_.summary(), inter_tokens_.summary(), objective_};
    }

private:
    // Each stage's oldest submission seen retired, by polling every millisecond while any waits, oldest stage first, so a pass's stage before is measured before it.
    // A stage's time runs from the latest of its recording's start, the stage's previous pass's end and the pass's previous stage's end, to its own end: the time a pass behind it waits for it.
    void loop() {
        std::unique_lock<std::mutex> lk(m_);
        while (!stop_) {
            bool waiting = false;
            for (size_t s = 0; s < pending_.size(); ++s) {
                while (!pending_[s].empty()) {
                    backend::Backend* b = pending_[s].front().backend;
                    const backend::Ticket t = pending_[s].front().ticket;
                    lk.unlock();
                    const bool done = !b || b->done(t);
                    const Clock::time_point now = Clock::now();
                    lk.lock();
                    if (!done) {
                        waiting = true;
                        break;
                    }
                    measure(pending_[s].front(), now);
                    pending_[s].pop_front();
                }
            }
            if (waiting) cv_.wait_for(lk, std::chrono::milliseconds(1));
            else cv_.wait(lk, [&] { return stop_ || std::any_of(pending_.begin(), pending_.end(), [](const std::deque<StageRun>& q) { return !q.empty(); }); });
        }
    }
    // Under m_.
    void measure(const StageRun& r, Clock::time_point end) {
        Clock::time_point from = std::max(r.begun, last_done_[r.stage]);
        const auto up = upstream_.find(r.pass);
        if (r.stage && up != upstream_.end() && up->second.first + 1 == r.stage) from = std::max(from, up->second.second);
        last_done_[r.stage] = end;
        if (r.stage + 1 < stages_.size()) upstream_[r.pass] = {r.stage, end};
        else if (up != upstream_.end()) upstream_.erase(up);
        // A pass abandoned before its last stage leaves its entry, which goes once passes far newer retire.
        while (!upstream_.empty() && upstream_.begin()->first + 4 * StageRecord::kRecentPasses < r.pass) upstream_.erase(upstream_.begin());
        const double ms = std::max(0.0, std::chrono::duration<double, std::milli>(end - from).count());
        StageRecord& st = stages_[r.stage];
        ++st.passes;
        st.measured_ms += ms;
        st.longest_ms = std::max(st.longest_ms, ms);
        st.recording_ms += std::max(0.0, std::chrono::duration<double, std::milli>(r.recorded - r.begun).count());
        st.lookup_ms += r.lookup_ms;
        st.sum_ms += r.sum_ms;
        st.after_ms += std::max(0.0, std::chrono::duration<double, std::milli>(end - std::max(r.recorded, from)).count());
        if (r.predicted_ms > 0) {
            ++st.predicted;
            st.predicted_ms += r.predicted_ms;
            st.error_ms += std::abs(ms - r.predicted_ms);
            st.over += ms > r.predicted_ms;
        }
        st.prices.observe(r.work, ms);
        st.recent.push_back({r.pass, ms, r.predicted_ms, r.work});
        if (st.recent.size() > StageRecord::kRecentPasses) st.recent.pop_front();
    }

    std::vector<StageRecord> stages_;                  // under m_
    std::vector<std::deque<StageRun>> pending_;        // under m_, per stage in recording order
    std::vector<Clock::time_point> last_done_;         // under m_, per stage, when its last pass seen ended
    std::map<uint64_t, std::pair<size_t, Clock::time_point>> upstream_;   // under m_, by pass, its last stage seen and when it ended
    RowKinds rows_;
    Objective objective_;
    std::vector<size_t> extents_;
    Recent queue_waits_, first_tokens_, inter_tokens_;
    bool stop_ = false;
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::thread thread_;
};

} // namespace server
