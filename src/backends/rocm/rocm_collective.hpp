#pragma once
// A tensor group's collective over ROCm devices and RocmBackend's part in it; rocm_backend.cpp includes it after the class.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace backend {
namespace {

constexpr size_t kGroupMembers = kMaxTensorWidth;   // the kernel's kGroupMembers (kernels/f32.hip)

// The kernel's view of the group (kernels/f32.hip, GroupPeers), written once into each member's device memory and passed by address.
struct GroupPeers {
    float* stage[kGroupMembers];
    unsigned* signal[kGroupMembers];
    unsigned* status;
    unsigned* fault;
    unsigned* cancel;
};
static_assert(kGroupMembers == 16 && sizeof(GroupPeers) == 280 && offsetof(GroupPeers, status) == 256 && offsetof(GroupPeers, cancel) == 272, "kernels/f32.hip, GroupPeers");
constexpr size_t kGroupBlocks = 32;       // the kernel's signal slots a member, so the most blocks a sum takes
constexpr size_t kOneShotBlocks = 16;     // the one-shot's most blocks
constexpr size_t kChainBlocks = 16, kChainUnits = 4096;   // the kernel's chain blocks each way and float4 units a chunk
constexpr unsigned kProbeTimeoutMs = 1000;   // how long a probe sum's member waits for its peers before the probe finds peer writes unseen
constexpr size_t kTimedFloor = 4096, kTimedSums = 3;   // the least floats a form is timed at, and the sums of each timing after a first
constexpr size_t kSumForms = 4;          // RocmSumForm's forms, each timed
constexpr double kOrderMargin = 0.04;    // how much faster than the order kept a member order must sum to replace it, so even links and noise keep the listed order

// A tensor group's ordered sum (docs/src/backends-rocm.md, the collective): each member's kernel, on its own stream, sends its partial rows to its peers in the form the sum's size takes and adds them in member order.
// A member waits for a late peer as long as it takes, and a sum the host cancelled or a probe that ran out turns the collective to the conservative path for good, where each member pulls the peers' rows behind events.
class RocmCollective final : public Collective {
public:
    // A probe bounds its members' waits by kProbeTimeoutMs, where peer writes that never become visible are what it looks for; `force` takes one form for every fast sum.
    RocmCollective(const std::vector<RocmBackend*>& members, size_t rows, size_t width, bool fast, bool probe = false, std::optional<RocmSumForm> force = std::nullopt)
        : members_(members), rows_(rows), width_(width), cap_((size_mul(rows, width) + 3) / 4 * 4), fast_(fast), force_(force),
          area_(members.size(), nullptr), ready_(members.size(), nullptr), pulled_(members.size(), nullptr), timeout_(members.size(), 0), tick_(members.size(), 0),
          last_(members.size(), 0), before_(members.size(), 0) {
        const HipFn& f = hip();
        const size_t W = members.size();
        try {
            hip_check(f.hipHostMalloc((void**)&status_, (W + 1) * sizeof(unsigned), hipHostMallocCoherent), "hipHostMalloc");
            for (size_t m = 0; m <= W; ++m) status_[m] = 0;
            for (size_t m = 0; m < W; ++m) {
                RocmBackend& b = *members[m];
                partial_.push_back(b.alloc(size_mul(size_mul(rows, width), sizeof(float)), Memory::device));
                hip_check(f.hipSetDevice(b.dev_.index), "hipSetDevice");
                // A fast collective keeps two parity halves of the slots, which the one-shot reads as signals (kernels/f32.hip, llmx_group_sum).
                const size_t bytes = size_add(kGroupSignalBytes, size_mul(size_mul(fast ? 2 * W : W, cap_), sizeof(float)));
                // Peers' stores reach uncached memory without the fine-grain setting, which ordinary allocations do not (docs/ROCM.md, Collective design and proof).
                if (fast) hip_check(f.hipExtMallocWithFlags(&area_[m], bytes, hipDeviceMallocUncached), "hipExtMallocWithFlags");
                else hip_check(f.hipMalloc(&area_[m], bytes), "hipMalloc");
                b.hold_fill(area_[m], kGroupSignalBytes);
                if (fast) b.group_arm(stage(m), size_mul(2 * W, cap_));
                hip_check(f.hipEventCreateWithFlags(&ready_[m], hipEventDisableTiming), "hipEventCreateWithFlags");
                hip_check(f.hipEventCreateWithFlags(&pulled_[m], hipEventDisableTiming), "hipEventCreateWithFlags");
                int khz = 0;
                hip_check(f.hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, b.dev_.index), "hipDeviceGetAttribute");
                timeout_[m] = probe ? (unsigned long long)khz * kProbeTimeoutMs : 0;
                tick_[m] = (unsigned long long)khz;
            }
            for (size_t m = 0; m < W; ++m) {
                GroupPeers p{};
                for (size_t k = 0; k < W; ++k) {
                    p.stage[k] = stage(k);
                    p.signal[k] = (unsigned*)area_[k];
                }
                p.status = status_;
                p.fault = members[m]->fault_;
                p.cancel = status_ + W;
                peers_.push_back(members[m]->alloc(sizeof(GroupPeers), Memory::device));
                members[m]->write(*peers_.back(), 0, &p, sizeof(GroupPeers));
                peer_table_.push_back(as_rocm(*peers_.back()).data());
            }
            // Every member's signals are zero and its slots armed before any peer may write them, which a fill still queued on its own stream would overwrite.
            for (RocmBackend* b : members) b->sync();
        } catch (...) {
            release();
            throw;
        }
    }
    ~RocmCollective() override { release(); }

    Slice partial(size_t member) override { return {partial_.at(member).get(), 0}; }

    // The members' submissions that hold the sum, made here where the caller has not yet submitted; a sum that gave up turns the collective conservative.
    void wait(size_t back) noexcept override {
        const std::vector<Ticket>& sum = back ? before_ : last_;
        for (size_t m = 0; m < sum.size(); ++m) {
            RocmBackend& b = *members_[m];
            if (sum[m] > b.tickets_.last()) {
                try { b.record(); } catch (const std::exception&) { continue; }
            }
            b.wait(sum[m]);
        }
        settle();
    }

    void sum_into(const std::vector<Slice>& residual, size_t rows, size_t width) override {
        if (residual.size() != members_.size() || rows > rows_ || width != width_)
            throw std::logic_error("backend: a sum of other members, rows or width than its collective was made for");
        summed(residual, size_mul(rows, width), false);
    }

    // Each member's own partial rows are its residual, which the sum replaces.
    void sum_in_place(size_t rows, size_t width) override {
        const size_t n = size_mul(rows, width);
        if (n > size_mul(rows_, width_)) throw std::logic_error("backend: a sum past the rows its collective was made for");
        std::vector<Slice> own;
        for (size_t m = 0; m < members_.size(); ++m) own.push_back(partial(m));
        summed(own, n, true);
    }

    // Whether the sums still take the fast path; a test's witness.
    bool fast() const { return fast_; }

    // Each fast sum's form by its size, kept by the group's first member.
    void use(const std::map<size_t, RocmSumForm>& forms) { forms_ = forms; }

    // The least milliseconds of kTimedSums sums in place of n floats in `form`, after a first; unbounded where a sum gave up, which leaves the collective conservative.
    double timed(RocmSumForm form, size_t n) {
        std::vector<Slice> own;
        for (size_t m = 0; m < members_.size(); ++m) own.push_back(partial(m));
        const std::optional<RocmSumForm> kept = force_;
        force_ = form;
        double ms = 1e30;
        for (size_t r = 0; fast_ && r <= kTimedSums; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            summed(own, n, true);
            wait(0);
            const double t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (r) ms = std::min(ms, t);
        }
        force_ = kept;
        return fast_ ? ms : 1e30;
    }

    // Each form timed at the reserved rows and halvings of them in whole rows down to one row, until the one-shot wins or n reaches what was timed before; the winner at each n goes into `forms`.
    // Returns the times as stderr reports them, empty where nothing was timed.
    std::string time_forms(std::map<size_t, RocmSumForm>& forms, size_t timed_before) {
        static const char* const names[] = {"one-shot", "two shots", "chain", "wrap"};
        std::string report;
        for (size_t r = rows_, n = size_mul(r, width_); fast_ && r && n > timed_before && n >= kTimedFloor; r /= 2, n = r * width_) {
            double ms[kSumForms];
            size_t won = 0;
            for (size_t f = 0; f < kSumForms; ++f) {
                ms[f] = timed((RocmSumForm)f, n);
                if (ms[f] < ms[won]) won = f;
            }
            if (!fast_) break;
            forms[n] = (RocmSumForm)won;
            char line[160];
            std::snprintf(line, sizeof line, "%s%zu floats %.3f %.3f %.3f %.3f %s", report.empty() ? "" : "; ", n, ms[0], ms[1], ms[2], ms[3], names[won]);
            report += line;
            if (!won) break;
        }
        return report;
    }

    // Why some member's add tells a value from the twin the one-shot sends for the armed bits, empty where none does: the sanitized operands' sums must give the raw ones' bits.
    static std::string add_check(const std::vector<RocmBackend*>& group, const std::vector<int>& ids) {
        static const uint32_t patterns[] = {0x7f800001u, 0x7fc00001u, 0xff800001u, 0xffc00001u, 0x7fc00000u, 0x7f812345u, 0x7fd12345u, 0x3fc00000u, 0x80000000u, 0x0u, 0x7f800000u, 0xff800000u, 0x00000001u};
        constexpr int np = sizeof(patterns) / sizeof(patterns[0]);
        try {
            for (size_t m = 0; m < group.size(); ++m) {
                RocmBackend& b = *group[m];
                const BufferPtr pat = b.alloc(sizeof(patterns), Memory::device), differ = b.alloc(3 * sizeof(uint32_t), Memory::device);
                const uint32_t zero[3] = {0, 0, 0};
                b.write(*pat, 0, patterns, sizeof(patterns));
                b.write(*differ, 0, zero, sizeof(zero));
                b.group_add_check(*pat, np, *differ);
                uint32_t got[3];
                b.read(*differ, 0, got, sizeof(got));
                if (got[0] || got[1] || got[2])
                    return "device " + std::to_string(ids[m]) + "'s add tells a sent value from its sanitized twin (a + b " + std::to_string(got[0]) + ", (a + b) + c " + std::to_string(got[1]) + ", c + (a + b) " +
                           std::to_string(got[2]) + " of " + std::to_string(np * np * np) + ")";
            }
        } catch (const std::exception& e) {
            return std::string("an add check failed: ") + e.what();
        }
        return {};
    }

private:
    void summed(const std::vector<Slice>& residual, size_t n, bool replace) {
        const size_t W = members_.size();
        for (size_t m = 0; m < W; ++m) span(*residual[m].buffer, residual[m].offset * sizeof(float), size_mul(n, sizeof(float)));
        settle();
        // Every member's earlier failure throws before any member launches, so none waits for a peer that never comes.
        for (RocmBackend* b : members_) b->raise_failed();
        // Every member's held work goes to its stream before the sum and the sum's launches go together, so no member's sum waits on a peer whose launch the host still holds, and the pulls' events follow what they cover.
        for (RocmBackend* b : members_) b->flush();
        // A sum whose launch throws is still the last sum, so a wait for it covers the members' kernels that did launch.
        const auto issued = [&] {
            before_ = last_;
            for (size_t m = 0; m < W; ++m) last_[m] = members_[m]->tickets_.last() + 1;
        };
        try {
            if (n) {
                if (fast_) pushed(residual, n, replace);
                else pulled(residual, n, replace);
            }
            for (RocmBackend* b : members_) b->flush();
        } catch (...) {
            cancel();
            issued();
            throw;
        }
        issued();
    }

    // The form a fast sum of n floats takes: the forced one, else the one that won at the largest timed size not above n.
    RocmSumForm form_of(size_t n) const {
        if (force_) return *force_;
        auto it = forms_.upper_bound(n);
        return it == forms_.begin() ? RocmSumForm::one_shot : (--it)->second;
    }

    // The blocks of every member's launch follow from n and the form alone, so every member's block b meets its peers' block b.
    void launch(size_t m, const Slice& out, size_t n, bool replace, bool fast, RocmSumForm form) {
        RocmBackend& b = *members_[m];
        const float* own = RocmBackend::f32(CSlice(partial(m)));
        float* o = RocmBackend::f32(out);
        unsigned long long nn = n, cap = cap_, timeout = timeout_[m], tick = tick_[m];
        const size_t units = (n + 3) / 4, W = members_.size();
        if (!fast) form = RocmSumForm::one_shot;
        size_t blocks = std::min<size_t>(kOneShotBlocks, (units + kThreads - 1) / kThreads);
        if (form == RocmSumForm::two_shots) blocks = std::min<size_t>(kGroupBlocks, ((units + W - 1) / W + kThreads - 1) / kThreads);
        else if (form == RocmSumForm::chain || form == RocmSumForm::wrap) blocks = 2 * std::min<size_t>(kChainBlocks, (units + kChainUnits - 1) / kChainUnits);
        int members = (int)W, rank = (int)m, flags = (replace ? 1 : 0) | (fast ? 2 | (int)half_ << 2 : 0) | (form == RocmSumForm::wrap ? 8 : 0);
        const Args args(peer_table_[m], own, o, nn, cap, members, rank, flags, timeout, tick);
        b.group_launch(form, (unsigned)blocks, args);
    }

    // Every fast sum takes the half the next one-shot takes, which every peer finished reading at the one-shot before the last (docs/src/backends-rocm.md, the collective).
    void pushed(const std::vector<Slice>& residual, size_t n, bool replace) {
        const RocmSumForm f = form_of(n);
        for (size_t m = 0; m < members_.size(); ++m) launch(m, residual[m], n, replace, true, f);
        if (f == RocmSumForm::one_shot) half_ ^= 1u;
    }

    // Each member records that its rows are ready, pulls every peer's once theirs are, records that it has read them, and sums once every peer has read its rows, which a later write of them must follow.
    void pulled(const std::vector<Slice>& residual, size_t n, bool replace) {
        const HipFn& f = hip();
        const size_t W = members_.size(), bytes = n * sizeof(float);
        for (size_t m = 0; m < W; ++m) hip_check(f.hipEventRecord(ready_[m], members_[m]->stream_), "hipEventRecord");
        for (size_t m = 0; m < W; ++m) {
            RocmBackend& b = *members_[m];
            for (size_t k = 0; k < W; ++k) {
                if (k == m) continue;
                hip_check(f.hipStreamWaitEvent(b.stream_, ready_[k], 0), "hipStreamWaitEvent");
                b.launch_check(f.hipMemcpyAsync(stage(m) + k * cap_, RocmBackend::f32(CSlice(partial(k))), bytes, hipMemcpyDefault, b.stream_), "hipMemcpyAsync");
            }
            hip_check(f.hipEventRecord(pulled_[m], b.stream_), "hipEventRecord");
        }
        for (size_t m = 0; m < W; ++m) {
            for (size_t k = 0; k < W; ++k)
                if (k != m) hip_check(f.hipStreamWaitEvent(members_[m]->stream_, pulled_[k], 0), "hipStreamWaitEvent");
            launch(m, residual[m], n, replace, false, RocmSumForm::one_shot);
        }
    }

    // A sum's kernel that gave up set its member's status word; the collective takes the conservative path from then on, decided here once.
    void settle() noexcept {
        if (!fast_) return;
        for (size_t m = 0; m < members_.size(); ++m)
            if (__atomic_load_n(&status_[m], __ATOMIC_ACQUIRE)) {
                fast_ = false;
                std::fprintf(stderr, "rocm: a tensor group's sum gave up waiting on %s; the group's later sums take the conservative path\n", members_[m]->identity().c_str());
                return;
            }
    }

    // A sum some member's launch of which failed: the members whose kernels did launch stop waiting for it, and the collective turns conservative now.
    void cancel() noexcept {
        if (!fast_) return;
        __atomic_store_n(&status_[members_.size()], 1u, __ATOMIC_RELEASE);
        fast_ = false;
        std::fprintf(stderr, "rocm: a tensor group's sum was cancelled since a member's launch failed; the group's later sums take the conservative path\n");
    }

    // A member's staging slots, past its signal block.
    float* stage(size_t m) const { return (float*)((uint8_t*)area_[m] + kGroupSignalBytes); }

    // Every member drained first, so no kernel or copy still names what goes.
    void release() noexcept {
        const HipFn& f = hip();
        for (RocmBackend* m : members_) m->sync();
        for (size_t m = 0; m < members_.size(); ++m) {
            (void)f.hipSetDevice(members_[m]->dev_.index);
            if (area_[m]) (void)f.hipFree(area_[m]);
            if (ready_[m]) (void)f.hipEventDestroy(ready_[m]);
            if (pulled_[m]) (void)f.hipEventDestroy(pulled_[m]);
            area_[m] = ready_[m] = pulled_[m] = nullptr;
        }
        if (status_) (void)f.hipHostFree(status_);
        status_ = nullptr;
    }

    std::vector<RocmBackend*> members_;
    size_t rows_, width_, cap_;   // cap_: a staging slot's floats, the reserved rows rounded up to whole units of four
    bool fast_;
    std::optional<RocmSumForm> force_;   // the form every fast sum takes, else by size from forms_
    unsigned half_ = 0;   // the parity half of the staging the next fast sum takes, flipped by each one-shot
    std::map<size_t, RocmSumForm> forms_;
    std::vector<BufferPtr> partial_;
    std::vector<void*> area_;   // per member: its signal block, then a staging slot per member, twice on the fast path
    std::vector<hipEvent_t> ready_, pulled_;   // the conservative path's events, per member
    std::vector<unsigned long long> timeout_, tick_;   // per member in its wall-clock ticks: a probe's kProbeTimeoutMs, else 0 for none, and a millisecond
    std::vector<BufferPtr> peers_;   // per member, its GroupPeers in its device memory
    std::vector<void*> peer_table_;   // per member, peers_'s device address, the kernel's first argument
    unsigned* status_ = nullptr;   // host memory, a word a member, set by a kernel that gave up, then the host's cancel word
    std::vector<Ticket> last_, before_;   // per member, the ticket holding the last sum and the sum before it
};

// Every directed pair reaches the other's memory, peer access enabled, every member's add leaves the one-shot's sanitized values invisible, and one fast sum of known rows gives their member-order sum; otherwise why not.
std::string group_probe(const std::vector<RocmBackend*>& group, const std::vector<int>& ids) {
    const HipFn& f = hip();
    for (int a : ids)
        for (int b : ids) {
            if (a == b) continue;
            int can = 0;
            if (f.hipDeviceCanAccessPeer(&can, a, b) != hipSuccess || !can)
                return "device " + std::to_string(a) + " cannot reach device " + std::to_string(b) + "'s memory";
            hip_check(f.hipSetDevice(a), "hipSetDevice");
            const hipError_t e = f.hipDeviceEnablePeerAccess(b, 0);
            if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled)
                return "peer access from device " + std::to_string(a) + " to device " + std::to_string(b) + " failed with " + f.hipGetErrorName(e);
        }
    const std::string adds = RocmCollective::add_check(group, ids);
    if (!adds.empty()) return adds;
    try {
        const size_t n = 1027;   // a tail past whole units, over one block
        RocmCollective c(group, 1, n, true, true);
        std::vector<float> want(n, 0.0f);
        for (size_t m = 0; m < group.size(); ++m) {
            std::vector<float> rows(n);
            for (size_t i = 0; i < n; ++i) rows[i] = float((int)((i * 7 + m * 13) % 61) - 30) * 0.125f;
            for (size_t i = 0; i < n; ++i) want[i] = m ? want[i] + rows[i] : rows[i];
            const Slice p = c.partial(m);
            group[m]->write(*p.buffer, 0, rows.data(), n * sizeof(float));
        }
        c.sum_in_place(1, n);
        c.wait(0);
        if (!c.fast()) return "a probe sum gave up waiting for its peers";
        for (size_t m = 0; m < group.size(); ++m) {
            std::vector<float> got(n);
            group[m]->read(*c.partial(m).buffer, 0, got.data(), n * sizeof(float));
            if (std::memcmp(got.data(), want.data(), n * sizeof(float)))
                return "a probe sum on device " + std::to_string(ids[m]) + " read other values than its peers wrote";
        }
    } catch (const std::exception& e) {
        return std::string("a probe sum failed: ") + e.what();
    }
    return {};
}

std::string device_list(const std::vector<int>& ids) {
    std::string names;
    for (int id : ids) names += (names.empty() ? "" : ", ") + std::to_string(id);
    return names;
}

std::vector<RocmBackend*> RocmBackend::group_of(const std::vector<Backend*>& members, std::vector<int>& ids) {
    if (members.empty() || members[0] != this) throw std::logic_error("backend: a collective is joined by its first member");
    if (members.size() > kGroupMembers) throw std::runtime_error("backend: a ROCm collective of more than " + std::to_string(kGroupMembers) + " devices");
    std::vector<RocmBackend*> group;
    ids.clear();
    for (Backend* b : members) {
        auto* r = dynamic_cast<RocmBackend*>(b);
        if (!r) throw std::runtime_error("backend: a ROCm collective over a backend that is not ROCm");
        if (r->dev_.caps.device != dev_.caps.device) throw std::runtime_error("backend: a ROCm collective over devices of different kinds");
        for (int id : ids)
            if (id == r->dev_.index) throw std::runtime_error("backend: a ROCm collective over one device twice");
        group.push_back(r);
        ids.push_back(r->dev_.index);
    }
    return group;
}

GroupPath& RocmBackend::group_path(const std::vector<RocmBackend*>& group, const std::vector<int>& ids) {
    auto path = group_paths_.find(ids);
    if (path != group_paths_.end()) return path->second;
    path = group_paths_.emplace(ids, GroupPath{group_probe(group, ids)}).first;
    if (path->second.why.empty()) std::fprintf(stderr, "rocm: tensor group of devices %s sums by peer writes\n", device_list(ids).c_str());
    else std::fprintf(stderr, "rocm: tensor group of devices %s sums by peer pulls, since %s\n", device_list(ids).c_str(), path->second.why.c_str());
    return path->second;
}

// The order whose fastest form sums rows x width floats fastest, by swap descent from the listing, a swap taken only where it beats the order kept by kOrderMargin (docs/TENSOR-SPLIT.md, section 4.1).
// The order's forms are timed here into the path of the group as ordered, so join times nothing twice, and every collective timed is gone before the fit reads free memory.
std::vector<size_t> RocmBackend::member_order(const std::vector<Backend*>& members, size_t rows, size_t width) {
    std::vector<int> ids;
    const std::vector<RocmBackend*> group = group_of(members, ids);
    GroupPath& listed = group_path(group, ids);
    const size_t W = group.size(), n = size_mul(rows, width);
    if (!listed.order.empty()) return listed.order;
    for (size_t m = 0; m < W; ++m) listed.order.push_back(m);
    if (W < 2 || !listed.why.empty() || n < kTimedFloor) return listed.order;
    std::map<std::vector<size_t>, double> costs;
    const auto cost = [&](const std::vector<size_t>& order) {
        auto at = costs.find(order);
        if (at != costs.end()) return at->second;
        std::vector<RocmBackend*> ordered;
        for (size_t k : order) ordered.push_back(group[k]);
        double ms = 1e30;
        // An order whose collective or timing throws is never taken, as a probe that fails falls conservative.
        try {
            RocmCollective c(ordered, rows, width, true);
            for (size_t f = 0; f < kSumForms; ++f) ms = std::min(ms, c.timed((RocmSumForm)f, n));
        } catch (const std::exception&) {
            ms = 1e30;
        }
        return costs[order] = ms;
    };
    std::vector<size_t> kept = listed.order;
    double ms = cost(kept);
    const double first = ms;
    for (bool moved = true; moved;) {
        moved = false;
        std::vector<size_t> best;
        double best_ms = ms;
        for (size_t i = 0; i < W; ++i)
            for (size_t j = i + 1; j < W; ++j) {
                std::vector<size_t> o = kept;
                std::swap(o[i], o[j]);
                const double t = cost(o);
                if (t < best_ms) best_ms = t, best = o;
            }
        if (!best.empty() && best_ms < ms * (1.0 - kOrderMargin)) kept = best, ms = best_ms, moved = true;
    }
    listed.order = kept;
    std::vector<RocmBackend*> ordered;
    std::vector<int> order_ids;
    for (size_t k : kept) ordered.push_back(group[k]), order_ids.push_back(ids[k]);
    std::fprintf(stderr, "rocm: tensor group of devices %s takes member order %s, %.3f ms a sum of %zu floats against %.3f as listed, %zu orders timed\n",
                 device_list(ids).c_str(), device_list(order_ids).c_str(), ms, n, first, costs.size());
    GroupPath& path = ordered[0]->group_path(ordered, order_ids);
    if (path.why.empty() && path.timed < n) {
        std::string times;
        try {
            RocmCollective c(ordered, rows, width, true);
            times = c.time_forms(path.forms, path.timed);
            if (c.fast()) path.timed = n;
        } catch (const std::exception& e) {
            path.why = std::string("timing its sums failed: ") + e.what();
            std::fprintf(stderr, "rocm: tensor group of devices %s sums by peer pulls, since %s\n", device_list(order_ids).c_str(), path.why.c_str());
        }
        if (!times.empty()) std::fprintf(stderr, "rocm: tensor group of devices %s timed its sums in place, ms one-shot, two shots, chain, wrap: %s\n", device_list(order_ids).c_str(), times.c_str());
    }
    return listed.order;
}

// A collective over ROCm devices of one kind, each once, decided fast or conservative once per group by group_probe and reported on stderr.
std::unique_ptr<Collective> RocmBackend::join(const std::vector<Backend*>& members, size_t rows, size_t width) {
    std::vector<int> ids;
    const std::vector<RocmBackend*> group = group_of(members, ids);
    const std::string names = device_list(ids);
    GroupPath& g = group_path(group, ids);
    std::unique_ptr<RocmCollective> c = std::make_unique<RocmCollective>(group, rows, width, g.why.empty());
    // The forms are timed on the group's own cards at its first collective past the sizes timed so far, its members idle, so the topology picks them (docs/TENSOR-SPLIT.md, section 4.3).
    const size_t n = size_mul(rows, width);
    if (g.why.empty() && group.size() > 1 && n > g.timed) {
        std::string times;
        try {
            times = c->time_forms(g.forms, g.timed);
        } catch (const std::exception& e) {
            g.why = std::string("timing its sums failed: ") + e.what();
            std::fprintf(stderr, "rocm: tensor group of devices %s sums by peer pulls, since %s\n", names.c_str(), g.why.c_str());
            c.reset();
            return std::make_unique<RocmCollective>(group, rows, width, false);
        }
        g.timed = n;
        if (!times.empty()) std::fprintf(stderr, "rocm: tensor group of devices %s timed its sums in place, ms one-shot, two shots, chain, wrap: %s\n", names.c_str(), times.c_str());
    }
    c->use(g.forms);
    return c;
}

// A crossing's host-visible buffer that ROCm readers copy from on their own streams, and that the sender writes again behind each reader's last copy (docs/src/backends-rocm.md, handoffs).
// The order is counters in coherent host memory that stream memory operations write and wait for, since a HIP event waited for on another device's stream holds the host until it completes.
class RocmHandoff final : public Handoff {
public:
    RocmHandoff(RocmBackend& sender, std::vector<RocmBackend*> readers)
        : sender_(sender), readers_(std::move(readers)), counts_(readers_.size(), 0), waited_(readers_.size(), 0) {
        const size_t words = 1 + readers_.size();
        hip_check(hip().hipHostMalloc((void**)&words_, words * sizeof(uint64_t), hipHostMallocCoherent), "hipHostMalloc");
        std::memset(words_, 0, words * sizeof(uint64_t));
    }
    ~RocmHandoff() override { release(); }

    bool serves(size_t reader) const override { return reader < readers_.size() && readers_[reader]; }
    // The sender waits for each reader's count of takes to reach the takes it has recorded.
    void begin() override {
        for (size_t r = 0; r < readers_.size(); ++r)
            if (counts_[r]) wait_for(sender_, &words_[1 + r], counts_[r]);
    }
    // Both marks are issued at once rather than held to the next submit, so no host wait on any device's ticket can stand behind a device waiting for them.
    void sent() override {
        const uint64_t n = sends_.load(std::memory_order_relaxed) + 1;
        write(sender_, &words_[0], n);
        sender_.flush();
        sends_.store(n, std::memory_order_release);
    }
    void take(size_t reader, Buffer& dst, size_t dst_off, const Buffer& src, size_t src_off, size_t bytes) override {
        if (!serves(reader)) throw std::logic_error("backend: a ROCm handoff taken by a reader it does not serve");
        RocmBackend& b = *readers_[reader];
        const uint64_t sends = sends_.load(std::memory_order_acquire);
        if (waited_[reader] != sends) {
            // Recorded as waited only once the flush that issues the wait returns, as the counts are.
            wait_for(b, &words_[0], sends);
            b.flush();
            waited_[reader] = sends;
        }
        b.copy(dst, dst_off, src, src_off, bytes);
    }
    void taken(size_t reader) override {
        if (!serves(reader)) throw std::logic_error("backend: a ROCm handoff taken by a reader it does not serve");
        // The count advances only once its write is issued, so a flush that throws drops a write no sender's begin waits for.
        const uint64_t c = counts_[reader] + 1;
        write(*readers_[reader], &words_[1 + reader], c);
        readers_[reader]->flush();
        counts_[reader] = c;
    }

private:
    // Held with the backend's other stream operations, so they keep its order.
    static void wait_for(RocmBackend& on, uint64_t* word, uint64_t value) { on.hold_wait(word, value); }
    static void write(RocmBackend& on, uint64_t* word, uint64_t value) { on.hold_write(word, value); }
    // The counters outlive every operation queued on them, each side's stream drained first.
    void release() noexcept {
        for (RocmBackend* b : readers_)
            if (b) b->sync();
        sender_.sync();
        if (words_) (void)hip().hipHostFree(words_);
    }
    RocmBackend& sender_;
    std::vector<RocmBackend*> readers_;   // per reader, null for one this handoff does not serve
    uint64_t* words_ = nullptr;   // coherent host memory: the sends so far, then each reader's takes so far
    std::vector<uint64_t> counts_;   // per reader, its takes so far, written on its thread before the sender's next begin reads it
    std::vector<uint64_t> waited_;   // per reader, the send its stream last waited for
    std::atomic<uint64_t> sends_{0};   // sent calls so far, read on the readers' threads
};

// Every ROCm reader is served, since it copies from the sender's host memory and needs no peer access.
std::unique_ptr<Handoff> RocmBackend::handoff(const std::vector<Backend*>& readers) {
    std::vector<RocmBackend*> to;
    bool any = false;
    for (Backend* b : readers) {
        auto* r = dynamic_cast<RocmBackend*>(b);
        to.push_back(r);
        any = any || r;
    }
    if (!any) return nullptr;
    return std::make_unique<RocmHandoff>(*this, std::move(to));
}

} // namespace
} // namespace backend
