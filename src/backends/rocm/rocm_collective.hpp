#pragma once
// A tensor group's collective over ROCm devices and RocmBackend's part in it; rocm_backend.cpp includes it after the class.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <memory>
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
};
static_assert(kGroupMembers == 16 && sizeof(GroupPeers) == 272 && offsetof(GroupPeers, status) == 256, "kernels/f32.hip, GroupPeers");
constexpr size_t kGroupBlocks = 16;       // the kernel's signal slots a member, so the most blocks a sum takes
constexpr unsigned kGroupTimeoutMs = 1000;   // how long a member's kernel waits for its peers before the group gives up

// A tensor group's ordered sum (docs/src/backends-rocm.md, the collective): each member's kernel, on its own stream, pushes its partial rows into every peer's staging, meets them on device signals and adds the rows in member order.
// A sum that gives up (a peer never came within kGroupTimeoutMs) turns the collective to the conservative path for good: each member pulls the peers' rows behind events on their streams and adds them in the same order.
class RocmCollective final : public Collective {
public:
    RocmCollective(const std::vector<RocmBackend*>& members, size_t rows, size_t width, bool fast)
        : members_(members), rows_(rows), width_(width), cap_((size_mul(rows, width) + 3) / 4 * 4), fast_(fast),
          area_(members.size(), nullptr), ready_(members.size(), nullptr), pulled_(members.size(), nullptr), timeout_(members.size(), 0),
          last_(members.size(), 0), before_(members.size(), 0) {
        const HipFn& f = hip();
        const size_t W = members.size();
        try {
            hip_check(f.hipHostMalloc((void**)&status_, W * sizeof(unsigned), hipHostMallocCoherent), "hipHostMalloc");
            for (size_t m = 0; m < W; ++m) status_[m] = 0;
            for (size_t m = 0; m < W; ++m) {
                RocmBackend& b = *members[m];
                partial_.push_back(b.alloc(size_mul(size_mul(rows, width), sizeof(float)), Memory::device));
                hip_check(f.hipSetDevice(b.dev_.index), "hipSetDevice");
                const size_t bytes = size_add(kGroupSignalBytes, size_mul(size_mul(W, cap_), sizeof(float)));
                // Peers' stores reach uncached memory without the fine-grain setting, which ordinary allocations do not (docs/ROCM.md, Collective design and proof).
                if (fast) hip_check(f.hipExtMallocWithFlags(&area_[m], bytes, hipDeviceMallocUncached), "hipExtMallocWithFlags");
                else hip_check(f.hipMalloc(&area_[m], bytes), "hipMalloc");
                b.hold_fill(area_[m], kGroupSignalBytes);
                hip_check(f.hipEventCreateWithFlags(&ready_[m], hipEventDisableTiming), "hipEventCreateWithFlags");
                hip_check(f.hipEventCreateWithFlags(&pulled_[m], hipEventDisableTiming), "hipEventCreateWithFlags");
                int khz = 0;
                hip_check(f.hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, b.dev_.index), "hipDeviceGetAttribute");
                timeout_[m] = (unsigned long long)khz * kGroupTimeoutMs;
            }
            for (size_t m = 0; m < W; ++m) {
                GroupPeers p{};
                for (size_t k = 0; k < W; ++k) {
                    p.stage[k] = stage(k);
                    p.signal[k] = (unsigned*)area_[k];
                }
                p.status = status_;
                p.fault = members[m]->fault_;
                peers_.push_back(members[m]->alloc(sizeof(GroupPeers), Memory::device));
                members[m]->write(*peers_.back(), 0, &p, sizeof(GroupPeers));
                peer_table_.push_back(as_rocm(*peers_.back()).data());
            }
            // Every member's signals are zero before any peer may signal into them, which a zero fill still queued on its own stream would overwrite.
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

private:
    void summed(const std::vector<Slice>& residual, size_t n, bool replace) {
        const size_t W = members_.size();
        for (size_t m = 0; m < W; ++m) span(*residual[m].buffer, residual[m].offset * sizeof(float), size_mul(n, sizeof(float)));
        settle();
        // Every member's earlier failure throws before any member launches, so none waits for a peer that never comes.
        for (RocmBackend* b : members_) b->raise_failed();
        // Every member's held work goes to its stream before the sum and the sum's launches go together, so no member's sum waits on a peer whose launch the host still holds, and the pulls' events follow what they cover.
        for (RocmBackend* b : members_) b->flush();
        if (n) {
            if (fast_) pushed(residual, n, replace);
            else pulled(residual, n, replace);
        }
        for (RocmBackend* b : members_) b->flush();
        before_ = last_;
        for (size_t m = 0; m < W; ++m) last_[m] = members_[m]->tickets_.last() + 1;
    }

    void launch(size_t m, const Slice& out, size_t n, bool replace, bool fast) {
        RocmBackend& b = *members_[m];
        const float* own = RocmBackend::f32(CSlice(partial(m)));
        float* o = RocmBackend::f32(out);
        unsigned long long nn = n, cap = cap_, timeout = timeout_[m];
        int members = (int)members_.size(), rank = (int)m, flags = (replace ? 1 : 0) | (fast ? 2 : 0);
        const Args args(peer_table_[m], own, o, nn, cap, members, rank, flags, timeout);
        const size_t units = (n + 3) / 4;
        b.group_launch((unsigned)std::min<size_t>(kGroupBlocks, (units + kThreads - 1) / kThreads), args);
    }

    void pushed(const std::vector<Slice>& residual, size_t n, bool replace) {
        for (size_t m = 0; m < members_.size(); ++m) launch(m, residual[m], n, replace, true);
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
            launch(m, residual[m], n, replace, false);
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
    std::vector<BufferPtr> partial_;
    std::vector<void*> area_;   // per member: its signal block, then a staging slot per member
    std::vector<hipEvent_t> ready_, pulled_;   // the conservative path's events, per member
    std::vector<unsigned long long> timeout_;   // per member, kGroupTimeoutMs in its wall-clock ticks
    std::vector<BufferPtr> peers_;   // per member, its GroupPeers in its device memory
    std::vector<void*> peer_table_;   // per member, peers_'s device address, the kernel's first argument
    unsigned* status_ = nullptr;   // host memory, a word a member, set by a kernel that gave up
    std::vector<Ticket> last_, before_;   // per member, the ticket holding the last sum and the sum before it
};

// Every directed pair reaches the other's memory, peer access enabled, and one fast sum of known rows gives their member-order sum; otherwise why not.
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
    try {
        const size_t n = 1027;   // a tail past whole units, over one block
        RocmCollective c(group, 1, n, true);
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

// A collective over ROCm devices of one kind, each once, decided fast or conservative once per group by group_probe and reported on stderr.
std::unique_ptr<Collective> RocmBackend::join(const std::vector<Backend*>& members, size_t rows, size_t width) {
    if (members.empty() || members[0] != this) throw std::logic_error("backend: a collective is joined by its first member");
    if (members.size() > kGroupMembers) throw std::runtime_error("backend: a ROCm collective of more than " + std::to_string(kGroupMembers) + " devices");
    std::vector<RocmBackend*> group;
    std::vector<int> ids;
    for (Backend* b : members) {
        auto* r = dynamic_cast<RocmBackend*>(b);
        if (!r) throw std::runtime_error("backend: a ROCm collective over a backend that is not ROCm");
        if (r->dev_.caps.device != dev_.caps.device) throw std::runtime_error("backend: a ROCm collective over devices of different kinds");
        for (int id : ids)
            if (id == r->dev_.index) throw std::runtime_error("backend: a ROCm collective over one device twice");
        group.push_back(r);
        ids.push_back(r->dev_.index);
    }
    auto path = group_paths_.find(ids);
    if (path == group_paths_.end()) {
        path = group_paths_.emplace(ids, group_probe(group, ids)).first;
        std::string names;
        for (int id : ids) names += (names.empty() ? "" : ", ") + std::to_string(id);
        if (path->second.empty()) std::fprintf(stderr, "rocm: tensor group of devices %s sums by peer writes\n", names.c_str());
        else std::fprintf(stderr, "rocm: tensor group of devices %s sums by peer pulls, since %s\n", names.c_str(), path->second.c_str());
    }
    return std::make_unique<RocmCollective>(group, rows, width, path->second.empty());
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
