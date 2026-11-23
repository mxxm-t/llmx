// Include the implementation to drive a tensor group's sum kernel (sum_rows) on one device, at every source count up to the widest group.
#include "backends/vulkan/vulkan_backend.cpp"
#include "group_sums.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>

namespace backend {
namespace {
struct VulkanSumTest {
    static bool device_clock(VulkanBackend& b) { return b.dev_->device_clock; }
    static VkDescriptorBufferInfo bind(VulkanBackend& b, Buffer& buffer, size_t offset) { return b.bind(CSlice{&buffer, offset}); }
    static void dispatch(VulkanBackend& b, KernelId id, std::initializer_list<VkDescriptorBufferInfo> buffers, const void* push, size_t bytes, uint32_t groups) {
        b.dispatch(id, buffers, push, bytes, groups);
    }
    // The tagged route forced on below `limit` floats, as the frozen size switch would set it; false where a member lacks what it needs.
    static bool tag(Collective& c, size_t limit) {
        auto& v = static_cast<VulkanCollective&>(c);
        v.tag(limit);
        return v.tagged_limit_ != 0;
    }
    static size_t tagged_sums(Collective& c) { return static_cast<VulkanCollective&>(c).tagged_sums_; }
    // A queue-thread failure on the member's submitter, given out at the next hand-over.
    static void inject(VulkanBackend& b) {
        b.submitter()->run({[] { throw std::runtime_error("vulkan-sum: injected queue failure"); }, [] {}}, false);
        b.submitter()->drain();
    }
    // What summed() does after a sum that failed part way: the collective released and its semaphores made again.
    static void release(Collective& c) {
        auto& v = static_cast<VulkanCollective&>(c);
        v.release();
        v.make_semaphores();
    }
    // A stand-in called before each queue call of every collective, null for none.
    static void probe(VkResult (*p)(const VulkanBackend&, bool receivers)) { VulkanCollective::queue_probe_ = p; }
    static size_t host_signals() { return VulkanCollective::host_signals_.load(); }
    static Ticket taken(Collective& c, size_t member) { return static_cast<VulkanCollective&>(c).taken_.at(member); }
    // The group's open batches closed without a caller and the close's job run, its failure left for the next hand-over.
    static void close(VulkanBackend& b) {
        b.close_batches(false);
        b.submitter()->drain();
    }
    // Whether member t's receivers of the collective's last closed batch have signalled, waiting up to `ns`.
    static bool received(Collective& c, size_t t, uint64_t ns) {
        auto& v = static_cast<VulkanCollective&>(c);
        const Device& d = *v.members_[t]->device();
        VkSemaphoreWaitInfo wi{};
        wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wi.semaphoreCount = 1;
        wi.pSemaphores = &v.rtimeline_[t];
        wi.pValues = &v.received_;
        return d.fn.vkWaitSemaphores(d.device, &wi, ns) == VK_SUCCESS;
    }
    // A timeline on the member's device that its next submission waits on at 1, until open_gate signals it.
    static VkSemaphore gate(VulkanBackend& b) {
        VkSemaphore s = VulkanCollective::semaphore(*b.device());
        b.wait_on(s, 1);
        return s;
    }
    static void open_gate(VulkanBackend& b, VkSemaphore s) {
        VkSemaphoreSignalInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
        si.semaphore = s;
        si.value = 1;
        b.device()->fn.vkSignalSemaphore(b.device()->device, &si);
    }
    static void destroy(VulkanBackend& b, VkSemaphore s) { b.device()->fn.vkDestroySemaphore(b.device()->device, s, nullptr); }
};
} // namespace
} // namespace backend

namespace {

void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}

// The flagged sum's members simulated on one device, a member's dispatches recorded one after another: each member's control words, its inboxes one region per sender, its partials one row per epoch, residual, kept and gathered share.
// A tagged group (keep 8, mode 16) has groups of C slots of three floats, its rows in each sender's region past the float rows at a stride of kSlots slots whatever n is.
struct FlagGroup {
    static constexpr uint32_t C = 256, kLong = 25000000, kShort = 2500;   // floats or slots a group; caps in device ticks, a second and 100 us at the 25 MHz the slowest clock seen ticks at
    static constexpr uint32_t kSlots = 2816, kCanary = 0x7fc0dead;   // tagged slots a row, enough for 8192 floats; the words past n in residual and kept
    backend::VulkanBackend& b;
    size_t W, n, offset, E;
    bool tag;
    uint32_t G, Gc;
    size_t tag_at, inbox;   // floats of one sender's region: a flag word every 16 floats per group, 8 rows of n, then from tag_at 8 tagged rows
    std::vector<backend::BufferPtr> ctl, in, part, res, kept, gathered;

    static uint32_t groups(size_t n, bool tag) { return uint32_t(((tag ? (n + 2) / 3 : n) + C - 1) / C); }
    FlagGroup(backend::VulkanBackend& b, size_t W, size_t n, size_t offset, size_t E, uint32_t base, bool tag = false)
        : b(b), W(W), n(n), offset(offset), E(E), tag(tag), G(groups(n, tag)), Gc(uint32_t((G + W - 1) / W)), tag_at((size_t(G) * 16 + 8 * n + 63) / 64 * 64),
          inbox(tag_at + (tag ? 32 * size_t(kSlots) : 0)) {
        const std::vector<uint32_t> fill(W * inbox, base), canary(4, kCanary);
        std::vector<uint32_t> words(16 * (size_t(G) + 2), 0);
        for (uint32_t g = 0; g < G; ++g) words[16 * (g + 1)] = words[16 * (g + 1) + 1] = base;
        for (size_t m = 0; m < W; ++m) {
            ctl.push_back(b.alloc(words.size() * sizeof(uint32_t), backend::Memory::device));
            b.write(*ctl.back(), 0, words.data(), words.size() * sizeof(uint32_t));
            in.push_back(b.alloc(fill.size() * sizeof(uint32_t), backend::Memory::device));
            b.write(*in.back(), 0, fill.data(), fill.size() * sizeof(uint32_t));
            part.push_back(b.alloc(E * n * sizeof(float), backend::Memory::device));
            res.push_back(b.alloc((n + offset + 4) * sizeof(float), backend::Memory::device));
            b.write(*res.back(), (n + offset) * sizeof(float), canary.data(), 4 * sizeof(uint32_t));
            kept.push_back(b.alloc((n + 4) * sizeof(float), backend::Memory::device));
            b.write(*kept.back(), n * sizeof(float), canary.data(), 4 * sizeof(uint32_t));
            gathered.push_back(b.alloc(n * sizeof(float), backend::Memory::device));
        }
    }
    // Later epochs of a tagged batch at another n, at most the constructed one, over the same buffers, whose tagged rows keep their stride.
    void resize(size_t to) {
        n = to;
        G = groups(n, tag);
    }
    uint32_t lo(size_t m) const { return std::min<uint32_t>(G, uint32_t(m) * Gc); }
    uint32_t hi(size_t m) const { return std::min<uint32_t>(G, uint32_t(m + 1) * Gc); }
    // Sender m's rows of epoch row `el` (or its gathered share) for groups [g0, g1).
    void push(size_t m, uint32_t epoch, uint32_t mode, uint32_t lag, uint32_t gen, size_t el, bool from_gathered, uint32_t g0, uint32_t g1, uint32_t cap = kLong) {
        if (g0 >= g1) return;
        const uint32_t pc[13] = {uint32_t(n), epoch, uint32_t(m), uint32_t(W), G, C, Gc, g0, lag, cap, mode, gen, kSlots};
        const size_t t = tag ? tag_at : 0;
        auto peer = [&](size_t k, size_t at) { return backend::VulkanSumTest::bind(b, *in[k < W && k != m ? k : (m + 1) % W], (k < W && k != m ? m : 0) * inbox + at); };
        backend::VulkanSumTest::dispatch(b, backend::K_FLAG_PUSH,
            {from_gathered ? backend::VulkanSumTest::bind(b, *gathered[m], 0) : backend::VulkanSumTest::bind(b, *part[m], el * n), backend::VulkanSumTest::bind(b, *ctl[m], 0),
             peer(0, t), peer(1, t), peer(2, t), peer(3, t), peer(4, t), peer(5, t), peer(6, t), peer(7, t), peer(8, t), peer(9, t), peer(10, t), peer(11, t), peer(12, t),
             peer(13, t), peer(14, t), peer(15, t)},
            pc, sizeof(pc), g1 - g0);
    }
    // Receiver m for the one epoch base + 1 over groups [g0, g1), owning [own_lo, own_hi).
    void recv(size_t m, uint32_t base, uint32_t keep, uint32_t gen, size_t el, uint32_t g0, uint32_t g1, uint32_t own_lo, uint32_t own_hi, uint32_t cap = kLong) {
        if (g0 >= g1) return;
        const uint32_t pc[15] = {uint32_t(n), base, 1u, uint32_t(m), uint32_t(W), G, C, Gc, g0, own_lo, own_hi, cap, keep, gen, kSlots};
        const size_t t = tag ? tag_at : 0;
        auto inbox_of = [&](size_t k, size_t at) { return backend::VulkanSumTest::bind(b, *in[m], (k < W ? k : 0) * inbox + at); };
        backend::VulkanSumTest::dispatch(b, backend::K_FLAG_RECV,
            {backend::VulkanSumTest::bind(b, *res[m], offset), backend::VulkanSumTest::bind(b, *kept[m], 0), backend::VulkanSumTest::bind(b, *part[m], el * n),
             backend::VulkanSumTest::bind(b, *ctl[m], 0), backend::VulkanSumTest::bind(b, *gathered[m], 0),
             inbox_of(0, t), inbox_of(1, t), inbox_of(2, t), inbox_of(3, t), inbox_of(4, t), inbox_of(5, t), inbox_of(6, t), inbox_of(7, t), inbox_of(8, t), inbox_of(9, t),
             inbox_of(10, t), inbox_of(11, t), inbox_of(12, t), inbox_of(13, t), inbox_of(14, t), inbox_of(15, t)},
            pc, sizeof(pc), g1 - g0);
    }
    // One epoch's whole sum on every member: the mesh sends every group to every peer, the two-shot sends each group to its share's owner, which sums it and gathers it back.
    // The ack wait (mode 8) starts at the second epoch, since here the receiver that rebases the ack runs after the senders rather than beside them.
    void epoch(uint32_t e, size_t el, bool two_shot, uint32_t keep, uint32_t gen) {
        const uint32_t wait = el ? 8u : 0u, scatter = two_shot ? 4u : 0u;
        if (tag) {
            for (size_t m = 0; m < W; ++m) push(m, e, 16u | wait, 4, gen, el, false, 0, G);
            for (size_t m = 0; m < W; ++m) recv(m, e - 1, keep | 8u, gen, el, 0, G, 0, G);
            return;
        }
        for (size_t m = 0; m < W; ++m) push(m, e, scatter | 1u | wait, 4, gen, el, false, 0, G);
        for (size_t m = 0; m < W; ++m) push(m, e, scatter | 2u, 4, gen, el, false, 0, G);
        if (!two_shot) {
            for (size_t m = 0; m < W; ++m) recv(m, e - 1, keep, gen, el, 0, G, 0, G);
            return;
        }
        for (size_t m = 0; m < W; ++m) recv(m, e - 1, keep | 4u, gen, el, lo(m), hi(m), lo(m), hi(m));
        for (size_t m = 0; m < W; ++m) push(m, e, 1u | 8u, 0, gen, el, true, lo(m), hi(m));
        for (size_t m = 0; m < W; ++m) push(m, e, 2u, 0, gen, el, true, lo(m), hi(m));
        for (size_t m = 0; m < W; ++m) {
            recv(m, e - 1, keep, gen, el, 0, lo(m), lo(m), hi(m));
            recv(m, e - 1, keep, gen, el, hi(m), G, lo(m), hi(m));
        }
    }
    std::vector<uint32_t> words(const backend::BufferPtr& buffer, size_t at, size_t count) {
        b.submit();
        std::vector<uint32_t> got(count);
        b.read(*buffer, at * sizeof(uint32_t), got.data(), count * sizeof(uint32_t));
        return got;
    }
};

// The flagged sum (shaders/flag_push.comp, flag_recv.comp) over E epochs, its 8 inbox rows wrapping, mesh and two-shot, added and written over with the sum kept, each member's residual the oracle's bits after every epoch's add in turn.
// A base 8 below the 32-bit wrap carries the epochs across it, which their signed comparisons hold.
size_t check_flagged(backend::VulkanBackend& b) {
    size_t checks = 0;
    uint32_t seed = 101, gen = 1;
    const size_t E = 20;
    for (size_t W : {size_t(2), size_t(3), size_t(4), size_t(5), size_t(8), backend::kMaxTensorWidth})
        for (size_t n : {size_t(1), size_t(255), size_t(256), size_t(257), size_t(1000), size_t(5000)})
            for (size_t offset : {size_t(0), size_t(7)})
                for (int mode = 0; mode < 4; ++mode) {
                    const bool replace = mode & 1, two_shot = mode & 2;
                    const uint32_t keep = replace ? 3u : 0u, base = offset ? 0xfffffff8u : 8u;
                    FlagGroup f(b, W, n, offset, E, base);
                    std::vector<group_sums::Case> cases;
                    for (size_t el = 0; el < E; ++el) {
                        cases.push_back(group_sums::make(W, n, seed++));
                        for (size_t m = 0; m < W; ++m) b.write(*f.part[m], el * n * sizeof(float), cases.back().partial[m].data(), n * sizeof(float));
                    }
                    for (size_t m = 0; m < W; ++m) b.write(*f.res[m], offset * sizeof(float), cases[0].start.data(), n * sizeof(float));
                    for (size_t el = 0; el < E; ++el) f.epoch(base + uint32_t(el) + 1, el, two_shot, keep, gen);
                    std::vector<float> want = cases[0].start;
                    for (size_t el = 0; el < E; ++el) {
                        group_sums::Case k = cases[el];
                        k.start = want;
                        want = group_sums::oracle(k, replace);
                    }
                    const std::vector<float> sum = group_sums::oracle(cases[E - 1], true);
                    for (size_t m = 0; m < W; ++m) {
                        const std::vector<uint32_t> got = f.words(f.res[m], offset, n);
                        require(!std::memcmp(got.data(), want.data(), n * sizeof(float)), "a flagged sum is not the oracle's in member order");
                        if (keep) {
                            const std::vector<uint32_t> k = f.words(f.kept[m], 0, n);
                            require(!std::memcmp(k.data(), sum.data(), n * sizeof(float)), "a flagged sum's kept rows are not the oracle's");
                        }
                        require(f.words(f.ctl[m], 0, 3) == std::vector<uint32_t>{0, 0, 0}, "a flagged sum that completed recorded an abort or a timeout");
                        checks += 1 + (keep ? 1 : 0);
                    }
                    ++gen;
                }
    return checks;
}

// A missing flag, a stale row and an aborted batch: a timeout aborts its batch and writes nothing, an aborted member's later stores and flags are skipped, and the next batch, at a fresh base and generation, sums exactly.
size_t check_flag_faults(backend::VulkanBackend& b) {
    const size_t W = 3, n = 1000, E = 20;
    const uint32_t base = 8, gen = 1;
    FlagGroup f(b, W, n, 0, E, base);
    std::vector<group_sums::Case> cases;
    for (size_t el = 0; el < E; ++el) {
        cases.push_back(group_sums::make(W, n, uint32_t(900 + el)));
        for (size_t m = 0; m < W; ++m) b.write(*f.part[m], el * n * sizeof(float), cases.back().partial[m].data(), n * sizeof(float));
    }
    for (size_t m = 0; m < W; ++m) b.write(*f.res[m], 0, cases[0].start.data(), n * sizeof(float));
    std::vector<uint32_t> start(n);
    std::memcpy(start.data(), cases[0].start.data(), n * sizeof(float));
    // Members 0 and 1 publish epoch 9, member 2 never does: receiver 0 times out.
    for (size_t m = 0; m < 2; ++m) f.push(m, base + 1, 1u, 4, gen, 0, false, 0, f.G);
    for (size_t m = 0; m < 2; ++m) f.push(m, base + 1, 2u, 4, gen, 0, false, 0, f.G);
    f.recv(0, base, 0, gen, 0, 0, f.G, 0, f.G, FlagGroup::kShort);
    std::vector<uint32_t> c = f.words(f.ctl[0], 0, 17);
    require(c[0] == gen && c[2] >= 1, "a receiver missing a flag did not abort its batch");
    require(c[16] == base, "a receiver that timed out acked the epoch");
    require(f.words(f.res[0], 0, n) == start, "a receiver that timed out wrote its residual");
    // Member 0's batch is aborted, so its next epoch's rows and flags are not stored.
    f.push(0, base + 2, 1u, 4, gen, 1, false, 0, f.G);
    f.push(0, base + 2, 2u, 4, gen, 1, false, 0, f.G);
    require(f.words(f.in[1], 0, 1)[0] == base + 1, "an aborted member published a flag");
    require(f.words(f.in[1], size_t(f.G) * 16 + ((base + 2) & 7) * n, n) == std::vector<uint32_t>(n, base), "an aborted member stored rows");
    // Receiver 1 at epoch 17 finds epoch 9's flags on the same row and must not take them.
    f.recv(1, base + 8, 0, gen, 0, 0, f.G, 0, f.G, FlagGroup::kShort);
    require(f.words(f.ctl[1], 0, 1)[0] == gen, "a receiver took a stale row's flag as its epoch's");
    require(f.words(f.res[1], 0, n) == start, "a receiver wrote a stale row's sum");
    // The next batch, at base 32 and generation 2, past every epoch the aborted one issued, sums exactly despite the abort word batch 1 left.
    const uint32_t next = 32;
    for (size_t m = 0; m < W; ++m) b.write(*f.res[m], 0, cases[0].start.data(), n * sizeof(float));
    for (size_t el = 0; el < E; ++el) f.epoch(next + uint32_t(el) + 1, el, false, 0, gen + 1);
    std::vector<float> want = cases[0].start;
    for (size_t el = 0; el < E; ++el) {
        group_sums::Case k = cases[el];
        k.start = want;
        want = group_sums::oracle(k, false);
    }
    for (size_t m = 0; m < W; ++m) {
        const std::vector<uint32_t> got = f.words(f.res[m], 0, n);
        require(!std::memcmp(got.data(), want.data(), n * sizeof(float)), "the batch after an aborted one is not the oracle's sum");
    }
    return 7 + W;
}

std::vector<float> oracle_run(const std::vector<group_sums::Case>& cases, size_t from, size_t to, std::vector<float> start, bool replace) {
    for (size_t el = from; el < to; ++el) {
        group_sums::Case k = cases[el];
        k.start = start;
        start = group_sums::oracle(k, replace);
    }
    return start;
}

// The tagged mesh sum (flag_push mode 16, flag_recv keep 8) over 20 epochs, its 8 tagged rows wrapping, from base 8 and across the 32-bit wrap, added and written over with the sum kept, each residual the oracle's bits.
// It holds the words past n in residual and kept, and the float rows' flag words, which a tagged sum never writes, unchanged.
size_t check_tagged(backend::VulkanBackend& b) {
    size_t checks = 0;
    uint32_t seed = 5001, gen = 1;
    const size_t E = 20;
    const std::vector<uint32_t> canary(4, FlagGroup::kCanary);
    for (size_t W : {size_t(2), size_t(3), size_t(4), size_t(5), size_t(8), backend::kMaxTensorWidth})
        for (size_t n : {1, 2, 3, 4, 255, 256, 257, 766, 767, 768, 769, 770, 8192})
            for (size_t offset : {size_t(0), size_t(7)})
                for (bool replace : {false, true}) {
                    const uint32_t keep = replace ? 3u : 0u, base = offset ? 0xfffffff8u : 8u;
                    FlagGroup f(b, W, n, offset, E, base, true);
                    std::vector<group_sums::Case> cases;
                    for (size_t el = 0; el < E; ++el) {
                        cases.push_back(group_sums::make(W, n, seed++));
                        for (size_t m = 0; m < W; ++m) b.write(*f.part[m], el * n * sizeof(float), cases.back().partial[m].data(), n * sizeof(float));
                    }
                    for (size_t m = 0; m < W; ++m) b.write(*f.res[m], offset * sizeof(float), cases[0].start.data(), n * sizeof(float));
                    for (size_t el = 0; el < E; ++el) f.epoch(base + uint32_t(el) + 1, el, false, keep, gen);
                    const std::vector<float> want = oracle_run(cases, 0, E, cases[0].start, replace), sum = group_sums::oracle(cases[E - 1], true);
                    for (size_t m = 0; m < W; ++m) {
                        const std::vector<uint32_t> got = f.words(f.res[m], offset, n + 4);
                        require(!std::memcmp(got.data(), want.data(), n * sizeof(float)), "a tagged sum is not the oracle's in member order");
                        require(std::vector<uint32_t>(got.begin() + n, got.end()) == canary, "a tagged sum wrote its residual past n");
                        if (keep) {
                            const std::vector<uint32_t> k = f.words(f.kept[m], 0, n + 4);
                            require(!std::memcmp(k.data(), sum.data(), n * sizeof(float)), "a tagged sum's kept rows are not the oracle's");
                            require(std::vector<uint32_t>(k.begin() + n, k.end()) == canary, "a tagged sum wrote its kept rows past n");
                        }
                        require(f.words(f.ctl[m], 0, 3) == std::vector<uint32_t>{0, 0, 0}, "a tagged sum that completed recorded an abort or a timeout");
                        for (size_t k = 0; k < W; ++k)
                            for (uint32_t g = 0; g < f.G; ++g)
                                require(f.words(f.in[m], k * f.inbox + 16 * g, 1)[0] == base, "a tagged sum wrote a float row's flag word");
                        checks += 1 + (keep ? 1 : 0);
                    }
                    ++gen;
                }
    return checks;
}

// A silent peer or own sender, an aborted sender, a missing slot, a slot an epoch behind or ahead, another n on the same rows, a fault in one of three groups and the next batch.
// A wait that times out or finds its row overwritten aborts its batch and writes neither residual, kept rows nor ack; a slot ahead of its epoch aborts at once rather than at the cap.
size_t check_tagged_faults(backend::VulkanBackend& b) {
    const size_t W = 3, n = 766, E = 20;   // 256 slots, one group, so an abort leaves every word of the residual
    const uint32_t base = 8;
    uint32_t gen = 1;
    size_t checks = 0;
    std::vector<group_sums::Case> cases;
    for (size_t el = 0; el < E; ++el) cases.push_back(group_sums::make(W, n, uint32_t(7000 + el)));
    auto group = [&](uint32_t start_base) {
        auto f = std::make_unique<FlagGroup>(b, W, n, 0, E, start_base, true);
        for (size_t el = 0; el < E; ++el)
            for (size_t m = 0; m < W; ++m) b.write(*f->part[m], el * n * sizeof(float), cases[el].partial[m].data(), n * sizeof(float));
        for (size_t m = 0; m < W; ++m) {
            b.write(*f->res[m], 0, cases[0].start.data(), n * sizeof(float));
            b.write(*f->kept[m], 0, cases[0].start.data(), n * sizeof(float));
        }
        return f;
    };
    std::vector<uint32_t> start(n), slot(4);
    std::memcpy(start.data(), cases[0].start.data(), n * sizeof(float));
    const size_t row = (FlagGroup::kSlots * 4) * ((base + 1) & 7);   // floats into a tagged region of epoch base + 1's row
    auto untouched = [&](FlagGroup& f, size_t m, const char* what) {
        const std::vector<uint32_t> c = f.words(f.ctl[m], 0, 17);
        require(c[0] == gen && c[2] == 1 && c[16] == base, what);
        require(f.words(f.res[m], 0, n) == start, "a tagged receiver that aborted wrote its residual");
        require(f.words(f.kept[m], 0, n) == start, "a tagged receiver that aborted wrote its kept rows");
        checks += 2;
    };
    {
        // Members 0 and 1 send epoch 9, member 2 never does: receiver 0 times out, and its aborted batch stores neither slots nor its own rows' epoch at epoch 10.
        auto f = group(base);
        for (size_t m = 0; m < 2; ++m) f->push(m, base + 1, 16u, 4, gen, 0, false, 0, f->G);
        f->recv(0, base, 3u | 8u, gen, 0, 0, f->G, 0, f->G, FlagGroup::kShort);
        untouched(*f, 0, "a tagged receiver missing a peer did not abort once, or acked");
        f->push(0, base + 2, 16u, 4, gen, 1, false, 0, f->G);
        const size_t row2 = (FlagGroup::kSlots * 4) * ((base + 2) & 7);
        require(f->words(f->in[1], f->tag_at + row2, 4) == std::vector<uint32_t>(4, base), "an aborted member stored tagged slots");
        require(f->words(f->ctl[0], 17, 1)[0] == base + 1, "an aborted member published its own rows' epoch");
        checks += 2;
    }
    // Every member sends epoch 9, then one slot of member 1's rows at receiver 0 is set to another epoch: missing (base), behind (e - 8) and ahead (e + 8).
    for (uint32_t tagged_as : {base, base + 1 - 8, base + 1 + 8}) {
        ++gen;
        auto f = group(base);
        for (size_t m = 0; m < W; ++m) f->push(m, base + 1, 16u, 4, gen, 0, false, 0, f->G);
        const size_t at = 1 * f->inbox + f->tag_at + row + 4 * 200;
        slot = f->words(f->in[0], at, 4);
        slot[3] = tagged_as;
        b.write(*f->in[0], at * sizeof(uint32_t), slot.data(), slot.size() * sizeof(uint32_t));
        const bool ahead = tagged_as == base + 1 + 8;
        const auto t0 = std::chrono::steady_clock::now();
        f->recv(0, base, 3u | 8u, gen, 0, 0, f->G, 0, f->G, ahead ? 0xf0000000u : FlagGroup::kShort);
        f->words(f->ctl[0], 0, 1);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        untouched(*f, 0, "a tagged receiver took a slot of another epoch, or acked");
        if (ahead) require(s < 0.5, "a tagged receiver waited out its cap on an overwritten slot");
        checks += ahead;
    }
    {
        // Seven epochs at 766 floats, seven at 257 and six at 766 again on the same rows, whose tags stay where they were, so slots 86 and up hold epochs older than the row's.
        ++gen;
        auto f = group(base);
        std::vector<float> want = cases[0].start;
        size_t el = 0;
        for (const auto& [width, count] : {std::pair<size_t, size_t>{766, 7}, {257, 7}, {766, 6}}) {
            f->words(f->ctl[0], 0, 1);
            f->resize(width);
            std::vector<group_sums::Case> phase;
            std::vector<float> head(want.begin(), want.begin() + width);
            for (size_t i = 0; i < count; ++i) {
                phase.push_back(group_sums::make(W, width, uint32_t(7100 + el + i)));
                for (size_t m = 0; m < W; ++m) b.write(*f->part[m], (el + i) * width * sizeof(float), phase.back().partial[m].data(), width * sizeof(float));
            }
            for (size_t i = 0; i < count; ++i, ++el) f->epoch(base + uint32_t(el) + 1, el, false, 0, gen);
            head = oracle_run(phase, 0, count, head, false);
            std::copy(head.begin(), head.end(), want.begin());
        }
        for (size_t m = 0; m < W; ++m) {
            const std::vector<uint32_t> got = f->words(f->res[m], 0, n);
            require(!std::memcmp(got.data(), want.data(), n * sizeof(float)), "a tagged sum after another n on the same rows is not the oracle's");
        }
        checks += W;
    }
    {
        // Members 1 and 2 send epoch 9, receiver 0's own member never does: it times out at its own rows.
        ++gen;
        auto f = group(base);
        for (size_t m = 1; m < W; ++m) f->push(m, base + 1, 16u, 4, gen, 0, false, 0, f->G);
        f->recv(0, base, 3u | 8u, gen, 0, 0, f->G, 0, f->G, FlagGroup::kShort);
        untouched(*f, 0, "a tagged receiver missing its own rows did not abort once, or acked");
    }
    {
        // Three groups at 2000 floats, one slot of group 2 never sent: group 2 aborts and writes none of its floats, whatever the other groups do.
        ++gen;
        const size_t wide = 2000;
        FlagGroup f(b, W, wide, 0, 1, base, true);
        const group_sums::Case k = group_sums::make(W, wide, 7300);
        for (size_t m = 0; m < W; ++m) {
            b.write(*f.part[m], 0, k.partial[m].data(), wide * sizeof(float));
            b.write(*f.res[m], 0, k.start.data(), wide * sizeof(float));
        }
        for (size_t m = 0; m < W; ++m) f.push(m, base + 1, 16u, 4, gen, 0, false, 0, f.G);
        const size_t at = 1 * f.inbox + f.tag_at + row + 4 * 600;
        slot = f.words(f.in[0], at, 4);
        slot[3] = base;
        b.write(*f.in[0], at * sizeof(uint32_t), slot.data(), slot.size() * sizeof(uint32_t));
        f.recv(0, base, 8u, gen, 0, 0, f.G, 0, f.G, FlagGroup::kShort);
        const std::vector<uint32_t> c = f.words(f.ctl[0], 0, 16 * 4);
        require(f.G == 3 && c[0] == gen && c[2] >= 1 && c[16 * 3] == base, "a tagged group missing a slot did not abort, or acked");
        std::vector<uint32_t> tail(wide - 1536);
        std::memcpy(tail.data(), k.start.data() + 1536, tail.size() * sizeof(float));
        require(f.words(f.res[0], 1536, wide - 1536) == tail, "a tagged group that aborted wrote its residual");
        checks += 2;
    }
    {
        // After an aborted batch, the next one at base 40 and generation gen + 1 sums exactly despite the abort word.
        auto f = group(base);
        ++gen;
        for (size_t m = 0; m < 2; ++m) f->push(m, base + 1, 16u, 4, gen, 0, false, 0, f->G);
        f->recv(0, base, 8u, gen, 0, 0, f->G, 0, f->G, FlagGroup::kShort);
        for (size_t m = 0; m < W; ++m) b.write(*f->res[m], 0, cases[0].start.data(), n * sizeof(float));
        const uint32_t next = 40;
        ++gen;
        for (size_t el = 0; el < E; ++el) f->epoch(next + uint32_t(el) + 1, el, false, 0, gen);
        const std::vector<float> want = oracle_run(cases, 0, E, cases[0].start, false);
        for (size_t m = 0; m < W; ++m) {
            const std::vector<uint32_t> got = f->words(f->res[m], 0, n);
            require(!std::memcmp(got.data(), want.data(), n * sizeof(float)), "the tagged batch after an aborted one is not the oracle's sum");
        }
        checks += W;
    }
    return checks;
}

// Every source count from 1 to backend::kMaxTensorWidth, a buffer a source, added into a residual, written over it and kept, each the oracle's bits (tests/group_sums.hpp).
// Written over its first source, as a sum in place is, the sum is read before it is written.
size_t check_sources(backend::VulkanBackend& b) {
    using backend::CSlice;
    using backend::Slice;
    size_t checks = 0;
    uint32_t seed = 7;
    for (size_t W = 1; W <= backend::kMaxTensorWidth; ++W)
        for (size_t n : {size_t(1), size_t(255), size_t(256), size_t(257), size_t(1000)})
            for (size_t offset : {size_t(0), size_t(7)})
                for (int mode = 0; mode < 4; ++mode) {
                    const bool replace = mode & 1, keep = mode & 2;
                    const group_sums::Case k = group_sums::make(W, n, seed++);
                    std::vector<backend::BufferPtr> src;
                    std::vector<CSlice> sources;
                    for (size_t m = 0; m < W; ++m) {
                        src.push_back(b.alloc(n * sizeof(float), backend::Memory::device));
                        b.write(*src.back(), 0, k.partial[m].data(), n * sizeof(float));
                        sources.push_back({src.back().get(), 0});
                    }
                    const backend::BufferPtr x = b.alloc((n + offset) * sizeof(float), backend::Memory::device), kept = b.alloc(n * sizeof(float), backend::Memory::device);
                    b.write(*x, offset * sizeof(float), k.start.data(), n * sizeof(float));
                    const Slice kept_rows{kept.get(), 0};
                    b.sum_rows(Slice{x.get(), offset}, sources.data(), W, n, keep ? &kept_rows : nullptr, replace);
                    b.submit();
                    std::vector<float> got(n);
                    b.read(*x, offset * sizeof(float), got.data(), n * sizeof(float));
                    const std::vector<float> want = group_sums::oracle(k, replace), sum = group_sums::oracle(k, true);
                    require(!std::memcmp(got.data(), want.data(), n * sizeof(float)), "a sum of sources is not the oracle's in member order");
                    if (keep) {
                        b.read(*kept, 0, got.data(), n * sizeof(float));
                        require(!std::memcmp(got.data(), sum.data(), n * sizeof(float)), "a kept sum is not the oracle's in member order");
                    }
                    checks += 1 + keep;
                }
    // In place over source 0: each member's own rows are its residual and its first source.
    for (size_t W : {size_t(2), size_t(5), backend::kMaxTensorWidth}) {
        const size_t n = 1000;
        const group_sums::Case k = group_sums::make(W, n, seed++);
        std::vector<backend::BufferPtr> src;
        std::vector<CSlice> sources;
        for (size_t m = 0; m < W; ++m) {
            src.push_back(b.alloc(n * sizeof(float), backend::Memory::device));
            b.write(*src.back(), 0, k.partial[m].data(), n * sizeof(float));
            sources.push_back({src.back().get(), 0});
        }
        b.sum_rows(Slice{src[0].get(), 0}, sources.data(), W, n, nullptr, true);
        b.submit();
        std::vector<float> got(n);
        b.read(*src[0], 0, got.data(), n * sizeof(float));
        const std::vector<float> want = group_sums::oracle(k, true);
        require(!std::memcmp(got.data(), want.data(), n * sizeof(float)), "a sum written over its first source is not the oracle's");
        ++checks;
    }
    bool refused = false;
    try {
        std::vector<CSlice> many(backend::kMaxTensorWidth + 1);
        const backend::BufferPtr x = b.alloc(sizeof(float), backend::Memory::device);
        b.sum_rows(Slice{x.get(), 0}, many.data(), many.size(), 1, nullptr);
    } catch (const std::logic_error&) {
        refused = true;
    }
    require(refused, "a sum of more sources than the widest group was taken");
    return checks + 1;
}

// The tagged route of VulkanCollective over 2 and 4 devices: group_sums::check's sums in place go by it below the limit and by sync files past it, then 128 sums in place in one batch of mixed n, every eighth past the limit.
// Each member's rows must be the oracle's bits in member order, and the route's count must show which sums it took.
size_t check_route(const std::vector<std::unique_ptr<backend::VulkanBackend>>& devices) {
    constexpr size_t limit = 3072, rows = 4096;
    size_t checks = 0;
    for (size_t W : {size_t(2), size_t(4)}) {
        if (devices.size() < W) continue;
        std::vector<backend::Backend*> group;
        for (size_t m = 0; m < W; ++m) group.push_back(devices[m].get());
        auto c = devices[0]->join(group, rows, 1);
        require(c != nullptr, "no Vulkan collective over devices that share memory");
        if (!backend::VulkanSumTest::tag(*c, limit)) {
            std::cout << "vulkan-sum: no tagged route over " << W << " devices, a member lacking a receive queue, a clock or uncached memory\n";
            return checks;
        }
        checks += group_sums::check(group, *c, 1, {1, 3, 766, limit, limit + 1}, uint32_t(9000 + W), [](bool ok, const std::string& what) { require(ok, what.c_str()); });
        const size_t before = backend::VulkanSumTest::tagged_sums(*c);
        require(before == 8, "the tagged route did not take every sum in place below its limit, or took one past it");
        const group_sums::Case k = group_sums::make(W, rows, uint32_t(9100 + W));
        std::vector<std::vector<float>> want = k.partial;
        for (size_t m = 0; m < W; ++m) {
            const backend::Slice p = c->partial(m);
            group[m]->write(*p.buffer, p.offset * sizeof(float), k.partial[m].data(), rows * sizeof(float));
        }
        const size_t ns[4] = {1, 766, 3, limit};
        size_t past = 0;
        for (size_t i = 0; i < 128; ++i) {
            const size_t n = i % 8 == 7 ? limit + 1 : ns[i % 4];
            past += n > limit;
            c->sum_in_place(n, 1);
            for (size_t j = 0; j < n; ++j) {
                float t = want[0][j];
                for (size_t m = 1; m < W; ++m) t += want[m][j];
                for (size_t m = 0; m < W; ++m) want[m][j] = t;
            }
        }
        for (size_t m = 0; m < W; ++m) {
            group[m]->submit();
            std::vector<float> got(rows);
            const backend::Slice p = c->partial(m);
            group[m]->read(*p.buffer, p.offset * sizeof(float), got.data(), rows * sizeof(float));
            require(!std::memcmp(got.data(), want[m].data(), rows * sizeof(float)), "a batch of tagged sums in place is not the oracle's in member order");
            ++checks;
        }
        require(backend::VulkanSumTest::tagged_sums(*c) == before + 128 - past, "the tagged route's count is not the batch's sums below its limit");
        std::cout << "vulkan-sum: " << W << " devices, " << backend::VulkanSumTest::tagged_sums(*c) << " sums by the tagged route, " << 2 * 5 - 8 + past << " past it by sync files\n";
    }
    return checks;
}

// Member 0's recorded commands, read from its device's function table: 'B' a barrier with its first memory barrier's accesses (none for an execution-only one), 'D' a dispatch, 'C' a copy.
struct Recorded {
    char kind;
    VkAccessFlags src, dst;
};
std::vector<Recorded> recorded;
PFN_vkCmdPipelineBarrier real_barrier;
PFN_vkCmdDispatch real_dispatch;
PFN_vkCmdCopyBuffer real_copy;
void VKAPI_PTR record_barrier(VkCommandBuffer c, VkPipelineStageFlags s, VkPipelineStageFlags d, VkDependencyFlags f, uint32_t mn, const VkMemoryBarrier* m, uint32_t bn,
                              const VkBufferMemoryBarrier* b, uint32_t in, const VkImageMemoryBarrier* i) {
    recorded.push_back({'B', mn ? m[0].srcAccessMask : 0u, mn ? m[0].dstAccessMask : 0u});
    real_barrier(c, s, d, f, mn, m, bn, b, in, i);
}
void VKAPI_PTR record_dispatch(VkCommandBuffer c, uint32_t x, uint32_t y, uint32_t z) {
    recorded.push_back({'D', 0, 0});
    real_dispatch(c, x, y, z);
}
void VKAPI_PTR record_copy(VkCommandBuffer c, VkBuffer s, VkBuffer d, uint32_t n, const VkBufferCopy* r) {
    recorded.push_back({'C', 0, 0});
    real_copy(c, s, d, n, r);
}

// A tagged sum's barriers on member 0, between the rows' writer (a dispatch, or a copy where `copied`), the sender, the gate and a dispatch reading the rows.
// On RADV only the write-back barrier between sender and gate, elsewhere one making shader writes visible; after a dispatch the sender follows by execution order alone where `weak_before`, after a copy by a full barrier.
void require_barriers(bool copied, bool radv, bool weak_before) {
    std::vector<size_t> cmds;
    for (size_t i = 0; i < recorded.size(); ++i)
        if (recorded[i].kind != 'B') cmds.push_back(i);
    require(cmds.size() >= 4, "a tagged sum between a writer and a reader did not record four commands on member 0");
    auto between = [&](size_t a) {
        std::vector<Recorded> b;
        for (size_t i = cmds[a] + 1; i < cmds[a + 1]; ++i) b.push_back(recorded[i]);
        return b;
    };
    const std::vector<Recorded> before = between(0), middle = between(1), after = between(2);
    require(recorded[cmds[0]].kind == (copied ? 'C' : 'D'), "the rows' writer is not the first command recorded");
    const auto shows = [](const std::vector<Recorded>& b) {
        return std::any_of(b.begin(), b.end(), [](const Recorded& r) { return (r.src & VK_ACCESS_SHADER_WRITE_BIT) && (r.dst & VK_ACCESS_SHADER_READ_BIT); });
    };
    if (radv) require(middle.size() == 1 && middle[0].src == VK_ACCESS_SHADER_WRITE_BIT && middle[0].dst == 0, "the sender and the gate are not separated by the write-back barrier alone");
    else require(shows(middle), "off RADV the sender and the gate are not separated by a barrier making shader writes visible");
    const bool visible = shows(after);
    require(visible, "a dispatch reading the rows after a tagged sum is not behind a barrier making shader writes visible");
    if (copied) {
        const bool full = std::any_of(before.begin(), before.end(), [](const Recorded& r) { return (r.src & VK_ACCESS_TRANSFER_WRITE_BIT) && (r.dst & VK_ACCESS_SHADER_READ_BIT); });
        require(full, "a sender after a copy into its rows is not behind a barrier making the copy visible");
    } else {
        if (radv) require(before.size() == 1, "a sender after a dispatch writing its rows is not behind exactly one barrier");
        else require(shows(before), "off RADV a sender after a dispatch writing its rows is not behind a barrier making shader writes visible");
        if (weak_before) require(before[0].src == 0 && before[0].dst == 0, "a sender after a dispatch writing its rows is behind a memory barrier rather than execution order");
    }
}

// Tagged sums over two members between a writer of each member's rows, a dispatch or a copy by turns, and a dispatch reading them, ordered and then unordered: member 0's barriers as require_barriers holds them.
// Ordered, each epoch's read is the oracle's sum; the reader leaves the rows in the cache of the units it ran on, so a reader behind too weak a barrier would see the last epoch's rows.
size_t check_tagged_barriers(const std::vector<std::unique_ptr<backend::VulkanBackend>>& devices) {
    constexpr size_t rows = 4096, limit = 3072, n = 3072, E = 32;
    if (devices.size() < 2) {
        std::cout << "vulkan-sum: skip a tagged sum's barriers, fewer than two devices\n";
        return 0;
    }
    std::vector<backend::Backend*> group{devices[0].get(), devices[1].get()};
    // The rows outlive the collective, whose teardown waits for the members' queued work.
    std::vector<group_sums::Case> cases;
    std::vector<backend::BufferPtr> src, out;
    auto c = devices[0]->join(group, rows, 1);
    require(c != nullptr, "no Vulkan collective over devices that share memory");
    if (!backend::VulkanSumTest::tag(*c, limit)) {
        std::cout << "vulkan-sum: skip a tagged sum's barriers, a member lacking a receive queue, a clock or uncached memory\n";
        return 0;
    }
    for (size_t el = 0; el < E; ++el) cases.push_back(group_sums::make(2, n, uint32_t(9500 + el)));
    for (size_t m = 0; m < 2; ++m) {
        src.push_back(group[m]->alloc(E * n * sizeof(float)));
        out.push_back(group[m]->alloc(E * n * sizeof(float)));
        for (size_t el = 0; el < E; ++el) group[m]->write(*src[m], el * n * sizeof(float), cases[el].partial[m].data(), n * sizeof(float));
        group[m]->submit();
    }
    backend::Fn& fn = devices[0]->device()->fn;
    const backend::Fn saved = fn;
    real_barrier = saved.vkCmdPipelineBarrier;
    real_dispatch = saved.vkCmdDispatch;
    real_copy = saved.vkCmdCopyBuffer;
    fn.vkCmdPipelineBarrier = record_barrier;
    fn.vkCmdDispatch = record_dispatch;
    fn.vkCmdCopyBuffer = record_copy;
    const bool matched = devices[0]->device()->matched_barriers, on_radv = devices[0]->device()->radv;
    const auto restore = [&] {
        fn = saved;
        for (size_t m = 0; m < 2; ++m) {
            devices[m]->device()->radv = on_radv;
            group[m]->unordered(false);
        }
    };
    size_t checks = 0;
    try {
        // On RADV the barriers other drivers take run as well, the members' driver flag cleared.
        for (int arm = 0; arm < (on_radv ? 4 : 2); ++arm) {
            const bool radv = on_radv && arm < 2;
            const int unordered = arm % 2;
            for (size_t m = 0; m < 2; ++m) devices[m]->device()->radv = radv;
            for (backend::Backend* d : group) d->unordered(unordered != 0);
            for (size_t el = 0; el < E; ++el) {
                const bool copied = el % 2 == 1;
                recorded.clear();
                for (size_t m = 0; m < 2; ++m) {
                    const backend::Slice p = c->partial(m);
                    if (copied) {
                        group[m]->copy(*p.buffer, p.offset * sizeof(float), *src[m], el * n * sizeof(float), n * sizeof(float));
                    } else {
                        const backend::CSlice s{src[m].get(), el * n};
                        devices[m]->sum_rows(p, &s, 1, n, nullptr, true);
                    }
                }
                c->sum_in_place(n, 1);
                for (size_t m = 0; m < 2; ++m) {
                    const backend::CSlice p = c->partial(m);
                    devices[m]->sum_rows(backend::Slice{out[m].get(), el * n}, &p, 1, n, nullptr, true);
                }
                require_barriers(copied, radv, radv && (matched || unordered));
                for (backend::Backend* d : group) d->submit();
                ++checks;
            }
            for (backend::Backend* d : group) d->unordered(false);
            if (unordered) continue;
            for (size_t m = 0; m < 2; ++m) {
                std::vector<float> got(E * n);
                group[m]->read(*out[m], 0, got.data(), got.size() * sizeof(float));
                for (size_t el = 0; el < E; ++el) {
                    std::vector<float> want(n);
                    for (size_t j = 0; j < n; ++j) want[j] = cases[el].partial[0][j] + cases[el].partial[1][j];
                    require(!std::memcmp(got.data() + el * n, want.data(), n * sizeof(float)), "a dispatch reading the rows after a tagged sum did not read the oracle's sum");
                    ++checks;
                }
            }
        }
    } catch (...) {
        restore();
        throw;
    }
    restore();
    std::cout << "vulkan-sum: " << checks << " checks of a tagged sum's barriers and reads behind them, " << (matched ? "matched" : "full") << " barriers" << (on_radv ? ", RADV's and other drivers'" : "") << "\n";
    return checks;
}

// Two tagged groups sharing a member, {0, 1} and {1, 2}: 6 sums of each with both batches open and the first member's flush alone submitting them, then 12 alternating, ordered and then unordered.
// A flush that closed one group would leave the other's senders without receivers, and alternating sums whose receivers queued by group would wait on each other, both until the cap aborts.
size_t check_shared(const std::vector<std::unique_ptr<backend::VulkanBackend>>& devices) {
    constexpr size_t limit = 3072, rows = 4096;
    if (devices.size() < 3) {
        std::cout << "vulkan-sum: skip two tagged groups sharing a member, fewer than three devices\n";
        return 0;
    }
    std::vector<backend::Backend*> ga{devices[0].get(), devices[1].get()}, gb{devices[1].get(), devices[2].get()};
    auto a = devices[0]->join(ga, rows, 1);
    auto b = devices[1]->join(gb, rows, 1);
    require(a && b, "no Vulkan collectives over devices that share memory");
    if (!backend::VulkanSumTest::tag(*a, limit) || !backend::VulkanSumTest::tag(*b, limit)) {
        std::cout << "vulkan-sum: skip two tagged groups sharing a member, a member lacking a receive queue, a clock or uncached memory\n";
        return 0;
    }
    size_t checks = 0;
    for (int unordered = 0; unordered < 2; ++unordered) {
        const group_sums::Case ka = group_sums::make(2, rows, uint32_t(9200 + unordered)), kb = group_sums::make(2, rows, uint32_t(9300 + unordered));
        std::vector<std::vector<float>> wa = ka.partial, wb = kb.partial;
        for (size_t m = 0; m < 2; ++m) {
            const backend::Slice pa = a->partial(m), pb = b->partial(m);
            ga[m]->write(*pa.buffer, pa.offset * sizeof(float), ka.partial[m].data(), rows * sizeof(float));
            gb[m]->write(*pb.buffer, pb.offset * sizeof(float), kb.partial[m].data(), rows * sizeof(float));
        }
        for (backend::Backend* d : {ga[0], ga[1], gb[1]}) d->unordered(unordered != 0);
        const size_t ns[3] = {766, 3, limit};
        for (size_t i = 0; i < 24; ++i) {
            const size_t n = ns[i % 3];
            const bool second = i < 12 ? i >= 6 : i % 2 != 0;
            (second ? b : a)->sum_in_place(n, 1);
            std::vector<std::vector<float>>& w = second ? wb : wa;
            for (size_t j = 0; j < n; ++j) w[0][j] = w[1][j] = w[0][j] + w[1][j];
        }
        for (backend::Backend* d : {ga[0], ga[1], gb[1]}) d->unordered(false);
        devices[0]->submit();
        for (size_t m = 0; m < 2; ++m) {
            std::vector<float> got(rows);
            const backend::Slice pa = a->partial(m), pb = b->partial(m);
            ga[m]->read(*pa.buffer, pa.offset * sizeof(float), got.data(), rows * sizeof(float));
            require(!std::memcmp(got.data(), wa[m].data(), rows * sizeof(float)), "a tagged group beside another sharing its member is not the oracle's");
            gb[m]->read(*pb.buffer, pb.offset * sizeof(float), got.data(), rows * sizeof(float));
            require(!std::memcmp(got.data(), wb[m].data(), rows * sizeof(float)), "a tagged group sharing a member with an open one is not the oracle's");
            checks += 2;
        }
    }
    require(backend::VulkanSumTest::tagged_sums(*a) == 24 && backend::VulkanSumTest::tagged_sums(*b) == 24, "the tagged route did not take the shared groups' sums");
    return checks;
}

// Collectives over groups that share members route each device through one queue thread while any of them lives and give it back with the last; a group over two live groups' queue threads is refused routing nobody.
// Then a fresh group sums by sync files, so the queues work alone again.
size_t check_routes(const std::vector<std::unique_ptr<backend::VulkanBackend>>& devices) {
    constexpr size_t rows = 256;
    if (devices.size() < 3) {
        std::cout << "vulkan-sum: skip groups sharing members giving their queues back, fewer than three devices\n";
        return 0;
    }
    backend::VulkanBackend *d0 = devices[0].get(), *d1 = devices[1].get(), *d2 = devices[2].get();
    size_t checks = 0;
    {
        auto a = d0->join({d0, d1}, rows, 1);
        auto b = d1->join({d1, d2}, rows, 1);
        require(a && b, "no Vulkan collectives over devices that share memory");
        require(d0->submitter() && d0->submitter() == d1->submitter() && d1->submitter() == d2->submitter(), "groups sharing a member are not on one queue thread");
        a.reset();
        require(!d0->submitter() && d1->submitter() && d2->submitter(), "a group gone did not give back the member no other group holds, or gave back a shared one");
        b.reset();
        require(!d1->submitter() && !d2->submitter(), "the last group over its members did not give their queue calls back");
        checks += 3;
    }
    {
        auto a = d0->join({d0, d1}, rows, 1);
        auto b = d2->join({d2, d1}, rows, 1);
        require(a && b && d2->submitter() == d0->submitter(), "a group whose first member is fresh did not take its other member's queue thread");
        ++checks;
        if (devices.size() >= 4) {
            backend::VulkanBackend* d3 = devices[3].get();
            b.reset();
            auto c = d2->join({d2, d3}, rows, 1);
            const auto s0 = d0->submitter(), s2 = d2->submitter();
            bool refused = false;
            try {
                auto x = d1->join({d1, d2}, rows, 1);
            } catch (const std::runtime_error&) {
                refused = true;
            }
            require(refused && d0->submitter() == s0 && d1->submitter() == s0 && d2->submitter() == s2 && d3->submitter() == s2, "a group over two groups' queue threads was not refused, or the refusal moved a member");
            ++checks;
        }
    }
    require(!d0->submitter() && !d1->submitter() && !d2->submitter() && (devices.size() < 4 || !devices[3]->submitter()), "groups gone left a member on a queue thread");
    auto c = d0->join({d0, d1}, rows, 1);
    std::vector<backend::Backend*> group{d0, d1};
    checks += group_sums::check(group, *c, 1, {1, 255}, 9400u, [](bool ok, const std::string& what) { require(ok, what.c_str()); });
    return checks + 1;
}

// A construction that throws part way through making the tagged route: one member's command pool refused, every pool and receive timeline made is destroyed, the members get their queue calls back, and a later group over them sums.
int pools_made = 0, pools_gone = 0, timelines_made = 0, timelines_gone = 0;
VkResult VKAPI_PTR refuse_pool(VkDevice, const VkCommandPoolCreateInfo*, const VkAllocationCallbacks*, VkCommandPool*) { return VK_ERROR_OUT_OF_DEVICE_MEMORY; }
PFN_vkCreateCommandPool real_create_pool;
PFN_vkDestroyCommandPool real_destroy_pool;
PFN_vkCreateSemaphore real_create_semaphore;
PFN_vkDestroySemaphore real_destroy_semaphore;
VkResult VKAPI_PTR count_pool(VkDevice d, const VkCommandPoolCreateInfo* i, const VkAllocationCallbacks* a, VkCommandPool* p) {
    const VkResult r = real_create_pool(d, i, a, p);
    pools_made += r == VK_SUCCESS;
    return r;
}
void VKAPI_PTR uncount_pool(VkDevice d, VkCommandPool p, const VkAllocationCallbacks* a) {
    pools_gone += p != VK_NULL_HANDLE;
    real_destroy_pool(d, p, a);
}
VkResult VKAPI_PTR count_semaphore(VkDevice d, const VkSemaphoreCreateInfo* i, const VkAllocationCallbacks* a, VkSemaphore* s) {
    const VkResult r = real_create_semaphore(d, i, a, s);
    timelines_made += r == VK_SUCCESS;
    return r;
}
void VKAPI_PTR uncount_semaphore(VkDevice d, VkSemaphore s, const VkAllocationCallbacks* a) {
    timelines_gone += s != VK_NULL_HANDLE;
    real_destroy_semaphore(d, s, a);
}

size_t check_tag_throws(const std::vector<std::unique_ptr<backend::VulkanBackend>>& devices) {
    constexpr size_t rows = 4096, limit = 3072;
    if (devices.size() < 2) {
        std::cout << "vulkan-sum: skip a construction throwing in the tagged route, fewer than two devices\n";
        return 0;
    }
    backend::VulkanBackend *d0 = devices[0].get(), *d1 = devices[1].get();
    for (backend::VulkanBackend* d : {d0, d1}) {
        const backend::Device& v = *d->device();
        if (!v.receive_queue || !v.device_clock || !v.uncached_memory || !(v.timestamp_ns > 0)) {
            std::cout << "vulkan-sum: skip a construction throwing in the tagged route, a member lacking a receive queue, a clock or uncached memory\n";
            return 0;
        }
    }
    backend::Fn saved[2] = {d0->device()->fn, d1->device()->fn};
    real_create_pool = saved[0].vkCreateCommandPool;
    real_destroy_pool = saved[0].vkDestroyCommandPool;
    real_create_semaphore = saved[0].vkCreateSemaphore;
    real_destroy_semaphore = saved[0].vkDestroySemaphore;
    for (backend::VulkanBackend* d : {d0, d1}) {
        backend::Fn& fn = d->device()->fn;
        fn.vkCreateCommandPool = d == d1 ? refuse_pool : count_pool;
        fn.vkDestroyCommandPool = uncount_pool;
        fn.vkCreateSemaphore = count_semaphore;
        fn.vkDestroySemaphore = uncount_semaphore;
    }
    bool threw = false;
    try {
        backend::VulkanCollective c({d0, d1}, rows, 1, limit);
    } catch (const std::exception&) {
        threw = true;
    }
    d0->device()->fn = saved[0];
    d1->device()->fn = saved[1];
    require(threw, "a refused command pool did not fail the collective's construction");
    require(pools_made == 1 && pools_gone == pools_made, "a construction that threw left a receive command pool");
    require(timelines_made > 0 && timelines_gone == timelines_made, "a construction that threw left a semaphore");
    require(!d0->submitter() && !d1->submitter(), "a construction that threw left its members on its queue thread");
    auto c = d0->join({d0, d1}, rows, 1);
    std::vector<backend::Backend*> group{d0, d1};
    require(c && backend::VulkanSumTest::tag(*c, limit), "no tagged collective after a construction that threw");
    const size_t checks = group_sums::check(group, *c, 1, {3, 766}, 9500u, [](bool ok, const std::string& what) { require(ok, what.c_str()); });
    return checks + 4;
}

bool injected(backend::VulkanBackend& d) {
    try {
        d.submit();
    } catch (const std::exception& e) {
        return std::strstr(e.what(), "injected") != nullptr;
    }
    return false;
}

// A queue-thread failure while a tagged batch is open is kept by a wait and by a group's teardown for the next hand-over, and a member alone after it still works.
// A sum failing part way closes the group's whole open batch, so a group sharing its member does not wait out the cap behind its pushes.
size_t check_teardown_faults(const std::vector<std::unique_ptr<backend::VulkanBackend>>& devices) {
    constexpr size_t limit = 3072, rows = 4096;
    if (devices.size() < 3) {
        std::cout << "vulkan-sum: skip teardown faults of two tagged groups sharing a member, fewer than three devices\n";
        return 0;
    }
    backend::VulkanBackend *d0 = devices[0].get(), *d1 = devices[1].get(), *d2 = devices[2].get();
    size_t checks = 0;
    for (int teardown = 0; teardown < 2; ++teardown) {
        auto a = d0->join({d0, d1}, rows, 1);
        auto b = d1->join({d1, d2}, rows, 1);
        require(a && b, "no Vulkan collectives over devices that share memory");
        if (!backend::VulkanSumTest::tag(*a, limit) || !backend::VulkanSumTest::tag(*b, limit)) {
            std::cout << "vulkan-sum: skip teardown faults of two tagged groups sharing a member, a member lacking a receive queue, a clock or uncached memory\n";
            return 0;
        }
        a->sum_in_place(766, 1);
        backend::VulkanSumTest::inject(*d0);
        if (teardown) a.reset();
        else a->wait(0);
        require(injected(teardown ? *d1 : *d0), teardown ? "a tagged group's teardown lost a queue-thread failure" : "a wait closing a tagged batch lost a queue-thread failure");
        ++checks;
        if (teardown) {
            const float one = 1.0f;
            float got = 0.0f;
            backend::BufferPtr x = d0->alloc(2 * sizeof(float), backend::Memory::device);
            d0->write(*x, 0, &one, sizeof(float));
            d0->copy(*x, sizeof(float), *x, 0, sizeof(float));
            d0->read(*x, sizeof(float), &got, sizeof(float));
            require(got == one, "a member alone after its group's teardown with a failure does not work");
            ++checks;
        }
    }
    auto a = d0->join({d0, d1}, rows, 1);
    auto b = d1->join({d1, d2}, rows, 1);
    std::vector<backend::Backend*> ga{d0, d1}, gb{d1, d2};
    require(a && b && backend::VulkanSumTest::tag(*a, limit) && backend::VulkanSumTest::tag(*b, limit), "no tagged collectives after a queue-thread failure");
    const group_sums::Case ka = group_sums::make(2, rows, 9600u), kb = group_sums::make(2, rows, 9700u);
    std::vector<std::vector<float>> wa = ka.partial, wb = kb.partial;
    for (size_t m = 0; m < 2; ++m) {
        const backend::Slice pa = a->partial(m), pb = b->partial(m);
        ga[m]->write(*pa.buffer, pa.offset * sizeof(float), ka.partial[m].data(), rows * sizeof(float));
        gb[m]->write(*pb.buffer, pb.offset * sizeof(float), kb.partial[m].data(), rows * sizeof(float));
    }
    const auto add = [](std::vector<std::vector<float>>& w, size_t n) {
        for (size_t j = 0; j < n; ++j) w[0][j] = w[1][j] = w[0][j] + w[1][j];
    };
    a->sum_in_place(766, 1);
    add(wa, 766);
    b->sum_in_place(766, 1);
    add(wb, 766);
    const auto start = std::chrono::steady_clock::now();
    backend::VulkanSumTest::release(*a);
    std::vector<float> got(rows);
    for (size_t m = 0; m < 2; ++m) {
        const backend::Slice pb = b->partial(m);
        gb[m]->read(*pb.buffer, pb.offset * sizeof(float), got.data(), rows * sizeof(float));
        require(!std::memcmp(got.data(), wb[m].data(), rows * sizeof(float)), "a tagged group beside one whose sum failed is not the oracle's");
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    require(ms < 500.0, "a sum failing part way left its tagged pushes on a shared member to wait out the cap");
    a->sum_in_place(rows, 1);
    add(wa, rows);
    a->sum_in_place(766, 1);
    add(wa, 766);
    d0->submit();
    for (size_t m = 0; m < 2; ++m) {
        const backend::Slice pa = a->partial(m);
        ga[m]->read(*pa.buffer, pa.offset * sizeof(float), got.data(), rows * sizeof(float));
        require(!std::memcmp(got.data(), wa[m].data(), rows * sizeof(float)), "a tagged group after its failed sum is undone is not the oracle's");
    }
    require(backend::VulkanSumTest::tagged_sums(*a) == 2 && backend::VulkanSumTest::tagged_sums(*b) == 1, "the tagged route did not take the sums beside a failed one");
    return checks + 5;
}

// The queue probes' settings: the backend whose call fails and whether that is a receivers' call, and the most calls seen at once.
const backend::VulkanBackend* probe_fails = nullptr;
bool probe_receivers = false;
std::atomic<int> probe_inside{0}, probe_most{0};

VkResult fail_one(const backend::VulkanBackend& b, bool receivers) {
    return &b == probe_fails && receivers == probe_receivers ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS;
}
// Each call held up to 200 ms for a second one beside it.
VkResult overlap(const backend::VulkanBackend&, bool) {
    const int now = ++probe_inside;
    int most = probe_most.load();
    while (now > most && !probe_most.compare_exchange_weak(most, now)) {}
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
    while (probe_most.load() < 2 && std::chrono::steady_clock::now() < end) std::this_thread::yield();
    --probe_inside;
    return VK_SUCCESS;
}

bool given_out(backend::VulkanBackend& d) {
    try {
        d.submit();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

// A close of two tagged groups sharing a member makes its three members' queue calls side by side.
// A close failing on one member leaves the others' queued submissions made, and gives up receivers only where none reached the queue.
size_t check_submit_faults(const std::vector<std::unique_ptr<backend::VulkanBackend>>& devices) {
    constexpr size_t limit = 3072, rows = 4096;
    if (devices.size() < 3) {
        std::cout << "vulkan-sum: skip a close's queue calls of two tagged groups sharing a member, fewer than three devices\n";
        return 0;
    }
    backend::VulkanBackend *d0 = devices[0].get(), *d1 = devices[1].get(), *d2 = devices[2].get();
    using T = backend::VulkanSumTest;
    std::string missed;
    size_t checks = 0;
    const auto expect = [&](bool ok, const char* what) {
        if (!ok) missed += std::string(missed.empty() ? "" : "; ") + what;
        ++checks;
    };
    for (int step = 0; step < 3; ++step) {
        auto a = d0->join({d0, d1}, rows, 1);
        auto b = d1->join({d1, d2}, rows, 1);
        require(a && b, "no Vulkan collectives over devices that share memory");
        if (!T::tag(*a, limit) || !T::tag(*b, limit)) {
            std::cout << "vulkan-sum: skip a close's queue calls of two tagged groups sharing a member, a member lacking a receive queue, a clock or uncached memory\n";
            return checks;
        }
        a->sum_in_place(766, 1);
        b->sum_in_place(766, 1);
        VkSemaphore gate = VK_NULL_HANDLE;
        if (step == 0) {
            probe_most = 0;
            T::probe(overlap);
            T::close(*d0);
            T::probe(nullptr);
            expect(probe_most.load() >= 2, "a close of two tagged groups sharing a member made its members' queue calls one after another");
            expect(!given_out(*d0), "a close of two tagged groups sharing a member failed");
        } else {
            if (step == 2) gate = T::gate(*d1);
            const size_t signals = T::host_signals();
            probe_fails = d1;
            probe_receivers = step == 2;
            T::probe(fail_one);
            T::close(*d0);
            T::probe(nullptr);
            if (step == 1) {
                expect(!d0->lost(T::taken(*a, 0)) && !d2->lost(T::taken(*b, 1)), "a close failing on one member recorded the others' queued submissions lost");
                expect(d1->lost(T::taken(*a, 1)), "a close failing on one member did not record its submission lost");
            } else {
                // The first group's second member and both of the second group's never reached a receive queue, so three are given up from the host.
                expect(T::host_signals() - signals == 3, "a failed close gave up from the host receivers already on their receive queue");
                const bool early = T::received(*a, 0, 0);
                T::open_gate(*d1, gate);
                expect(!early, "a failed close signalled receivers from the host while their submission was pending");
                expect(T::received(*a, 0, 3000000000ull), "receivers submitted before a failed close never signalled");
            }
            expect(given_out(*d0), "a failed close lost its failure");
        }
        a.reset();
        b.reset();
        if (gate) T::destroy(*d1, gate);
    }
    require(missed.empty(), missed.c_str());
    return checks;
}

} // namespace

int main() {
    // A lost device aborts, so each check's line is written as it finishes.
    std::cout << std::unitbuf;
    try {
        auto owner = std::make_unique<backend::VulkanBackend>(0);
        const size_t checks = check_sources(*owner);
        std::cout << "vulkan-sum: " << checks << " sums of 1 to " << backend::kMaxTensorWidth << " sources in member order\n";
        if (!backend::VulkanSumTest::device_clock(*owner)) {
            std::cout << "vulkan-sum: no flagged sums, the device's shaders reading no clock\n";
            return 0;
        }
        const size_t flagged = check_flagged(*owner), faults = check_flag_faults(*owner);
        std::cout << "vulkan-sum: " << flagged << " flagged sums of 2 to " << backend::kMaxTensorWidth << " members over 20 epochs, " << faults << " fault checks\n";
        const size_t tagged = check_tagged(*owner), tagged_faults = check_tagged_faults(*owner);
        std::cout << "vulkan-sum: " << tagged << " tagged sums of 2 to " << backend::kMaxTensorWidth << " members over 20 epochs, " << tagged_faults << " fault checks\n";
        std::vector<std::unique_ptr<backend::VulkanBackend>> devices;
        devices.push_back(std::move(owner));
        for (int i = 1; i < 4; ++i) {
            try {
                devices.push_back(std::make_unique<backend::VulkanBackend>(i));
            } catch (const std::exception&) {
                break;
            }
        }
        const size_t routed = check_route(devices);
        std::cout << "vulkan-sum: " << routed << " collective checks over " << devices.size() << " devices\n";
        check_tagged_barriers(devices);
        const size_t shared = check_shared(devices);
        std::cout << "vulkan-sum: " << shared << " checks of two tagged groups sharing a member, ordered and unordered\n";
        const size_t routes = check_routes(devices), thrown = check_tag_throws(devices);
        std::cout << "vulkan-sum: " << routes << " checks of groups sharing members giving their queues back, " << thrown << " of a construction throwing in the tagged route\n";
        const size_t teardown = check_teardown_faults(devices);
        std::cout << "vulkan-sum: " << teardown << " checks of a queue-thread failure and a failed sum beside an open tagged batch\n";
        const size_t submit = check_submit_faults(devices);
        std::cout << "vulkan-sum: " << submit << " checks of a close's queue calls side by side and failing on one member\n";
        return 0;
    } catch (const backend::VulkanUnavailable& e) {
        std::cerr << e.what() << '\n';
        return 77;
    } catch (const std::exception& e) {
        std::cerr << "vulkan-sum: " << e.what() << '\n';
        return 1;
    }
}
