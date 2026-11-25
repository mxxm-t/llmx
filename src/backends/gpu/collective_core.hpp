#pragma once
// A tensor group's collective over a vendor's devices and GpuBackend's part in it; backend_core.hpp includes it after the class.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#endif

namespace backend {
namespace {

constexpr size_t kGroupMembers = kMaxTensorWidth;   // the kernel's kGroupMembers (kernels/f32.gpu)

// The kernel's view of the group (kernels/f32.gpu, GroupPeers), written once into each member's device memory and passed by address.
struct GroupPeers {
    float* stage[kGroupMembers];
    unsigned* signal[kGroupMembers];
    unsigned* status;
    unsigned* fault;
    unsigned* cancel;
};
static_assert(kGroupMembers == 16 && sizeof(GroupPeers) == 280 && offsetof(GroupPeers, status) == 256 && offsetof(GroupPeers, cancel) == 272, "kernels/f32.gpu, GroupPeers");
constexpr size_t kGroupBlocks = 32;       // the kernel's signal slots a member, so the most blocks a chain or meeting takes
constexpr unsigned kPushTile = 16;        // blocks of a pushing held build whose rows go out together (kernels/int16.gpu, held_push)
constexpr size_t kChainBlocks = 16, kChainUnits = 4096;  // the kernel's chain blocks each way and float4 units a chunk
constexpr unsigned kProbeTimeoutMs = 1000;   // how long a probe sum's member waits for its peers before the probe finds peer writes unseen
constexpr unsigned kSumTimeoutMs = 30000;    // how long any other sum's member waits for a peer before it takes the peer's card as gone
constexpr size_t kTimedFloor = 4096, kTimedSums = 3, kTimedRounds = 3;   // the least floats a form is timed at, the sums of each timing and its rounds
constexpr double kTimedSpanMs = 0.5;   // a short timing's rounds each span about this long, so a few microseconds of host jitter do not pick the form after a first
constexpr size_t kDuelRounds = 5;      // interleaved timings of the wrap and the best other pick before the wrap is kept
constexpr double kWrapMargin = 0.9;    // the wrap's median must be under this fraction of the other pick's
constexpr double kCapSpan = 8;   // how many times the best pick's time a kernel form may take at its own blocks and still have its cap timed, since fewer blocks summed up to 4.1x faster on a slow link
// The forms a join times for each staged transport, in the order stderr lists their times; copies and pulls only where the rows go through host memory, and last.
constexpr GpuSumForm kTimedForms[] = {GpuSumForm::one_shot, GpuSumForm::chain, GpuSumForm::wrap, GpuSumForm::shares, GpuSumForm::copies, GpuSumForm::pulls};
// Copies send a share in at most kCopyChunks chunks, by default one a kCopyChunkFloats, 4 MB a copy, so a chunk's issue and count cost little beside its transfer (docs/src/backends-gpu.md, the collective).
constexpr size_t kCopyChunks = 16, kCopyChunkFloats = 1048576;
constexpr size_t kCopyWords = 8;   // a member's words for copies, a cache line: rows sent, totals sent, abort (kernels/f32.gpu, llmx_group_copies)
// The most blocks a lane's sum beside a member's work takes, since more slow the part computing beside it more than they speed the sum, and the fewest rows of a lane worth sending beside (docs/TENSOR-SPLIT.md, section 4.3).
constexpr size_t kBesideBlocks = 4, kBesideRows = 128;
// The form a lane's sum beside takes: the picks by size are timed uncapped, and a lane runs capped at kBesideBlocks (docs/TENSOR-SPLIT.md, section 4.3).
constexpr GpuSumForm kLaneForm = GpuSumForm::shares;

#if defined(__linux__)
// The CPUs of the NUMA node a PCI device sits on, empty where the system does not say.
inline std::vector<int> node_cpus(const std::string& pci_bus) {
    std::string bus = pci_bus;
    for (char& c : bus) c = (char)std::tolower((unsigned char)c);
    std::FILE* f = std::fopen(("/sys/bus/pci/devices/" + bus + "/local_cpulist").c_str(), "r");
    if (!f) return {};
    char line[4096] = {};
    const bool read = std::fgets(line, sizeof line, f) != nullptr;
    std::fclose(f);
    std::vector<int> cpus;
    for (char* q = line; read && *q && *q != '\n';) {
        char* end = nullptr;
        const long a = std::strtol(q, &end, 10);
        long b = a;
        if (end == q) return {};
        if (*end == '-') {
            q = end + 1;
            b = std::strtol(q, &end, 10);
            if (end == q || b < a) return {};
        }
        for (long c = a; c <= b; ++c) cpus.push_back((int)c);
        q = *end == ',' ? end + 1 : end;
    }
    return cpus;
}

// Pinned host memory on the NUMA node of the device at `pci_bus`, in huge pages where the kernel gives them, first touched by a thread bound to that node's CPUs and then registered, or null with why not.
// `bytes` rounds up to whole huge pages, since 4 KB pages cut a group's host stores across four RTX 3080s about fivefold (docs/src/backends-gpu.md, the collective).
template <class Api>
void* placed_host(size_t& bytes, const std::string& pci_bus, std::string& why) {
    const std::vector<int> cpus = node_cpus(pci_bus);
    constexpr size_t kHuge = size_t(2) << 20;
    bytes = size_add(bytes, kHuge - 1) / kHuge * kHuge;
    char* const map = (char*)mmap(nullptr, bytes + kHuge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (map == (char*)MAP_FAILED) {
        why = "mmap failed";
        return nullptr;
    }
    char* const p = map + (kHuge - (uintptr_t)map % kHuge) % kHuge;
    if (p > map) munmap(map, p - map);
    munmap(p + bytes, map + bytes + kHuge - (p + bytes));
    if (madvise(p, bytes, MADV_HUGEPAGE) != 0) {
        static std::atomic<bool> told{false};
        if (!told.exchange(true)) std::fprintf(stderr, "%sa tensor group's host memory is in 4 KB pages, since the kernel refused huge pages\n", Api::prefix.c_str());
    }
    // Where the thread cannot be bound to the node's CPUs, the pages are touched unbound and stay huge, which still stores several times faster than 4 KB pages on the node.
    std::thread touch([&] {
        cpu_set_t set;
        CPU_ZERO(&set);
        for (int c : cpus)
            if (c >= 0 && c < CPU_SETSIZE) CPU_SET(c, &set);
        if (!cpus.empty()) (void)pthread_setaffinity_np(pthread_self(), sizeof set, &set);
        std::memset(p, 0, bytes);
    });
    touch.join();
    if (!register_host<Api>(p, bytes)) {
        why = "registering it failed";
        munmap(p, bytes);
        return nullptr;
    }
    return p;
}
#else
template <class Api>
void* placed_host(size_t&, const std::string&, std::string& why) {
    why = "only Linux places it";
    return nullptr;
}
#endif
constexpr double kOrderMargin = 0.04;    // how much faster than the groups it replaces a swap of devices must sum to be taken, so even links and noise keep the listing

const char* transport_name(GroupTransport t) { return t == GroupTransport::peer_writes ? "peer writes" : t == GroupTransport::host_writes ? "host writes" : "peer pulls"; }

// A tensor group's ordered sum (docs/src/backends-gpu.md, the collective): each member's kernel, on its own stream, sends its partial rows to its peers in the transport and form the sum's size takes and adds them in member order.
// A member waits for a late peer up to kSumTimeoutMs of its device's clock, and a sum that ran out or the host cancelled turns the collective to the conservative path for good, where each member pulls the peers' rows behind events.
template <class Api>
class GpuCollective final : public Collective {
public:
    // `transports` are the fast ones it stages for, peer writes first, or pulls alone; a probe bounds its members' waits by kProbeTimeoutMs, and `force` takes one pick for every fast sum.
    // `copies` stages and times copies under host writes, where the backend has them (GroupPath::copies).
    GpuCollective(const std::vector<GpuBackend<Api>*>& members, size_t rows, size_t width, const std::vector<GroupTransport>& transports, bool probe = false,
                  std::optional<GpuSumPick> force = std::nullopt, bool copies = false)
        : members_(members), rows_(rows), width_(width), cap_((size_mul(rows, width) + 63) / 64 * 64), fast_(!transports.empty() && transports[0] != GroupTransport::pulls), copies_(Api::copy_sums && copies), force_(force),
          ready_(members.size(), nullptr), timeout_(members.size(), 0), tick_(members.size(), 0), last_(members.size(), 0), before_(members.size(), 0),
          copy_stream_(members.size(), nullptr), added_(members.size(), nullptr), sent_(members.size(), nullptr) {
        if (transports.empty()) throw std::logic_error("backend: a collective of no transport");
        const size_t W = members.size();
        try {
            Api::check(Api::host_malloc((void**)&status_, (W + 1) * sizeof(unsigned), true), "host_malloc");
            for (size_t m = 0; m <= W; ++m) status_[m] = 0;
            for (size_t m = 0; m < W; ++m) {
                GpuBackend<Api>& b = *members[m];
                partial_.push_back(b.alloc(size_mul(size_mul(rows, width), sizeof(float)), Memory::device));
                Api::check(Api::set_device(b.dev_.index), "set_device");
                Api::check(Api::event_create(&ready_[m], false), "event_create");
                int khz = 0;
                Api::check(Api::wall_clock_khz(&khz, b.dev_.index), "wall_clock_khz");
                timeout_[m] = (unsigned long long)khz * (probe ? kProbeTimeoutMs : kSumTimeoutMs);
                tick_[m] = (unsigned long long)khz;
                if (fast_ && W > 1) {
                    beside_.resize(W, nullptr);
                    computed_.resize(W, nullptr);
                    for (auto& e : summed_) e.resize(W, nullptr);
                    Api::check(Api::stream_create(&beside_[m]), "stream_create");
                    Api::check(Api::event_create(&computed_[m], false), "event_create");
                    for (auto& e : summed_) Api::check(Api::event_create(&e[m], false), "event_create");
                }
            }
            // The one-shot takes at most a block a multiprocessor of the devices its members share, so every members blocks stay resident while they wait for their peers; the timing lowers it where fewer win.
            for (size_t m = 0; m < W; ++m) {
                size_t sharing = 0;
                for (GpuBackend<Api>* p : members) sharing += p->dev_.index == members[m]->dev_.index;
                one_shot_blocks_ = std::min<size_t>(one_shot_blocks_, std::max<size_t>(1, members[m]->dev_.caps.compute_units / sharing));
            }
            for (GroupTransport t : transports) stage_for(t);
            // A tile counter for every 64 rows, the fewest a tile holds, where the device's held builds push a group of this width over peer writes (DeviceProfile::held_push_members).
            const uint32_t push = members[0]->dev_.profile.held_push_members;
            if (push && W >= push && std::any_of(sets_.begin(), sets_.end(), [](const Areas& s) { return s.fast() && !s.host(); }))
                for (GpuBackend<Api>* b : members) {
                    const size_t bytes = (cap_ + 63) / 64 * sizeof(unsigned);
                    tile_bufs_.push_back(b->alloc(bytes, Memory::device));
                    tiles_.push_back((unsigned*)as_gpu<Api>(*tile_bufs_.back()).data());
                    b->hold_fill(tiles_.back(), bytes);
                }
            // Every member's signals are zero and its slots armed before any peer may write them, which a fill still queued on its own stream would overwrite.
            for (GpuBackend<Api>* b : members) b->sync();
        } catch (...) {
            release();
            throw;
        }
    }
    ~GpuCollective() override { release(); }

    Slice partial(size_t member) override { return {partial_.at(member).get(), 0}; }

    // The members' submissions that hold the sum, made here where the caller has not yet submitted; a sum that gave up turns the collective conservative.
    // A member whose device reported an error cancels the sum, so its peers stop waiting for it now rather than at their deadline.
    void wait(size_t back) noexcept override {
        const std::vector<Ticket>& sum = back ? before_ : last_;
        for (size_t m = 0; m < sum.size(); ++m) {
            GpuBackend<Api>& b = *members_[m];
            if (sum[m] > b.tickets_.last()) {
                try { b.record(); } catch (const std::exception&) { continue; }
            }
            b.wait(sum[m]);
            if (b.device_failed()) cancel();
        }
        settle();
    }

    void sum_into(const std::vector<Slice>& residual, size_t rows, size_t width) override {
        if (residual.size() != members_.size() || rows > rows_ || width != width_)
            throw std::logic_error("backend: a sum of other members, rows or width than its collective was made for");
        one_call();
        summed(residual, size_mul(rows, width), false);
    }

    // Each member's own partial rows are its residual, which the sum replaces; a generated row's sum folds its take into the next mix only on members whose mixes take the entry.
    void sum_in_place(size_t rows, size_t width) override {
        const size_t n = size_mul(rows, width);
        if (n > size_mul(rows_, width_)) throw std::logic_error("backend: a sum past the rows its collective was made for");
        one_call();
        std::vector<Slice> own;
        for (size_t m = 0; m < members_.size(); ++m) own.push_back(partial(m));
        const bool mixes = std::all_of(members_.begin(), members_.end(), [](const GpuBackend<Api>* b) { return b->entry_mixes_; });
        summed(own, n, true, rows == 1 && width % GpuBackend<Api>::kEntryGroup == 0 && mixes);
    }

    // A lane's sum crosses on a second stream of each member, behind an event of the member's work before it, in a kernel form with at most kBesideBlocks blocks; its finish holds each member's stream for it.
    // A part is worth cutting from kBesideRows rows at two or three members and half that from four, below which the parts lose to the whole pass (docs/TENSOR-SPLIT.md, section 4.3).
    bool copies_beside(size_t floats) const override {
        const size_t least = members_.size() >= 4 ? kBesideRows / 2 : kBesideRows;
        return fast_ && !beside_.empty() && floats >= size_mul(least, width_) && kernel_form(lane_pick(floats).form);
    }

    // The lane's rows summed in place, beside the members' streams where its pick is a kernel form on the fast path, else on them at once; the kernel reads float4 units from `base`.
    void send(size_t lane, size_t base, size_t rows, size_t width) override {
        const size_t W = members_.size(), n = size_mul(rows, width), from = size_mul(base, width);
        if (lane > 1 || size_add(base, rows) > rows_ || width != width_ || lane_sent_[lane] || from % 4) throw std::logic_error("backend: a sum of a lane, rows or width its collective was not made for");
        std::vector<Slice> own;
        for (size_t m = 0; m < W; ++m) own.push_back({partial_[m].get(), from});
        const bool beside = fast_ && !beside_.empty() && kernel_form(lane_pick(n).form);
        // A lane summed on the members' streams follows the other lane's sum beside, which may still read the staging it writes.
        if (!beside && lane_sent_[lane ^ 1] && lane_beside_[lane ^ 1])
            for (size_t m = 0; m < W; ++m) members_[m]->hold_wait_event(summed_[lane ^ 1][m]);
        lane_sent_[lane] = true;
        lane_beside_[lane] = beside;
        summed(own, n, true, false, from, beside ? (int)lane : -1);
    }

    // Each member's stream waits for the lane's sum on its device, then adds the summed rows to residual[m] where given, one add of the sum as sum_into makes it.
    void finish(size_t lane, size_t base, size_t rows, size_t width, const std::vector<Slice>* residual) override {
        const size_t W = members_.size(), n = size_mul(rows, width), from = size_mul(base, width);
        if (lane > 1 || !lane_sent_[lane] || (residual && residual->size() != W)) throw std::logic_error("backend: a sum finished that was not sent, or of other members");
        lane_sent_[lane] = false;
        for (size_t m = 0; m < W; ++m) {
            GpuBackend<Api>& b = *members_[m];
            if (lane_beside_[lane]) {
                b.hold_wait_event(summed_[lane][m]);
                b.forget_twins();
            }
            if (residual) b.add((*residual)[m], CSlice{partial_[m].get(), from}, n);
        }
        before_ = last_;
        for (size_t m = 0; m < W; ++m) last_[m] = members_[m]->tickets_.last() + 1;
    }

    // A lane's sum still beside may read the staging the sums after it write, so the members' streams wait for it.
    void drop() noexcept override {
        for (size_t k = 0; k < 2; ++k)
            for (size_t m = 0; m < members_.size() && lane_sent_[k] && lane_beside_[k]; ++m) try { members_[m]->hold_wait_event(summed_[k][m]); } catch (const std::exception&) {}
        lane_sent_[0] = lane_sent_[1] = false;
    }

    // Whether the sums still take the fast path; a test's witness.
    bool fast() const { return fast_; }
    // Every transport and form this collective can time: each staged transport's kernel forms, the wrap from three members, where it differs from the chain, and copies and pulls where the rows go through host memory.
    std::vector<GpuSumPick> picks() const {
        std::vector<GpuSumPick> all;
        for (const Areas& s : sets_)
            for (GpuSumForm f : kTimedForms)
                if (s.fast() && (f != GpuSumForm::wrap || members_.size() > 2) && ((f != GpuSumForm::pulls && f != GpuSumForm::copies) || (s.host() && (f != GpuSumForm::copies || copies_))))
                    all.push_back({s.transport, f});
        return all;
    }
    // The pinned host memory the collective holds, which a host's memory limit has to allow, placed areas at their whole huge pages.
    size_t pinned_bytes() const {
        size_t bytes = 0;
        for (const Areas& s : sets_)
            if (s.host())
                for (size_t m = 0; m < members_.size(); ++m) {
                    if (!s.area[m]) continue;
                    bytes = size_add(bytes, s.placed[m] ? s.placed[m] : page_round(area_bytes()));
                    if (m < s.copy.size()) bytes = size_add(bytes, s.copy_placed[m] ? s.copy_placed[m] : copy_bytes());
                }
        return bytes;
    }

    // Each fast sum's transport and form by its size, kept by the group's first member; a transport no size picks gives up its staging, so the group holds only what its sums use.
    void use(const std::map<size_t, GpuSumPick>& forms) {
        forms_ = forms;
        if (!fast_ || force_) return;
        unsigned picked = forms.empty() ? transport_bit(sets_[0].transport) : 0;
        for (const auto& [n, p] : forms) picked |= transport_bit(areas(p.transport).transport);
        for (GpuBackend<Api>* b : members_) b->sync();
        for (size_t s = sets_.size(); s-- > 0;)
            if (!(picked & transport_bit(sets_[s].transport))) {
                free_areas(sets_[s]);
                sets_.erase(sets_.begin() + (std::ptrdiff_t)s);
            }
    }

    // The milliseconds a sum of kTimedSums sums in place of n floats by `pick`, issued back to back after a first and waited for once as a pass issues them; unbounded where a sum gave up.
    double timed(GpuSumPick pick, size_t n) {
        std::vector<Slice> own;
        for (size_t m = 0; m < members_.size(); ++m) own.push_back(partial(m));
        const std::optional<GpuSumPick> kept = force_;
        force_ = pick;
        summed(own, n, true);
        wait(0);
        double ms = 1e30;
        // Sums shorter than the span are timed in further rounds of as many sums as fill it, the least round kept.
        for (size_t round = 0, sums = kTimedSums; fast_ && round < kTimedRounds; ++round) {
            const auto t0 = std::chrono::steady_clock::now();
            for (size_t k = 0; fast_ && k < sums; ++k) summed(own, n, true);
            wait(0);
            const double t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / (double)sums;
            ms = std::min(ms, t);
            if (t * (double)sums >= kTimedSpanMs) break;
            sums = std::max<size_t>(kTimedSums, (size_t)std::ceil(kTimedSpanMs / std::max(t, 1e-3)));
        }
        force_ = kept;
        return fast_ ? ms : 1e30;
    }

    // Every pick timed at the reserved rows and halvings of them in whole rows down to one row, and capped kernel forms beside them, until n reaches what was timed before or kTimedFloor; each n's winner goes into `forms`.
    // Returns the times as stderr reports them, a transport's forms beside its name and the winner last with its blocks where fewer won, empty where nothing was timed.
    std::string time_forms(std::map<size_t, GpuSumPick>& forms, size_t timed_before) {
        const std::vector<GpuSumPick> all = picks();
        std::string report;
        size_t above = 0;   // the size timed before this one, twice it or so, with its winner and that winner's time
        GpuSumPick above_pick{};
        double above_ms = 0;
        for (size_t r = rows_, n = size_mul(r, width_); fast_ && r && n > timed_before && n >= kTimedFloor; r /= 2, n = r * width_) {
            std::vector<double> ms(all.size());
            size_t won = 0;
            for (size_t p = 0; p < all.size(); ++p) {
                ms[p] = timed(all[p], n);
                if (ms[p] < ms[won]) won = p;
            }
            if (!fast_) break;
            GpuSumPick best = all[won], other{};
            double best_ms = ms[won], other_ms = 1e30;   // other: the best pick that is not the wrap, which the wrap has to beat by kWrapMargin
            for (size_t p = 0; p < all.size(); ++p)
                if (all[p].form != GpuSumForm::wrap && ms[p] < other_ms) {
                    other = all[p];
                    other_ms = ms[p];
                }
            // A link that takes few writers at once sums faster from fewer blocks (docs/src/backends-gpu.md, the pick), so every kernel form within kCapSpan of the best has its cap halved while that is faster, then tried at 3/4 and 3/2 of the kept cap.
            for (size_t kernel = 0; kernel < all.size(); ++kernel) {
                if (all[kernel].form == GpuSumForm::pulls || ms[kernel] > kCapSpan * ms[won]) continue;
                GpuSumPick capped = all[kernel];
                double capped_ms = ms[kernel];
                for (unsigned b = (unsigned)launched_blocks(capped, n) / 2; fast_ && b >= 1; b /= 2) {
                    GpuSumPick fewer = capped;
                    fewer.blocks = b;
                    const double t = timed(fewer, n);
                    if (t >= capped_ms) break;
                    capped = fewer;
                    capped_ms = t;
                }
                // One step between the halving's cap and its neighbours, the cap timed again beside them, finds a cap the halving steps over.
                const unsigned own = (unsigned)launched_blocks(all[kernel], n), b = (unsigned)launched_blocks(capped, n);
                if (fast_) capped_ms = std::min(capped_ms, timed(capped, n));
                for (const unsigned m : {b * 3 / 4, b * 3 / 2}) {
                    if (!fast_ || m < 1 || m == b || m >= own) continue;
                    GpuSumPick step = capped;
                    step.blocks = m;
                    const double t = timed(step, n);
                    if (t < capped_ms) {
                        capped = step;
                        capped_ms = t;
                    }
                }
                if (capped_ms < best_ms) {
                    best = capped;
                    best_ms = capped_ms;
                }
                if (capped.form != GpuSumForm::wrap && capped_ms < other_ms) {
                    other = capped;
                    other_ms = capped_ms;
                }
            }
            // The wrap is kept only where the median of kDuelRounds timings, each beside one of the best other pick, is under kWrapMargin of that pick's median, so noise does not swap two forms whose links load differently.
            if (fast_ && best.form == GpuSumForm::wrap && other_ms < 1e30) {
                std::vector<double> w, o;
                for (size_t duel = 0; fast_ && duel < kDuelRounds; ++duel) {
                    const bool first = duel % 2 == 0;   // alternated, so neither pick always runs after the other
                    if (first) w.push_back(timed(best, n));
                    o.push_back(timed(other, n));
                    if (!first) w.push_back(timed(best, n));
                }
                if (!fast_) break;
                std::sort(w.begin(), w.end());
                std::sort(o.begin(), o.end());
                if (w[w.size() / 2] <= kWrapMargin * o[o.size() / 2]) {
                    best_ms = w[w.size() / 2];
                } else {
                    best = other;
                    best_ms = o[o.size() / 2];
                }
            }
            if (!fast_) break;
            forms[n] = best;
            // Between two timed sizes whose winners differ, the larger size's winner takes over where the two winners' times, each taken as linear in n between the sizes, cross, and just past n where it ties n's winner at n.
            // A wrap on either side takes over only at the timed size, since the single timings of the crossing would carry it past its duel's margin.
            size_t from = 0;
            if (above && best.form != GpuSumForm::wrap && above_pick.form != GpuSumForm::wrap && (best.transport != above_pick.transport || best.form != above_pick.form || best.blocks != above_pick.blocks)) {
                const double gap = best_ms - timed(above_pick, n), gap_above = timed(best, above) - above_ms;
                if (!fast_) break;
                if (gap >= 0) from = n + 1;
                else if (gap_above > gap) from = n + (size_t)std::ceil((double)(above - n) * -gap / (gap_above - gap));
                if (from > n && from < above) forms[from] = above_pick;
                else from = 0;
            }
            if (from) report += ", from " + std::to_string(from) + " floats";
            above = n;
            above_pick = best;
            above_ms = best_ms;
            std::string line = (report.empty() ? "" : "; ") + std::to_string(n) + " floats";
            for (size_t p = 0; p < all.size(); ++p) {
                char t[24];
                std::snprintf(t, sizeof t, " %.3f", ms[p]);
                line += (p == 0 || all[p].transport != all[p - 1].transport ? std::string(p ? ", " : " ") + transport_name(all[p].transport) : std::string()) + t;
            }
            report += line + ": " + transport_name(best.transport) + " " + sum_form_name(best.form);
            if (best.blocks) {
                char t[48];
                std::snprintf(t, sizeof t, best.form == GpuSumForm::copies ? " in %u chunks, %.3f" : " of %u blocks, %.3f", best.blocks, best_ms);
                report += t;
            }
        }
        return report;
    }

    // Why some member's add tells a value from the twin the one-shot sends for the armed bits, empty where none does: the sanitized operands' sums must give the raw ones' bits.
    static std::string add_check(const std::vector<GpuBackend<Api>*>& group, const std::vector<int>& ids) {
        static const uint32_t patterns[] = {0x7f800001u, 0x7fc00001u, 0xff800001u, 0xffc00001u, 0x7fc00000u, 0x7f812345u, 0x7fd12345u, 0x3fc00000u, 0x80000000u, 0x0u, 0x7f800000u, 0xff800000u, 0x00000001u};
        constexpr int np = sizeof(patterns) / sizeof(patterns[0]);
        try {
            for (size_t m = 0; m < group.size(); ++m) {
                GpuBackend<Api>& b = *group[m];
                const BufferPtr pat = b.alloc(sizeof(patterns), Memory::device), differ = b.alloc(4 * sizeof(uint32_t), Memory::device);
                const uint32_t zero[4] = {0, 0, 0, 0};
                b.write(*pat, 0, patterns, sizeof(patterns));
                b.write(*differ, 0, zero, sizeof(zero));
                b.group_add_check(*pat, np, *differ);
                uint32_t got[4];
                b.read(*differ, 0, got, sizeof(got));
                if (got[0] || got[1] || got[2])
                    return "device " + std::to_string(ids[m]) + "'s add tells a sent value from its sanitized twin (a + b " + std::to_string(got[0]) + ", (a + b) + c " + std::to_string(got[1]) + ", c + (a + b) " +
                           std::to_string(got[2]) + " of " + std::to_string(np * np * np) + ")";
                if (got[3]) return "device " + std::to_string(ids[m]) + "'s add yields the armed bits in " + std::to_string(got[3]) + " of " + std::to_string(np * np * np) + " sums";
            }
        } catch (const std::exception& e) {
            return std::string("an add check failed: ") + e.what();
        }
        return {};
    }

private:
    // One transport's staging: per member its signal block and slots, the kernel's tables over them, and the parity halves the transport's next one-shot and next pulls take.
    struct Areas {
        GroupTransport transport;
        std::vector<void*> area;   // per member: its signal block, then a slot per member, twice on a fast transport; pinned host memory under host writes
        std::vector<void*> at;     // per member, the device's address of its area
        std::vector<size_t> placed;   // per member under host writes, the bytes of an area placed on its device's node, 0 where host_malloc took it
        std::vector<BufferPtr> peers;   // per member its GroupPeers in its device memory, then under host writes those naming pull
        std::vector<void*> table, pull_table;   // per member, the device address of its GroupPeers, and under host writes of those naming pull
        std::vector<BufferPtr> pull;   // per member under host writes, the device memory pulls copy into, two parity halves of a slot per member
        // Under host writes, per member: the copies' host area on its device's node (two halves of a share slot per member and its totals), its device address and bytes placed, and its GroupPeers.
        std::vector<void*> copy, copy_at;
        std::vector<size_t> copy_placed;
        std::vector<BufferPtr> copy_peers;
        std::vector<void*> copy_table;
        unsigned half = 0, pull_half = 0, copy_half = 0;
        bool host() const { return transport == GroupTransport::host_writes; }
        bool fast() const { return transport != GroupTransport::pulls; }
    };

    // A member's area: its signal block, then two parity halves of a slot per member.
    size_t area_bytes() const { return size_add(kGroupSignalBytes, size_mul(size_mul(2 * members_.size(), cap_), sizeof(float))); }

    // A member's copies area: two parity halves of a share slot per member and of its totals.
    size_t copy_slot() const { return (cap_ / 4 + members_.size() - 1) / members_.size() * 4; }
    size_t copy_bytes() const { return page_round(size_mul(size_mul(2 * (members_.size() + 1), copy_slot()), sizeof(float))); }
    static size_t page_round(size_t bytes) { return size_add(bytes, 4095) / 4096 * 4096; }

    // Pinned host memory for member m on its device's NUMA node where registered memory is coherent, else host_malloc's, told once on stderr; `placed` gets the bytes placed, 0 where not.
    void* host_area(size_t m, size_t bytes, size_t& placed) {
        placed = 0;
        void* p = nullptr;
        if (!Api::registered_coherent) {
            Api::check(Api::host_malloc(&p, bytes, true), "host_malloc");
            std::memset(p, 0, bytes);
            return p;
        }
        std::string why;
        size_t rounded = bytes;
        p = placed_host<Api>(rounded, members_[m]->dev_.pci_bus, why);
        placed = p ? rounded : 0;
        if (p) return p;
        static std::atomic<bool> told{false};
        if (!told.exchange(true))
            std::fprintf(stderr, "%sa tensor group's host memory is not placed on device %d's NUMA node, since %s\n", Api::prefix.c_str(), members_[m]->dev_.index, why.c_str());
        Api::check(Api::host_malloc(&p, bytes, true), "host_malloc");
        std::memset(p, 0, bytes);
        return p;
    }

    void stage_for(GroupTransport t) {
        const size_t W = members_.size();
        sets_.emplace_back();
        Areas& s = sets_.back();
        s.transport = t;
        s.area.assign(W, nullptr);
        s.at.assign(W, nullptr);
        s.placed.assign(W, 0);
        const size_t bytes = page_round(area_bytes());
        if (s.host() && copies_ && !words_) {
            Api::check(Api::host_malloc((void**)&words_, W * kCopyWords * sizeof(uint64_t), true), "host_malloc");
            std::memset(words_, 0, W * kCopyWords * sizeof(uint64_t));
            Api::check(Api::host_device_pointer((void**)&words_at_, words_), "host_device_pointer");
        }
        for (size_t m = 0; m < W; ++m) {
            GpuBackend<Api>& b = *members_[m];
            Api::check(Api::set_device(b.dev_.index), "set_device");
            // Peers' stores reach uncached memory without the fine-grain setting, which ordinary allocations do not (docs/ROCM.md, Collective design and proof); coherent host memory every member maps needs no peer access.
            // Host slots go on the member's NUMA node where registered memory is as coherent as the armed slots need, which HIP's registration does not promise.
            if (s.host()) {
                s.area[m] = host_area(m, bytes, s.placed[m]);
                Api::check(Api::host_device_pointer(&s.at[m], s.area[m]), "host_device_pointer");
            } else {
                Api::check(s.fast() ? Api::malloc_uncached(&s.area[m], bytes) : Api::malloc(&s.area[m], bytes), "malloc");
                s.at[m] = s.area[m];
            }
            if (s.host()) std::memset(s.area[m], 0, kGroupSignalBytes);
            else b.hold_fill(s.area[m], kGroupSignalBytes);
            if (s.fast()) b.group_arm(stage(s, m), size_mul(2 * W, cap_));
            // Host writes also time pulls and copies: pulls copy into device memory of their own, so the armed slots stay armed, and copies into a host area of their own on the member's node.
            if (s.host()) s.pull.push_back(b.alloc(size_mul(size_mul(2 * W, cap_), sizeof(float)), Memory::device));
            if (s.host() && copies_) {
                s.copy_placed.push_back(0);
                s.copy.push_back(host_area(m, copy_bytes(), s.copy_placed.back()));
                s.copy_at.push_back(nullptr);
                Api::check(Api::host_device_pointer(&s.copy_at.back(), s.copy.back()), "host_device_pointer");
                if (!copy_stream_[m]) {
                    Api::check(Api::stream_create(&copy_stream_[m]), "stream_create");
                    Api::check(Api::event_create(&added_[m], false), "event_create");
                    Api::check(Api::event_create(&sent_[m], false), "event_create");
                }
            }
        }
        for (const int table : {0, 1, 2}) {
            if ((table && !s.host()) || (table == 2 && !copies_)) break;
            for (size_t m = 0; m < W; ++m) {
                GroupPeers p{};
                for (size_t k = 0; k < W; ++k) {
                    p.stage[k] = table == 2 ? (float*)s.copy_at[k] : table ? pull_stage(s, k) : stage(s, k);
                    p.signal[k] = table == 2 ? (unsigned*)(words_at_ + k * kCopyWords) : (unsigned*)s.at[k];
                }
                p.status = status_;
                p.fault = members_[m]->fault_;
                p.cancel = status_ + W;
                std::vector<BufferPtr>& held = table == 2 ? s.copy_peers : s.peers;
                held.push_back(members_[m]->alloc(sizeof(GroupPeers), Memory::device));
                members_[m]->write(*held.back(), 0, &p, sizeof(GroupPeers));
                (table == 2 ? s.copy_table : table ? s.pull_table : s.table).push_back(as_gpu<Api>(*held.back()).data());
            }
        }
    }

    // The staged set of transport t; a pick of a transport this collective does not stage is a caller's error.
    Areas& areas(GroupTransport t) {
        for (Areas& s : sets_)
            if (s.transport == t) return s;
        throw std::logic_error(std::string("backend: a sum by ") + transport_name(t) + ", which its collective does not stage");
    }

    // fold: a generated row's sum in place whose take each member holds as its pending join, which the mix after its inject may take into one launch (GpuBackend::entry).
    // from: the float of the partial rows the sum starts at; lane: a lane's sum beside the members' streams (send), -1 for one on them.
    void summed(const std::vector<Slice>& residual, size_t n, bool replace, bool fold = false, size_t from = 0, int lane = -1) {
        const size_t W = members_.size();
        for (size_t m = 0; m < W; ++m) span(*residual[m].buffer, residual[m].offset * sizeof(float), size_mul(n, sizeof(float)));
        settle();
        // Every member's earlier failure throws before any member launches, so none waits for a peer that never comes.
        for (GpuBackend<Api>* b : members_) b->raise_failed();
        const GpuSumPick p = !fast_ ? GpuSumPick{sets_[0].transport, GpuSumForm::pulls} : lane >= 0 ? lane_pick(n) : form_of(n);
        // A one-shot's rows over peer writes that a member's held row launch still holds go out from that launch, which leaves the member's sum its take (GpuBackend::push_rows).
        std::vector<char> sent(W, 0);
        if (n && p.form == GpuSumForm::one_shot && tiles_.size() == W && !areas(p.transport).host() && !(fold && replace && W == 2)) {
            const Areas& s = areas(p.transport);
            for (size_t m = 0; m < W; ++m) {
                const typename GpuBackend<Api>::GroupTail t{s.table[m], tiles_[m], (unsigned long long)s.half * W * cap_, cap_, (int)W, (int)m, kPushTile};
                sent[m] = members_[m]->push_rows(GpuBackend<Api>::f32(CSlice(partial(m))), n, t);
            }
        }
        // Every member's held work goes to its stream before the sum and the sum's launches go together, so no member's sum waits on a peer whose launch the host still holds, and the pulls' events follow what they cover.
        for (GpuBackend<Api>* b : members_) b->flush();
        // A sum whose launch throws is still the last sum, so a wait for it covers the members' kernels that did launch.
        // A lane's sum beside is the last sum only at its finish, which holds the members' streams for it.
        const auto issued = [&] {
            if (lane >= 0) return;
            before_ = last_;
            for (size_t m = 0; m < W; ++m) last_[m] = members_[m]->tickets_.last() + 1;
        };
        try {
            std::vector<typename GpuBackend<Api>::TakeDesc> takes;
            if (n) {
                if (p.form == GpuSumForm::copies) copied(residual, n, replace, areas(p.transport), p, from);
                else if (p.form != GpuSumForm::pulls) pushed(residual, n, replace, areas(p.transport), p, fold && replace && members_.size() == 2 ? &takes : nullptr, sent, from, lane);
                else pulled(residual, n, replace, fast_ ? areas(p.transport) : sets_[0], from);
            }
            for (GpuBackend<Api>* b : members_) b->flush();
            for (size_t m = 0; m < takes.size(); ++m) members_[m]->pend_take(takes[m]);
        } catch (...) {
            drop();
            cancel();
            issued();
            throw;
        }
        issued();
    }

    // The pick a fast sum of n floats takes: the forced one, else the winner at the largest timed size not above n, below them all the least size's one-shot, at its floats a block where capped.
    GpuSumPick form_of(size_t n) const {
        if (force_) return *force_;
        auto it = forms_.upper_bound(n);
        if (it == forms_.begin()) {
            if (forms_.empty()) return {sets_[0].transport, GpuSumForm::one_shot};
            const GpuSumPick& least = it->second;
            if (least.form != GpuSumForm::one_shot || !least.blocks) return {least.transport, GpuSumForm::one_shot};
            return {least.transport, GpuSumForm::one_shot, (unsigned)std::max<size_t>(1, launched_blocks(least, it->first) * n / it->first)};
        }
        return (--it)->second;
    }

    // A lane's pick: kLaneForm in the transport n's pick takes where that pick is a kernel form, else n's pick; a forced pick stays.
    GpuSumPick lane_pick(size_t n) const {
        const GpuSumPick p = form_of(n);
        return force_ || !kernel_form(p.form) ? p : GpuSumPick{p.transport, kLaneForm};
    }

    // The blocks each member's kernel of a sum of n floats by `pick` takes: the form's own count by n, at most the pick's cap where it has one.
    size_t launched_blocks(GpuSumPick pick, size_t n) const {
        const size_t units = (n + 3) / 4, W = members_.size();
        // A copies pick's count is its chunks a share, by default one a kCopyChunkFloats.
        if (pick.form == GpuSumForm::copies)
            return pick.blocks ? std::min<size_t>(kCopyChunks, pick.blocks) : std::min(kCopyChunks, std::max<size_t>(1, ((units + W - 1) / W * 4 + kCopyChunkFloats - 1) / kCopyChunkFloats));
        size_t blocks = std::min<size_t>(one_shot_blocks_, (units + kThreads - 1) / kThreads);
        if (pick.form == GpuSumForm::chain || pick.form == GpuSumForm::wrap) blocks = std::min<size_t>(kChainBlocks, (units + kChainUnits - 1) / kChainUnits);
        else if (pick.form == GpuSumForm::shares) blocks = std::min<size_t>(one_shot_blocks_, ((units + W - 1) / W + kThreads - 1) / kThreads);
        return pick.blocks ? std::min<size_t>(blocks, pick.blocks) : blocks;
    }

    // The blocks of every member's launch follow from n and the form alone, so every member's block b meets its peers' block b.
    // takes, where given, gets the one-shot's take and the member launches its push alone; sent, its rows already pushed by their product, launches the take alone; beside is the stream to launch on, its blocks capped at kBesideBlocks.
    void launch(size_t m, const Slice& out, size_t n, bool replace, bool fast, GpuSumPick pick, const Areas& s, unsigned half, std::vector<typename GpuBackend<Api>::TakeDesc>* takes = nullptr, bool sent = false,
                size_t from = 0, typename Api::Stream beside = nullptr) {
        GpuBackend<Api>& b = *members_[m];
        const float* own = GpuBackend<Api>::f32(CSlice(partial(m))) + from;
        float* o = GpuBackend<Api>::f32(out);
        unsigned long long nn = n, cap = cap_, timeout = timeout_[m], tick = tick_[m];
        const size_t W = members_.size();
        if (!fast) pick = {pick.transport, GpuSumForm::one_shot};
        const GpuSumForm form = pick.form;
        const bool chained = form == GpuSumForm::chain || form == GpuSumForm::wrap;
        const size_t own_blocks = launched_blocks(pick, n), blocks = (beside ? std::min(own_blocks, chained ? kBesideBlocks / 2 : kBesideBlocks) : own_blocks) * (chained ? 2 : 1);
        int members = (int)W, rank = (int)m, flags = (replace ? 1 : 0) | (fast ? 2 : 0) | (int)half << 2 | (form == GpuSumForm::wrap ? 8 : 0);
        const Args args(fast || !s.host() ? s.table[m] : s.pull_table[m], own, o, nn, cap, members, rank, flags, timeout, tick);
        if (takes && form == GpuSumForm::one_shot) {
            b.group_push((unsigned)blocks, args);
            takes->push_back({s.table[m], own, o, nn, cap, (unsigned long long)half * W * cap_, timeout, tick, members, rank, flags, (unsigned)blocks, &status_[m]});
            return;
        }
        b.group_launch(form, (unsigned)blocks, args, sent && fast && form == GpuSumForm::one_shot, beside);
    }

    // Every fast sum takes the half its transport's next one-shot takes, which every peer finished reading at that transport's one-shot before the last (docs/src/backends-gpu.md, the collective).
    // A lane's sum beside waits for its member's work recorded before it, which summed flushed, and records the lane's event behind it.
    void pushed(const std::vector<Slice>& residual, size_t n, bool replace, Areas& s, GpuSumPick pick, std::vector<typename GpuBackend<Api>::TakeDesc>* takes, const std::vector<char>& sent, size_t from = 0, int lane = -1) {
        for (size_t m = 0; m < members_.size(); ++m) {
            if (lane < 0) {
                launch(m, residual[m], n, replace, true, pick, s, s.half, pick.form == GpuSumForm::one_shot ? takes : nullptr, sent[m], from);
                continue;
            }
            GpuBackend<Api>& b = *members_[m];
            Api::check(Api::set_device(b.dev_.index), "set_device");
            Api::check(Api::event_record(computed_[m], b.stream_), "event_record");
            Api::check(Api::stream_wait_event(beside_[m], computed_[m]), "stream_wait_event");
            launch(m, residual[m], n, replace, true, pick, s, s.half, nullptr, false, from, beside_[m]);
            Api::check(Api::event_record(summed_[(size_t)lane][m], beside_[m]), "event_record");
        }
        if (pick.form == GpuSumForm::one_shot || pick.form == GpuSumForm::shares) s.half ^= 1u;
    }

    // Copies (docs/src/backends-gpu.md, the collective): each member's copy stream sends chunk j of every other share of its rows into the owners' host slots and counts it, its first kernel adds its share as the rows land.
    // The stream then sends its total chunk by chunk and its second kernel takes the other totals as they land; no kernel waits for its own copy stream, so a stream that shares the kernel's hardware queue cannot hang it.
    void copied(const std::vector<Slice>& residual, size_t n, bool replace, Areas& s, GpuSumPick pick, size_t from = 0) {
        if (!copies_ || s.copy.empty()) throw std::logic_error(Api::prefix + "a copies sum on a collective that does not stage them");
        const size_t W = members_.size(), units = (n + 3) / 4, share = (units + W - 1) / W, J = launched_blocks(pick, n), per = (share + J - 1) / J, slot = copy_slot();
        const unsigned h = s.copy_half;
        s.copy_half ^= 1u;
        const uint64_t base = copy_base_;
        copy_base_ += J;
        const auto lo = [&](size_t k) { return std::min(share * k, units); };
        const unsigned blocks = (unsigned)std::min(one_shot_blocks_, std::max<size_t>(1, (share + kThreads - 1) / kThreads));
        const auto own = [&](size_t m) { return GpuBackend<Api>::f32(CSlice(partial(m))) + from; };
        const auto total = [&](size_t m) { return GpuBackend<Api>::f32(Slice{s.pull.at(m).get(), 0}); };
        const auto launch_copies = [&](size_t m, bool takes) {
            GpuBackend<Api>& b = *members_[m];
            unsigned long long nn = n, cap = slot, timeout = timeout_[m], tick = tick_[m], at = base;
            int members = (int)W, rank = (int)m, flags = (replace ? 1 : 0) | 2 | (int)h << 2 | (takes ? 8 : 0);
            unsigned chunks = (unsigned)J;
            b.group_launch(GpuSumForm::copies, blocks, Args(s.copy_table[m], own(m), GpuBackend<Api>::f32(residual[m]), nn, cap, members, rank, flags, timeout, tick, total(m), at, chunks));
            b.flush();
        };
        // Every member's rows are issued before any kernel, every total before any second kernel, and each copy before the kernel after it on its member's stream.
        // So neither a copy stream that shares its kernel's hardware queue nor a runtime that holds the host on a copy into host memory until its stream reaches it can hang the sum.
        for (size_t m = 0; m < W; ++m) {
            const typename Api::Stream c = copy_stream_[m];
            Api::check(Api::set_device(members_[m]->dev_.index), "set_device");
            Api::check(Api::event_record(ready_[m], members_[m]->stream_), "event_record");
            Api::check(Api::stream_wait_event(c, ready_[m]), "stream_wait_event");
            for (size_t j = 0; j < J; ++j) {
                for (size_t r = 1; r < W; ++r) {
                    const size_t k = (m + r) % W, a = std::min(lo(k) + j * per, lo(k + 1)), e = std::min(a + per, lo(k + 1));
                    if (a < e)
                        Api::check(Api::copy_async((float*)s.copy[k] + (h * (W + 1) + m) * slot + (a - lo(k)) * 4, own(m) + a * 4, (std::min(e * 4, n) - a * 4) * sizeof(float), Api::kToHost, c), "copy_async");
                }
                Api::check(Api::write_word(c, words_at_ + m * kCopyWords, base + j + 1), "write_word");
            }
        }
        for (size_t m = 0; m < W; ++m) {
            const typename Api::Stream c = copy_stream_[m];
            launch_copies(m, false);
            Api::check(Api::set_device(members_[m]->dev_.index), "set_device");
            Api::check(Api::event_record(added_[m], members_[m]->stream_), "event_record");
            Api::check(Api::stream_wait_event(c, added_[m]), "stream_wait_event");
            for (size_t j = 0; j < J; ++j) {
                const size_t a = std::min(lo(m) + j * per, lo(m + 1)), e = std::min(a + per, lo(m + 1));
                if (a < e)
                    Api::check(Api::copy_async((float*)s.copy[m] + (h * (W + 1) + W) * slot + (a - lo(m)) * 4, total(m) + (a - lo(m)) * 4, (std::min(e * 4, n) - a * 4) * sizeof(float), Api::kToHost, c), "copy_async");
                Api::check(Api::write_word(c, words_at_ + m * kCopyWords + 1, base + j + 1), "write_word");
            }
            Api::check(Api::event_record(sent_[m], c), "event_record");
        }
        // The member's next work, which may write its rows or total again, follows the copies that read them.
        for (size_t m = 0; m < W; ++m) {
            launch_copies(m, true);
            Api::check(Api::set_device(members_[m]->dev_.index), "set_device");
            Api::check(Api::stream_wait_event(members_[m]->stream_, sent_[m]), "stream_wait_event");
        }
    }

    // Each member copies its rows into every peer's pull half on its own stream and records that they landed, and each sums from that half once every peer's rows have.
    // A copy into a half follows the copying member's last sum, which waited for the receiver's copies of that sum, which follow the receiver's sum before, the last to read the half.
    void pulled(const std::vector<Slice>& residual, size_t n, bool replace, Areas& s, size_t from = 0) {
        const size_t W = members_.size(), bytes = n * sizeof(float);
        const unsigned h = s.pull_half;
        s.pull_half ^= 1u;
        for (size_t m = 0; m < W; ++m) {
            GpuBackend<Api>& b = *members_[m];
            for (size_t k = 0; k < W; ++k)
                if (k != m) b.launch_check(Api::copy_async(pull_stage(s, k) + (h * W + m) * cap_, GpuBackend<Api>::f32(CSlice(partial(m))) + from, bytes, Api::kAnyCopy, b.stream_), "copy_async");
            Api::check(Api::event_record(ready_[m], b.stream_), "event_record");
        }
        for (size_t m = 0; m < W; ++m) {
            for (size_t k = 0; k < W; ++k)
                if (k != m) Api::check(Api::stream_wait_event(members_[m]->stream_, ready_[k]), "stream_wait_event");
            launch(m, residual[m], n, replace, false, GpuSumPick{s.transport, GpuSumForm::one_shot}, s, h, nullptr, false, from);
        }
    }

    // A sum's kernel that gave up set its member's status word; the collective takes the conservative path from then on, decided here once.
    void settle() noexcept {
        if (!fast_) return;
        for (size_t m = 0; m < members_.size(); ++m)
            if (host_load(&status_[m])) {
                fast_ = false;
                std::fprintf(stderr, "%sa tensor group's sum gave up waiting on %s; the group's later sums take the conservative path\n", Api::prefix.c_str(), members_[m]->ordinal().c_str());
                return;
            }
    }

    // A sum some member's launch of which failed, or whose device reported an error: the members whose kernels did launch stop waiting for it, and the collective turns conservative now.
    void cancel() noexcept {
        if (!fast_) return;
        host_store(&status_[members_.size()], 1u);
        fast_ = false;
        std::fprintf(stderr, "%sa tensor group's sum was cancelled since a member's launch or device failed; the group's later sums take the conservative path\n", Api::prefix.c_str());
    }

    // A lane's sum shares the staging and signals with every other sum, so none is made in one call between its send and finish.
    void one_call() const {
        if (lane_sent_[0] || lane_sent_[1]) throw std::logic_error("backend: a sum made in one call while a lane's sum is sent and not finished");
    }
    // The forms a member's kernel sends alone, which a lane's sum may take beside its member's stream; copies and pulls keep their copies on the member's streams.
    static bool kernel_form(GpuSumForm f) { return f == GpuSumForm::one_shot || f == GpuSumForm::chain || f == GpuSumForm::wrap || f == GpuSumForm::shares; }

    // A member's staging slots, past its signal block, and the slots pulls copy into: under host writes device memory of their own, else the same.
    static float* stage(const Areas& s, size_t m) { return (float*)((uint8_t*)s.at[m] + kGroupSignalBytes); }
    static float* pull_stage(const Areas& s, size_t m) { return s.host() ? GpuBackend<Api>::f32(Slice{s.pull.at(m).get(), 0}) : stage(s, m); }

    // The members are drained by the caller, so no kernel or copy still names what goes.
    void free_areas(Areas& s) noexcept {
        for (size_t m = 0; m < s.area.size(); ++m) {
            (void)Api::set_device(members_[m]->dev_.index);
            if (s.area[m]) free_host_or_device(s.area[m], s.host(), s.placed.empty() ? 0 : s.placed[m]);
            s.area[m] = nullptr;
            if (m < s.copy.size() && s.copy[m]) free_host_or_device(s.copy[m], true, s.copy_placed[m]);
        }
        s.peers.clear();
        s.pull.clear();
        s.copy.clear();
        s.copy_at.clear();
        s.copy_placed.clear();
        s.copy_peers.clear();
        s.copy_table.clear();
    }

    static void free_host_or_device(void* p, bool host, size_t placed) noexcept {
        if (placed) {
            unregister_host<Api>(p);
#if defined(__linux__)
            munmap(p, placed);
#endif
        } else {
            free_memory<Api>(p, host);
        }
    }

    // Every member and copy stream drained first, so no kernel or copy still names what goes.
    void release() noexcept {
        for (GpuBackend<Api>* m : members_) m->sync();
        for (size_t m = 0; m < copy_stream_.size(); ++m)
            if (copy_stream_[m]) {
                (void)Api::set_device(members_[m]->dev_.index);
                sync_stream<Api>(copy_stream_[m]);
                (void)Api::stream_destroy(copy_stream_[m]);
                (void)Api::event_destroy(added_[m]);
                (void)Api::event_destroy(sent_[m]);
                copy_stream_[m] = nullptr;
            }
        for (size_t m = 0; m < beside_.size(); ++m) {
            (void)Api::set_device(members_[m]->dev_.index);
            if (beside_[m]) sync_stream<Api>(beside_[m]), (void)Api::stream_destroy(beside_[m]);
            if (computed_[m]) (void)Api::event_destroy(computed_[m]);
            for (auto& e : summed_)
                if (e[m]) (void)Api::event_destroy(e[m]);
        }
        beside_.clear();
        for (Areas& s : sets_) free_areas(s);
        sets_.clear();
        for (size_t m = 0; m < members_.size(); ++m) {
            (void)Api::set_device(members_[m]->dev_.index);
            if (ready_[m]) (void)Api::event_destroy(ready_[m]);
            ready_[m] = nullptr;
        }
        if (status_) free_memory<Api>(status_, true);
        status_ = nullptr;
        if (words_) free_memory<Api>(words_, true);
        words_ = nullptr;
    }

    std::vector<GpuBackend<Api>*> members_;
    size_t one_shot_blocks_ = SIZE_MAX;   // the one-shot's most blocks, set at construction
    size_t rows_, width_, cap_;   // cap_: a staging slot's floats, the reserved rows rounded up to 256 bytes so every slot starts on a run
    bool fast_;
    const bool copies_;   // copies are staged and timed under host writes
    std::optional<GpuSumPick> force_;   // the pick every fast sum takes, else by size from forms_
    std::map<size_t, GpuSumPick> forms_;
    std::vector<Areas> sets_;   // per staged transport, in the order given; the first is the conservative path's
    std::vector<BufferPtr> tile_bufs_;   // per member, the pushing held build's tile counters, or none where its held builds do not push
    std::vector<unsigned*> tiles_;
    std::vector<BufferPtr> partial_;
    std::vector<typename Api::Event> ready_;   // per member, the event behind its pulls' copies
    std::vector<unsigned long long> timeout_, tick_;   // per member in its wall-clock ticks: a probe's kProbeTimeoutMs, else kSumTimeoutMs, and a millisecond
    unsigned* status_ = nullptr;   // host memory, a word a member, set by a kernel that gave up, then the host's cancel word
    std::vector<Ticket> last_, before_;   // per member, the ticket holding the last sum and the sum before it
    std::vector<typename Api::Stream> copy_stream_;   // per member under host writes, the stream its copies' copies go on
    std::vector<typename Api::Event> added_, sent_;   // per member, the events after its copies' first kernel and after its total's copies
    uint64_t* words_ = nullptr;   // coherent host memory, kCopyWords a member for copies, and its device address
    uint64_t* words_at_ = nullptr;
    uint64_t copy_base_ = 0;   // the count every member's copies words reached at the last copies sum
    std::vector<typename Api::Stream> beside_;   // per member on the fast path of two or more, the stream lanes' sums cross on (send)
    std::vector<typename Api::Event> computed_, summed_[2];   // per member, its work before a lane's sum, and per lane the sum's end on beside_
    bool lane_sent_[2] = {false, false}, lane_beside_[2] = {false, false};   // per lane, whether its sum is sent and not finished, and whether it went beside
};

// Every directed pair of `ids` reaches the other's memory, its access enabled here where it was not, else why not; `enabled`, where given, gets each pair this call enabled.
template <class Api>
std::string group_access(const std::vector<int>& ids, std::vector<std::pair<int, int>>* enabled) {
    for (int a : ids)
        for (int b : ids) {
            if (a == b) continue;
            int can = 0;
            if (Api::can_access_peer(&can, a, b) != Api::kSuccess || !can)
                return "device " + std::to_string(a) + " cannot reach device " + std::to_string(b) + "'s memory";
            Api::check(Api::set_device(a), "set_device");
            const typename Api::Error e = Api::enable_peer_access(b);
            if (e != Api::kSuccess && e != Api::kPeerAlreadyEnabled)
                return "peer access from device " + std::to_string(a) + " to device " + std::to_string(b) + " failed with " + Api::error_name(e);
            if (e == Api::kSuccess && enabled) enabled->emplace_back(a, b);
        }
    return {};
}

// Empty where every directed pair reaches the other's memory under peer_writes and a fast sum by `transport` gives the member-order sum; else why not (group_transport checks the adds once before).
// `enabled`, where given, gets each directed pair whose access this probe enabled, so a caller that does not keep the group can disable it; `force` probes that pick alone, with copies staged.
template <class Api>
std::string group_probe(const std::vector<GpuBackend<Api>*>& group, const std::vector<int>& ids, GroupTransport transport, std::vector<std::pair<int, int>>* enabled = nullptr,
                        std::optional<GpuSumPick> force = std::nullopt) {
    if (transport == GroupTransport::peer_writes) {
        const std::string why = group_access<Api>(ids, enabled);
        if (!why.empty()) return why;
    }
    const std::string by = force ? " by copies" : "";
    try {
        const size_t n = 1027;   // a tail past whole units, over one block
        GpuCollective<Api> c(group, 1, n, {transport}, true, force, force.has_value());
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
        if (!c.fast()) return "a probe sum" + by + " gave up waiting for its peers";
        for (size_t m = 0; m < group.size(); ++m) {
            std::vector<float> got(n);
            group[m]->read(*c.partial(m).buffer, 0, got.data(), n * sizeof(float));
            if (std::memcmp(got.data(), want.data(), n * sizeof(float)))
                return "a probe sum" + by + " on device " + std::to_string(ids[m]) + " read other values than its peers wrote";
        }
    } catch (const std::exception& e) {
        return "a probe sum" + by + " failed: " + e.what();
    }
    return {};
}

// Every fast transport whose probe passes, peer writes then host writes, none where every member's add does not hide the one-shot's sanitized values, with why each dropped one failed.
template <class Api>
GroupPath group_transport(const std::vector<GpuBackend<Api>*>& group, const std::vector<int>& ids, std::vector<std::pair<int, int>>* enabled = nullptr) {
    GroupPath path;
    path.why = GpuCollective<Api>::add_check(group, ids);
    if (!path.why.empty()) return path;
    for (const GroupTransport t : {GroupTransport::peer_writes, GroupTransport::host_writes}) {
        const std::string why = group_probe(group, ids, t, enabled);
        if (why.empty()) path.transports.push_back(t);
        else path.why += (path.why.empty() ? "" : "; ") + why;
    }
    // Host writes keep their kernel forms where copies fail their own probe, which sums once by copies in two chunks, as their counts and copy streams are a path of their own.
    if (Api::copy_sums && !path.transports.empty() && path.transports.back() == GroupTransport::host_writes) {
        const std::string why = group_probe(group, ids, GroupTransport::host_writes, enabled, GpuSumPick{GroupTransport::host_writes, GpuSumForm::copies, 2});
        path.copies = why.empty();
        if (!why.empty()) std::fprintf(stderr, "%sa tensor group's host writes sum without copies, since %s\n", Api::prefix.c_str(), why.c_str());
    }
    return path;
}

// The transports a group's picks name, or its first one's where nothing was timed, as transport_bit flags.
unsigned picked_transports(const GroupPath& path) {
    unsigned by = path.forms.empty() && !path.transports.empty() ? transport_bit(path.transports[0]) : 0;
    for (const auto& [n, p] : path.forms) by |= transport_bit(p.transport);
    return by;
}

// What a collective of `path` stages: every kept transport where sizes past those timed may still pick it, else only what the picks name, and pulls where none is kept.
std::vector<GroupTransport> staged_transports(const GroupPath& path, size_t n) {
    if (path.transports.empty()) return {GroupTransport::pulls};
    if (n > path.timed && n >= kTimedFloor) return path.transports;
    const unsigned by = picked_transports(path);
    std::vector<GroupTransport> kept;
    for (GroupTransport t : path.transports)
        if (by & transport_bit(t)) kept.push_back(t);
    return kept;
}

// The fast transports `forms` names, peer writes first, or peer writes where it names none.
inline std::vector<GroupTransport> named_transports(const std::map<size_t, GpuSumPick>& forms) {
    std::vector<GroupTransport> named;
    for (GroupTransport t : {GroupTransport::peer_writes, GroupTransport::host_writes})
        for (const auto& [n, p] : forms)
            if (p.transport == t) {
                named.push_back(t);
                break;
            }
    if (named.empty()) named.push_back(GroupTransport::peer_writes);
    return named;
}

std::string device_list(const std::vector<int>& ids) {
    std::string names;
    for (int id : ids) names += (names.empty() ? "" : ", ") + std::to_string(id);
    return names;
}

template <class Api>
std::vector<GpuBackend<Api>*> GpuBackend<Api>::group_of(const std::vector<Backend*>& members, std::vector<int>& ids) {
    if (members.empty() || members[0] != this) throw std::logic_error("backend: a collective is joined by its first member");
    if (members.size() > kGroupMembers) throw std::runtime_error(Api::prefix + "a collective of more than " + std::to_string(kGroupMembers) + " devices");
    std::vector<GpuBackend<Api>*> group;
    ids.clear();
    for (Backend* b : members) {
        auto* r = dynamic_cast<GpuBackend<Api>*>(b);
        if (!r) throw std::runtime_error(Api::prefix + "a collective over a backend of another kind");
        if (r->dev_.caps.device != dev_.caps.device) throw std::runtime_error(Api::prefix + "a collective over devices of different kinds");
        for (int id : ids)
            if (id == r->dev_.index) throw std::runtime_error(Api::prefix + "a collective over one device twice");
        group.push_back(r);
        ids.push_back(r->dev_.index);
    }
    return group;
}

template <class Api>
GroupPath& GpuBackend<Api>::group_path(const std::vector<GpuBackend<Api>*>& group, const std::vector<int>& ids) {
    auto path = group_paths_.find(ids);
    if (path != group_paths_.end()) return path->second;
    path = group_paths_.emplace(ids, group_transport(group, ids)).first;
    const GroupPath& p = path->second;
    std::string by;
    for (GroupTransport t : p.transports) by += std::string(by.empty() ? "" : " or ") + transport_name(t);
    if (by.empty()) by = transport_name(GroupTransport::pulls);
    std::fprintf(stderr, "%stensor group of devices %s sums by %s%s%s\n", Api::prefix.c_str(), device_list(ids).c_str(), by.c_str(), p.why.empty() ? "" : ", since ", p.why.c_str());
    return path->second;
}

template <class Api>
void GpuBackend<Api>::timed_group(const std::vector<GpuBackend<Api>*>& group, const std::vector<int>& ids, const GroupPath& path, const std::string& times) {
    const unsigned by = path.transports.empty() ? transport_bit(GroupTransport::pulls) : picked_transports(path);
    for (GpuBackend<Api>* b : group) b->summed_by_ |= by;
    if (times.empty()) return;
    std::string forms;
    for (GpuSumForm f : kTimedForms)
        if (f != GpuSumForm::pulls && f != GpuSumForm::copies) forms += std::string(sum_form_name(f)) + ", ";
    std::fprintf(stderr, "%stensor group of devices %s timed its sums in place, ms by transport %sand %spulls under host writes: %s\n", Api::prefix.c_str(), device_list(ids).c_str(),
                 forms.c_str(), path.copies ? "copies and " : "", times.c_str());
}

// A group whose timing threw takes the conservative path for good, reported on stderr.
template <class Api>
void timing_failed(GroupPath& path, const std::vector<int>& ids, const std::exception& e) {
    path.transports.clear();
    path.forms.clear();
    path.why = std::string("timing its sums failed: ") + e.what();
    std::fprintf(stderr, "%stensor group of devices %s sums by peer pulls, since %s\n", Api::prefix.c_str(), device_list(ids).c_str(), path.why.c_str());
}

// The permutation of the listing whose groups of `width` sum rows x summed floats fastest in all, by one swap descent from the listing over every pair of devices (docs/TENSOR-SPLIT.md, section 4.1).
// A group's cost is its fastest form in listing order, a group this backend cannot sum fast is never formed, and of the swaps that beat what they replace by kOrderMargin the first within kOrderMargin of the best gain is taken.
template <class Api>
std::vector<size_t> GpuBackend<Api>::member_order(const std::vector<Backend*>& devices, size_t width, size_t rows, size_t summed) {
    if (devices.empty() || devices[0] != this) throw std::logic_error("backend: a member order is asked of the listing's first device");
    const size_t N = devices.size(), W = width, n = size_mul(rows, summed);
    std::vector<size_t> perm(N);
    for (size_t d = 0; d < N; ++d) perm[d] = d;
    if (W < 2 || N % W || n < kTimedFloor) return perm;
    const auto key = std::make_pair(std::vector<const Backend*>(devices.begin(), devices.end()), W);
    const auto cached = picks_.find(key);
    if (cached != picks_.end()) return cached->second;
    const auto started = std::chrono::steady_clock::now();
    std::map<std::vector<size_t>, double> costs;
    const auto cost = [&](const std::vector<size_t>& members) {
        auto at = costs.find(members);
        if (at != costs.end()) return at->second;
        double ms = 1e30;
        // A group of another kind, or whose probe or timing throws, is never formed, as a probe that fails falls conservative.
        // The access a probe enabled is disabled once the group is timed, as a CUDA device holds at most eight peers; group_path enables the picked groups' again.
        std::vector<GpuBackend<Api>*> group;
        std::vector<std::pair<int, int>> enabled;
        try {
            auto* lead = dynamic_cast<GpuBackend<Api>*>(devices[members[0]]);
            if (!lead) throw std::runtime_error("a group led by another kind");
            std::vector<Backend*> listed;
            for (size_t d : members) listed.push_back(devices[d]);
            std::vector<int> ids;
            group = lead->group_of(listed, ids);
            const GroupPath kept = group_transport(group, ids, &enabled);
            // The group's cost is its fastest pick over every kept transport, which its sums of this size will take.
            if (!kept.transports.empty()) {
                GpuCollective<Api> c(group, rows, summed, kept.transports, false, std::nullopt, kept.copies);
                for (const GpuSumPick& p : c.picks()) ms = std::min(ms, c.timed(p, n));
            }
        } catch (const std::exception&) {
            ms = 1e30;
        }
        for (auto* g : group) g->sync();
        for (const auto& [a, b] : enabled)
            if (Api::set_device(a) == Api::kSuccess) (void)Api::disable_peer_access(b);
        return costs[members] = ms;
    };
    // A group is its members in listing order, so its cost and its sums' bits follow its set alone.
    const auto group_at = [&](size_t g) {
        std::vector<size_t> set(perm.begin() + g * W, perm.begin() + (g + 1) * W);
        std::sort(set.begin(), set.end());
        return set;
    };
    std::vector<double> group_ms(N / W);
    for (size_t g = 0; g < group_ms.size(); ++g) group_ms[g] = cost(group_at(g));
    double listed_ms = 0;
    for (double ms : group_ms) listed_ms += ms;
    struct Swap {
        size_t i, j;
        double drop, ms_i, ms_j;
    };
    for (bool moved = true; moved;) {
        moved = false;
        double total = 0, best = 0;
        for (double ms : group_ms) total += ms;
        std::vector<Swap> better;
        for (size_t i = 0; i < N; ++i)
            for (size_t j = i + 1; j < N; ++j) {
                const size_t gi = i / W, gj = j / W;
                if (gi == gj) continue;
                std::swap(perm[i], perm[j]);
                const double ti = cost(group_at(gi)), tj = cost(group_at(gj));
                std::swap(perm[i], perm[j]);
                const double before = group_ms[gi] + group_ms[gj];
                if (ti + tj >= before * (1.0 - kOrderMargin)) continue;
                better.push_back({i, j, before - (ti + tj), ti, tj});
                best = std::max(best, before - (ti + tj));
            }
        for (const Swap& s : better) {
            if (s.drop < best - kOrderMargin * total) continue;
            std::swap(perm[s.i], perm[s.j]);
            group_ms[s.i / W] = s.ms_i;
            group_ms[s.j / W] = s.ms_j;
            moved = true;
            break;
        }
    }
    // Each group in listing order, and the groups by their first member, so the listing's first device leads the first group.
    std::vector<std::vector<size_t>> groups;
    for (size_t g = 0; g < N / W; ++g) groups.push_back(group_at(g));
    std::sort(groups.begin(), groups.end());
    for (size_t g = 0; g < groups.size(); ++g) std::copy(groups[g].begin(), groups[g].end(), perm.begin() + g * W);
    picks_[key] = perm;
    double picked_ms = 0;
    for (double ms : group_ms) picked_ms += ms;
    const auto id_of = [&](size_t d) {
        auto* r = dynamic_cast<GpuBackend<Api>*>(devices[d]);
        return r ? std::to_string(r->dev_.index) : std::string("?");
    };
    std::string listed, picked;
    for (size_t d = 0; d < N; ++d) {
        listed += (d == 0 ? "" : d % W ? "+" : ", ") + id_of(d);
        picked += (d == 0 ? "" : d % W ? "+" : ", ") + id_of(perm[d]);
    }
    const double took = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    std::fprintf(stderr, "%stensor groups %s take devices %s, %.3f ms of sums of %zu floats in all against %.3f as listed; %zu groups timed in %.0f ms\n",
                 Api::prefix.c_str(), listed.c_str(), picked.c_str(), picked_ms, n, listed_ms, costs.size(), took);
    // Each group's forms are timed in place into its path as ordered, so join times nothing twice, and every collective timed is gone before the fit reads free memory.
    for (size_t g = 0; g < N / W; ++g) {
        std::vector<Backend*> members;
        for (size_t k = g * W; k < (g + 1) * W; ++k) members.push_back(devices[perm[k]]);
        auto* lead = dynamic_cast<GpuBackend<Api>*>(members[0]);
        std::vector<int> ids;
        std::vector<GpuBackend<Api>*> group;
        try {
            if (lead) group = lead->group_of(members, ids);
        } catch (const std::exception&) {
            group.clear();
        }
        if (group.empty()) continue;
        GroupPath& path = lead->group_path(group, ids);
        if (path.transports.empty() || path.timed >= n) continue;
        std::string times;
        try {
            GpuCollective<Api> c(group, rows, summed, path.transports, false, std::nullopt, path.copies);
            times = c.time_forms(path.forms, path.timed);
            if (c.fast()) path.timed = n;
        } catch (const std::exception& e) {
            timing_failed<Api>(path, ids, e);
        }
        timed_group(group, ids, path, times);
    }
    return perm;
}

// A collective over devices of this backend and one kind, each once, decided fast or conservative once per group by group_probe and reported on stderr.
template <class Api>
std::unique_ptr<Collective> GpuBackend<Api>::join(const std::vector<Backend*>& members, size_t rows, size_t width) {
    std::vector<int> ids;
    const std::vector<GpuBackend<Api>*> group = group_of(members, ids);
    const std::string names = device_list(ids);
    GroupPath& g = group_path(group, ids);
    const size_t n = size_mul(rows, width);
    std::unique_ptr<GpuCollective<Api>> c = std::make_unique<GpuCollective<Api>>(group, rows, width, staged_transports(g, n), false, std::nullopt, g.copies);
    // The picks are timed on the group's own cards at its first collective past the sizes timed so far, its members idle, so the topology picks them (docs/TENSOR-SPLIT.md, section 4.3).
    if (!g.transports.empty() && group.size() > 1 && n > g.timed) {
        std::string times;
        try {
            times = c->time_forms(g.forms, g.timed);
        } catch (const std::exception& e) {
            timing_failed<Api>(g, ids, e);
            c.reset();
            timed_group(group, ids, g, {});
            return std::make_unique<GpuCollective<Api>>(group, rows, width, std::vector<GroupTransport>{GroupTransport::pulls});
        }
        g.timed = n;
        timed_group(group, ids, g, times);
    }
    // The collective that timed is the one returned, holding only the staging its picks use.
    c->use(g.forms);
    if (c->pinned_bytes())
        std::fprintf(stderr, "%stensor group of devices %s holds %.1f MiB of pinned host memory for sums of %zu floats\n", Api::prefix.c_str(), names.c_str(), c->pinned_bytes() / 1048576.0, n);
    return c;
}

// A crossing's host-visible buffer that readers of this backend copy from on their own streams, and that the sender writes again behind each reader's last copy (docs/src/backends-gpu.md, handoffs).
// The order is counters in coherent host memory that stream memory operations write and wait for, since a HIP event waited for on another device's stream holds the host until it completes.
template <class Api>
class GpuHandoff final : public Handoff {
public:
    GpuHandoff(GpuBackend<Api>& sender, std::vector<GpuBackend<Api>*> readers)
        : sender_(sender), readers_(std::move(readers)), counts_(readers_.size(), 0), waited_(readers_.size(), 0) {
        const size_t words = 1 + readers_.size();
        Api::check(Api::host_malloc((void**)&words_, words * sizeof(uint64_t), true), "host_malloc");
        std::memset(words_, 0, words * sizeof(uint64_t));
    }
    ~GpuHandoff() override { release(); }

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
        if (!serves(reader)) throw std::logic_error(Api::prefix + "a handoff taken by a reader it does not serve");
        GpuBackend<Api>& b = *readers_[reader];
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
        if (!serves(reader)) throw std::logic_error(Api::prefix + "a handoff taken by a reader it does not serve");
        // The count advances only once its write is issued, so a flush that throws drops a write no sender's begin waits for.
        const uint64_t c = counts_[reader] + 1;
        write(*readers_[reader], &words_[1 + reader], c);
        readers_[reader]->flush();
        counts_[reader] = c;
    }

private:
    // Held with the backend's other stream operations, so they keep its order.
    static void wait_for(GpuBackend<Api>& on, uint64_t* word, uint64_t value) { on.hold_wait(word, value); }
    static void write(GpuBackend<Api>& on, uint64_t* word, uint64_t value) { on.hold_write(word, value); }
    // The counters outlive every operation queued on them, each side's stream drained first.
    void release() noexcept {
        for (GpuBackend<Api>* b : readers_)
            if (b) b->sync();
        sender_.sync();
        if (words_) free_memory<Api>(words_, true);
    }
    GpuBackend<Api>& sender_;
    std::vector<GpuBackend<Api>*> readers_;   // per reader, null for one this handoff does not serve
    uint64_t* words_ = nullptr;   // coherent host memory: the sends so far, then each reader's takes so far
    std::vector<uint64_t> counts_;   // per reader, its takes so far, written on its thread before the sender's next begin reads it
    std::vector<uint64_t> waited_;   // per reader, the send its stream last waited for
    std::atomic<uint64_t> sends_{0};   // sent calls so far, read on the readers' threads
};

// Every reader of this backend is served, since it copies from the sender's host memory and needs no peer access.
template <class Api>
std::unique_ptr<Handoff> GpuBackend<Api>::handoff(const std::vector<Backend*>& readers) {
    std::vector<GpuBackend<Api>*> to;
    bool any = false;
    for (Backend* b : readers) {
        auto* r = dynamic_cast<GpuBackend<Api>*>(b);
        to.push_back(r);
        any = any || r;
    }
    if (!any) return nullptr;
    return std::make_unique<GpuHandoff<Api>>(*this, std::move(to));
}

} // namespace
} // namespace backend
