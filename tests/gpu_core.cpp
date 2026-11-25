// gpu-core: the shared GPU host core (gpu/backend_core.hpp) over a fake Api backed by host memory, with no device (docs/TESTS.md).
#include "backends/gpu/backend_core.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace backend {
namespace {

// A group sum's launch as the fake saw it: the kernel, its blocks, its stream and its packed arguments.
struct GroupLaunch {
    std::string name;
    unsigned blocks = 0;
    int* stream = nullptr;
    std::vector<uint8_t> args;
};

// What the fake has handed out and not yet taken back, each kind counted, and calls a test makes fail.
struct FakeState {
    std::atomic<long> memory{0}, host{0}, streams{0}, events{0}, modules{0}, launches{0};
    std::atomic<bool> refuse_malloc{false}, refuse_elapsed{false}, refuse_launch{false}, no_binary{false};
    std::atomic<uint64_t*> waited_word{nullptr};   // the word of the last wait_word, and the value it waited for
    std::atomic<uint64_t> waited_value{0};
    bool first_wait_closed = false;
    std::atomic<int*> dead{nullptr};   // a stream whose events report a device error, a member whose card is gone
    bool recording = false, watching = false;
    std::vector<GroupLaunch> groups;   // the group sums launched while recording
    std::vector<GroupLaunch> watched;  // every launch while watching
};
FakeState fake;

struct FakeDevice {
    int index = 0;
    std::string arch = "fake9";
    DeviceCaps caps;
    std::string pci_bus = "0000:00:00.0", pci_root;
    size_t lanes_resident = 1024;
    DeviceProfile profile;
};

struct FakeEvent {
    std::atomic<bool> recorded{false};
    std::atomic<int*> stream{nullptr};
};

struct FakeApi {
    using Error = int;
    using Stream = int*;
    using Event = FakeEvent*;
    using Module = int*;
    using Function = const char*;
    using CopyKind = int;
    using Device = FakeDevice;
    static constexpr Error kSuccess = 0, kNotReady = 1, kPeerAlreadyEnabled = 2, kNoBinary = 4, kOutOfMemory = 5;
    static bool no_binary(Error e) { return e == kNoBinary; }
    static constexpr int kRefused = 3;
    static constexpr CopyKind kToHost = 0, kToDevice = 0, kAnyCopy = 0;
    static constexpr const char* name = "fake";
    static constexpr bool int8_tensor_tiles = false;
    static constexpr bool copy_sums = false, registered_coherent = false;
    static constexpr bool shared_block_cap = false;
    static constexpr bool dot4_everywhere = false;
    static inline const std::string prefix = "fake: ";
    static constexpr const char* arch_option = "FAKE_ARCHS";
    static constexpr const unsigned char f32_image[1] = {1}, int16_image[1] = {2};
    static constexpr const unsigned char* f32_module = f32_image;
    static constexpr const unsigned char* int16_module = int16_image;
    static constexpr const unsigned char* int16_dot_module = nullptr;

    static Device device(int index) {
        FakeDevice d;
        d.index = index;
        d.pci_bus = "0000:0" + std::to_string(index) + ":00.0";
        d.caps.subgroup_size = 32;
        d.caps.compute_units = 4;
        d.caps.shared_bytes = 48 << 10;
        d.caps.device = "fake";
        d.caps.driver = "fake 1";
        return d;
    }
    static void check(Error e, const char* what) {
        if (e != kSuccess) throw std::runtime_error(prefix + what + " failed with " + error_name(e));
    }
    static const char* error_name(Error e) { return e == kRefused ? "REFUSED" : e == kNotReady ? "NOT_READY" : "ERROR"; }
    static Error set_device(int) { return kSuccess; }
    static Error malloc(void** p, size_t bytes) {
        if (fake.refuse_malloc) return kRefused;
        *p = std::calloc(1, bytes ? bytes : 1);
        ++fake.memory;
        return kSuccess;
    }
    static Error malloc_uncached(void** p, size_t bytes) { return malloc(p, bytes); }
    static unsigned* flush_word(const Device&) { return nullptr; }
    static Error free(void* p) {
        std::free(p);
        --fake.memory;
        return kSuccess;
    }
    static Error host_malloc(void** p, size_t bytes, bool coherent) {
        *p = std::calloc(1, bytes ? bytes : 1);
        if (coherent && bytes == sizeof(uint64_t)) std::memset(*p, 1, bytes);
        ++fake.host;
        return kSuccess;
    }
    static Error host_free(void* p) {
        std::free(p);
        --fake.host;
        return kSuccess;
    }
    static Error host_device_pointer(void** device, void* host) {
        *device = host;
        return kSuccess;
    }
    static Error host_register(void*, size_t) { return kSuccess; }
    static Error host_unregister(void*) { return kSuccess; }
    static Error mem_info(size_t* free, size_t* total) {
        *free = size_t(1) << 30;
        *total = size_t(2) << 30;
        return kSuccess;
    }
    // The fake's stream runs every call at once, so a copy is done when it returns.
    static Error copy_async(void* dst, const void* src, size_t bytes, CopyKind, Stream) {
        std::memmove(dst, src, bytes);
        return kSuccess;
    }
    static Error zero_async(void* dst, size_t bytes, Stream) {
        std::memset(dst, 0, bytes);
        return kSuccess;
    }
    static Error stream_create(Stream* s) {
        *s = new int(0);
        ++fake.streams;
        return kSuccess;
    }
    static Error stream_destroy(Stream s) {
        delete s;
        --fake.streams;
        return kSuccess;
    }
    static Error stream_sync(Stream) { return kSuccess; }
    static Error stream_wait_event(Stream, Event) { return kSuccess; }
    static Error wait_word(Stream, void* word, uint64_t value) {
        if (!fake.waited_word.load()) fake.first_wait_closed = *(uint64_t*)word < value;
        fake.waited_word = (uint64_t*)word;
        fake.waited_value = value;
        return kSuccess;
    }
    static Error write_word(Stream, void* word, uint64_t value) {
        *(uint64_t*)word = value;
        return kSuccess;
    }
    static Error event_create(Event* e, bool) {
        *e = new FakeEvent;
        ++fake.events;
        return kSuccess;
    }
    static Error event_destroy(Event e) {
        delete e;
        --fake.events;
        return kSuccess;
    }
    static Error event_record(Event e, Stream s) {
        e->recorded = true;
        e->stream = s;
        return kSuccess;
    }
    static Error event_query(Event e) { return e->stream && e->stream == fake.dead ? kRefused : kSuccess; }
    static Error event_sync(Event e) { return event_query(e); }
    static Error event_elapsed(float* ms, Event, Event) {
        *ms = 0.0f;
        return fake.refuse_elapsed ? kRefused : kSuccess;
    }
    static Error module_load(Module* m, const void* image) {
        if (!image) return kRefused;
        if (fake.no_binary) return kNoBinary;
        *m = new int(0);
        ++fake.modules;
        return kSuccess;
    }
    static Error module_unload(Module m) {
        delete m;
        --fake.modules;
        return kSuccess;
    }
    static Error module_function(Function* f, Module, const char* name) {
        *f = name;
        return kSuccess;
    }
    static Error function_local_bytes(int* v, Function) {
        *v = 0;
        return kSuccess;
    }
    static Error occupancy(int* blocks, Function, unsigned, size_t) {
        *blocks = 4;
        return kSuccess;
    }
    static Error launch(Function f, unsigned gx, unsigned, unsigned, unsigned, unsigned, Stream s, void* args, size_t size) {
        if (fake.refuse_launch) return kRefused;
        ++fake.launches;
        const std::string name = f;
        if (fake.recording && (name == "llmx_group_sum" || name == "llmx_group_shares" || name == "llmx_group_chain"))
            fake.groups.push_back({name, gx, s, std::vector<uint8_t>((uint8_t*)args, (uint8_t*)args + size)});
        if (fake.watching) fake.watched.push_back({name, gx, s, std::vector<uint8_t>((uint8_t*)args, (uint8_t*)args + size)});
        return kSuccess;
    }
    static Error can_access_peer(int* can, int, int) {
        *can = 0;
        return kSuccess;
    }
    static Error enable_peer_access(int) { return kSuccess; }
    static Error disable_peer_access(int) { return kSuccess; }
    static Error wall_clock_khz(int* khz, int) {
        *khz = 1000000;
        return kSuccess;
    }
};


// A group sum's arguments as GpuCollective::launch packs them (kernels/f32.gpu, llmx_group_sum).
struct SumArgs {
    const GroupPeers* peers = nullptr;
    unsigned long long n = 0, cap = 0, timeout = 0, tick = 0;
    int members = 0, rank = 0, flags = 0;
};
SumArgs sum_args(const GroupLaunch& l) {
    if (l.args.size() != 72) throw std::logic_error("a group sum's arguments of " + std::to_string(l.args.size()) + " bytes, not 72");
    SumArgs a;
    const uint8_t* b = l.args.data();
    std::memcpy(&a.peers, b, 8);
    std::memcpy(&a.n, b + 24, 8);
    std::memcpy(&a.cap, b + 32, 8);
    std::memcpy(&a.members, b + 40, 4);
    std::memcpy(&a.rank, b + 44, 4);
    std::memcpy(&a.flags, b + 48, 4);
    std::memcpy(&a.timeout, b + 56, 8);
    std::memcpy(&a.tick, b + 64, 8);
    return a;
}

// The kernel's GroupSignal (kernels/group.h) as far as the port reads it.
struct PortSignal {
    unsigned epoch[32];
    unsigned abort;
    unsigned done[32][16];
};
static_assert(offsetof(PortSignal, done) == 132 && sizeof(PortSignal) <= kGroupSignalBytes, "kernels/group.h, GroupSignal");

// The device's wall clock for the port, at the fake's rate times a speed-up.
struct PortClock {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double per_ns = 1.0;
    unsigned long long now() const { return (unsigned long long)(std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() * per_ns); }
};

// A CPU port of a member's block 0 at a meeting (kernels/f32.gpu group_meet, kernels/group.h group_stop and group_failed): it tells and waits for every peer, and where it gives up sets every abort word and its status and fault words.
// `killed` is the test's watchdog, which ends a wait the kernel would never end and leaves the words as they were; returns the clock's ticks at the end.
unsigned long long port_member(const GroupPeers* p, int members, int rank, unsigned long long timeout, unsigned long long tick, const PortClock& clock, const std::atomic<bool>& killed) {
    PortSignal* self = (PortSignal*)p->signal[rank];
    const unsigned epoch = 1;
    const unsigned long long deadline = timeout ? clock.now() + timeout : ~0ull;
    for (int j = 0; j < members; ++j)
        if (j != rank) host_store(&((PortSignal*)p->signal[j])->done[0][rank], epoch);
    bool failed = false;
    for (int j = 0; j < members && !failed; ++j) {
        if (j == rank) continue;
        unsigned long long next = clock.now() + tick;
        while (host_load(&self->done[0][j]) != epoch) {
            if (killed) return clock.now();
            if (host_load(&self->abort)) {
                failed = true;
                break;
            }
            const unsigned long long now = clock.now();
            if (now < next) continue;
            next = now + tick;
            if (now > deadline || host_load(p->cancel)) {
                failed = true;
                break;
            }
        }
    }
    if (failed) {
        for (int k = 0; k < members; ++k) host_store(&((PortSignal*)p->signal[k])->abort, 1u);
        host_store(&p->status[rank], 1u);
        host_store(p->fault, 1u);
    }
    return clock.now();
}

// The kernel a sum of this form launches.
const char* form_kernel(GpuSumForm f) { return f == GpuSumForm::one_shot ? "llmx_group_sum" : f == GpuSumForm::shares ? "llmx_group_shares" : "llmx_group_chain"; }

// Members 0 to W - 1 of fake backends, one a device index.
struct FakeGroup {
    std::vector<std::unique_ptr<GpuBackend<FakeApi>>> cards;
    std::vector<GpuBackend<FakeApi>*> members;
    explicit FakeGroup(size_t W) {
        for (size_t m = 0; m < W; ++m) {
            cards.push_back(std::make_unique<GpuBackend<FakeApi>>((int)m, false));
            members.push_back(cards.back().get());
        }
    }
};

// The launches of one sum in place of n floats.
std::vector<GroupLaunch> recorded_sum(GpuCollective<FakeApi>& c, size_t n) {
    fake.groups.clear();
    fake.recording = true;
    c.sum_in_place(1, n);
    fake.recording = false;
    return fake.groups;
}

} // namespace
} // namespace backend

using namespace backend;

int main() {
    int failures = 0;
    auto require = [&](bool ok, const std::string& what) {
        std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
        if (!ok) ++failures;
    };
    {
        GpuBackend<FakeApi> core(0, false);
        Backend& b = core;
        require(b.identity() == "fake fake; fake 1", "identity names the backend, the device and its driver: " + b.identity());
        require(fake.modules == 2, "the f32 and int16 modules load and a missing dot module is refused, not fatal");

        // A buffer written and read back, then a ticket for the work before it.
        std::vector<float> in(4096), out(4096, -1.0f);
        for (size_t i = 0; i < in.size(); ++i) in[i] = (float)i * 0.5f - 7.0f;
        BufferPtr buf = b.alloc(in.size() * sizeof(float), Memory::device);
        b.write(*buf, 0, in.data(), in.size() * sizeof(float));
        b.read(*buf, 0, out.data(), out.size() * sizeof(float));
        require(std::memcmp(in.data(), out.data(), in.size() * sizeof(float)) == 0, "16 KiB written to a device buffer reads back the same");
        BufferPtr other = b.alloc(1024 * sizeof(float), Memory::device);
        b.copy(*other, 0, *buf, 512 * sizeof(float), 1024 * sizeof(float));
        std::vector<float> part(1024);
        b.read(*other, 0, part.data(), part.size() * sizeof(float));
        require(std::memcmp(part.data(), in.data() + 512, part.size() * sizeof(float)) == 0, "a copy between buffers at an offset moves those bytes");
        const auto t = b.submit();
        b.wait(t);
        require(b.done(t), "a ticket waited on is done");

        // A refused allocation is an error that names the backend and the call.
        fake.refuse_malloc = true;
        bool named = false;
        try {
            BufferPtr big = b.alloc(size_t(64) << 20, Memory::device);
        } catch (const std::exception& e) {
            const std::string w = e.what();
            named = w.find("fake: ") == 0 && w.find("REFUSED") != std::string::npos;
            std::printf("  refused: %s\n", e.what());
        }
        fake.refuse_malloc = false;
        require(named, "a refused allocation throws with the backend's prefix and the driver's reason");
        b.add({buf.get(), 0}, {buf.get(), 0}, 1024);
        require(b.kernel_times().kernels.empty(), "a backend made without diagnostics times nothing");
    }
    {
        // A diagnostics backend times each kernel, copy and fill it issues, read once.
        fake.waited_word = nullptr;
        GpuBackend<FakeApi> timed(0, true);
        Backend& b = timed;
        BufferPtr buf = b.alloc(4096 * sizeof(float), Memory::device);
        std::vector<float> in(4096, 1.0f);
        b.write(*buf, 0, in.data(), in.size() * sizeof(float));
        b.add({buf.get(), 0}, {buf.get(), 0}, 1024);
        BufferPtr other = b.alloc(1024 * sizeof(float), Memory::device);
        b.copy(*other, 0, *buf, 0, 1024 * sizeof(float));
        b.read(*buf, 0, in.data(), 1024 * sizeof(float));
        const Backend::KernelTimes t = b.kernel_times();
        std::string names;
        for (const auto& k : t.kernels) names += " " + k.first;
        require(names == " copy fill llmx_add readback upload" && t.dispatches == 1 && t.gaps.size() == 5 && t.unresolved == 0,
                "a diagnostics backend times its fills, upload, kernel, device copy and readback apart:" + names);
        // Each flush waits behind a word the host sets once the flush has issued every op, so the host's time to issue them is idle before the first (docs/src/backends-gpu.md).
        require(fake.first_wait_closed, "the timing gate is closed when the first flush issues its wait");
        uint64_t* const gate = fake.waited_word;
        require(gate && *gate >= fake.waited_value && fake.waited_value > 0, "a diagnostics backend's flush waits behind its gate word and releases it");
        fake.refuse_launch = true;
        bool refused = false;
        try {
            b.add({buf.get(), 0}, {buf.get(), 0}, 1024);
            (void)b.submit();
        } catch (const std::exception&) {
            refused = true;
        }
        fake.refuse_launch = false;
        require(refused && gate && fake.waited_word == gate && *gate >= fake.waited_value, "a flush whose launch is refused releases its gate word");
        (void)b.kernel_times();
        // A reading times its first 4096 ops and counts the rest, so a section past the cap says how much of it went untimed.
        for (size_t i = 0; i < 4100; ++i) b.add({buf.get(), 0}, {buf.get(), 0}, 1024);
        const Backend::KernelTimes capped = b.kernel_times();
        require(capped.ops == 4096 && capped.untimed == 4 && capped.dispatches == 4096, "a reading past the cap times 4096 ops and counts the rest: " + std::to_string(capped.untimed));
        require(b.kernel_times().untimed == 0, "the next reading counts its own ops past the cap");
        require(b.kernel_times().kernels.empty(), "a reading starts the next interval empty");
        b.add({buf.get(), 0}, {buf.get(), 0}, 1024);
        fake.refuse_elapsed = true;
        const Backend::KernelTimes lost = b.kernel_times();
        fake.refuse_elapsed = false;
        require(lost.kernels.empty() && lost.dispatches == 0 && lost.unresolved == 1, "an op whose events give no time is counted as unresolved, not timed");
    }
    // Every member's launch of one sum: the form its size takes, the same blocks and arguments apart from the member's own, within the kernel's maxima, and a deadline on every sum.
    {
        const size_t rows = 32, width = 520;   // timed sizes 16640, 8320 and 4160, the rows halved down to kTimedFloor
        const GroupTransport t = GroupTransport::peer_writes;
        const std::map<size_t, GpuSumPick> forms{{4160, {t, GpuSumForm::shares}}, {8320, {t, GpuSumForm::chain}}, {16640, {t, GpuSumForm::wrap}}};
        const std::vector<size_t> sizes{1, 3, 4, 5, 255, 1023, 1024, 1025, 1027, kTimedFloor - 1, kTimedFloor, kTimedFloor + 1, 4159, 4160, 4161, 4163, 8319, 8320, 8321, 12345, 16639, 16640};
        std::string same, maxima, deadline, probe;
        size_t sums = 0;
        for (size_t W = 2; W <= kGroupMembers; ++W) {
            FakeGroup g(W);
            {
                GpuCollective<FakeApi> c(g.members, rows, width, {t});
                c.use(forms);
                for (size_t n : sizes) {
                    const std::vector<GroupLaunch> got = recorded_sum(c, n);
                    ++sums;
                    auto at = forms.upper_bound(n);
                    const GpuSumForm f = at == forms.begin() ? GpuSumForm::one_shot : (--at)->second.form;
                    const std::string where = "width " + std::to_string(W) + ", " + std::to_string(n) + " floats: ";
                    if (got.size() != W) {
                        if (same.empty()) same = where + std::to_string(got.size()) + " launches";
                        continue;
                    }
                    const SumArgs a0 = sum_args(got[0]);
                    for (size_t m = 0; m < W; ++m) {
                        const SumArgs a = sum_args(got[m]);
                        const bool wrap = (a.flags & 8) != 0;
                        if (same.empty() && (got[m].name != form_kernel(f) || wrap != (f == GpuSumForm::wrap) || got[m].blocks != got[0].blocks || a.n != n || a.cap != a0.cap || a.members != (int)W ||
                                             a.rank != (int)m || a.flags != a0.flags || !(a.flags & 2) || a.timeout != a0.timeout || a.tick != a0.tick))
                            same = where + "member " + std::to_string(m) + " launched " + got[m].name + " on " + std::to_string(got[m].blocks) + " blocks, rank " + std::to_string(a.rank) + ", flags " + std::to_string(a.flags);
                        // The one-shot and the shares take at most a block a multiprocessor, the fake device's 4.
                        const size_t most = f == GpuSumForm::one_shot || f == GpuSumForm::shares ? 4 : 2 * kChainBlocks;
                        const bool even = (f != GpuSumForm::chain && f != GpuSumForm::wrap) || got[m].blocks % 2 == 0;
                        if (maxima.empty() && (got[m].blocks < 1 || got[m].blocks > most || !even)) maxima = where + std::to_string(got[m].blocks) + " blocks of " + got[m].name;
                        if (deadline.empty() && (!a.tick || a.timeout < 1000 * a.tick))
                            deadline = where + "member " + std::to_string(m) + "'s sum waits " + std::to_string(a.timeout) + " ticks at most, " + std::to_string(a.tick) + " a millisecond";
                    }
                }
            }
            {
                GpuCollective<FakeApi> c(g.members, 1, 1027, {GroupTransport::peer_writes}, true);
                for (const GroupLaunch& l : recorded_sum(c, 1027)) {
                    const SumArgs a = sum_args(l);
                    if (probe.empty() && a.timeout != kProbeTimeoutMs * a.tick) probe = "width " + std::to_string(W) + ": a probe sum waits " + std::to_string(a.timeout) + " ticks";
                }
            }
        }
        require(same.empty(), "every member's launch of " + std::to_string(sums) + " sums at widths 2 to 16, below, at and around each size where the form changes, takes that form with the same kernel, blocks, size, staging, flags and deadline, apart from its rank" +
                                  (same.empty() ? "" : ": " + same));
        require(maxima.empty(), "every sum's blocks are within the kernel's: a block a multiprocessor for the one-shot and the shares, an even number up to 32 for the chain" + (maxima.empty() ? "" : ": " + maxima));
        require(deadline.empty(), "every serving sum gives up after at least a second of its device's clock, so a member whose peer's card is gone does not wait for ever" + (deadline.empty() ? "" : ": " + deadline));
        require(probe.empty(), "a probe sum gives up after kProbeTimeoutMs" + (probe.empty() ? "" : ": " + probe));
    }

    // A member whose card is gone: the others' waits, ported to CPU threads, end at their deadline or where the host learns of its device error, each live member's status word set and none left spinning.
    for (const bool host_error : {false, true}) {
        const size_t W = 4, dead = 2;
        FakeGroup g(W);
        const std::string what = host_error ? "where the gone member's ticket reports a device error, " : "where nothing reports the gone member, ";
        GpuCollective<FakeApi> c(g.members, 1, 1027, {GroupTransport::peer_writes});
        const std::vector<GroupLaunch> launched = recorded_sum(c, 1027);
        std::vector<SumArgs> args;
        for (const GroupLaunch& l : launched) args.push_back(sum_args(l));
        if (args.size() != W) {
            require(false, what + "a sum launches on every member");
            continue;
        }
        if (host_error) fake.dead = launched[dead].stream;
        // At the host's error the clock runs at real time, so only the cancel word can end the waits before the watchdog; else a thousand times faster, so the deadline passes within it.
        PortClock clock;
        clock.per_ns = (host_error ? 1.0 : 1000.0) * 1e6 / 1e6;   // the fake's 1 GHz clock
        std::atomic<bool> killed{false};
        std::atomic<int> ended{0};
        std::vector<unsigned long long> end(W, 0);
        std::vector<std::thread> threads;
        for (size_t r = 0; r < W; ++r)
            if (r != dead)
                threads.emplace_back([&, r] {
                    end[r] = port_member(args[r].peers, (int)W, (int)r, args[r].timeout, args[r].tick, clock, killed);
                    ++ended;
                });
        if (host_error) c.wait(0);
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (ended < (int)W - 1 && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const bool alone = ended == (int)W - 1;
        killed = true;
        for (std::thread& t : threads) t.join();
        bool status = true;
        unsigned long long first = ~0ull;
        for (size_t r = 0; r < W; ++r) {
            if ((host_load(&args[0].peers->status[r]) != 0) != (r != dead)) status = false;
            if (r != dead) first = std::min(first, end[r]);
        }
        c.wait(0);
        require(alone, what + "every live member's wait ends by itself within 5 s, none left spinning");
        require(status, what + "every live member's status word is set and the gone member's is not");
        require(!c.fast(), what + "the collective takes the conservative path for its later sums");
        if (!host_error) require(alone && first >= args[0].timeout, "no member gives up before its sum's deadline");
        bool threw = false;
        try {
            g.members[0]->raise(0);
        } catch (const std::exception&) {
            threw = true;
        }
        require(threw, what + "a live member's backend throws at its next check");
        fake.dead = nullptr;
    }
    {
        // A mixed round, decode rows on both sides of a prompt the tile kernel answers: the per-row kernel and its merge launch over the decode rows alone, named by the row list they read.
        GpuBackend<FakeApi> core(0, false);
        Backend& b = core;
        const int n_head = 4, n_head_kv = 2, dim = 128;
        const size_t prompt = 40, rows = prompt + 2, qfloats = rows * n_head * dim;
        std::unique_ptr<KVStorage> kv = b.kv_alloc(1, n_head_kv, dim, 4 * kBlockTokens, KVType::f16, KVType::f16, {}, {}, {});
        const int32_t first[1] = {0}, middle[1] = {1}, last[2] = {2, 3};
        BufferPtr q = b.alloc(qfloats * sizeof(float), Memory::device), o = b.alloc(qfloats * sizeof(float), Memory::device);
        // The views as written, then as attended: each row's history ends at its own position.
        KVView wr[3] = {{kv.get(), first, 1, 0, 11}, {kv.get(), middle, 1, 0, prompt, prompt}, {kv.get(), last, 2, 0, 101}};
        BufferPtr fill = b.alloc((11 + prompt + 101) * n_head_kv * dim * sizeof(float), Memory::device);
        b.kv_write(0, wr, 3, {fill.get(), 0}, {fill.get(), 0}, {});
        const KVView at[3] = {{kv.get(), first, 1, 10, 1}, {kv.get(), middle, 1, 0, prompt, prompt}, {kv.get(), last, 2, 100, 1}};
        fake.watched.clear();
        fake.watching = true;
        b.attention({q.get(), 0}, 0, at, 3, {o.get(), 0}, n_head, n_head_kv, dim);
        b.wait(b.submit());
        fake.watching = false;
        const GroupLaunch* vec = nullptr;
        const GroupLaunch* merge = nullptr;
        std::string names;
        for (const GroupLaunch& l : fake.watched) {
            names += " " + l.name;
            if (l.name == "llmx_attention_vec") vec = &l;
            if (l.name == "llmx_attention_vec_merge") merge = &l;
        }
        const unsigned pairs = 2 * n_head;
        bool listed = false;
        if (vec && merge && vec->args.size() >= 64 && merge->args.size() >= 24) {
            const int32_t *vrows = nullptr, *mrows = nullptr;
            std::memcpy(&vrows, vec->args.data() + 56, sizeof(vrows));
            std::memcpy(&mrows, merge->args.data() + 16, sizeof(mrows));
            listed = vrows && vrows == mrows && vrows[0] == 0 && vrows[1] == (int32_t)prompt + 1;
        }
        require(vec && merge && merge->blocks == pairs && vec->blocks > pairs && vec->blocks % pairs == 0,
                "a mixed round's per-row attention launches over its 2 decode rows' 8 (row, head) pairs and their splits, not the prompt's 40 rows:" + names +
                    (vec ? " vec " + std::to_string(vec->blocks) : "") + (merge ? " merge " + std::to_string(merge->blocks) : ""));
        require(listed, "the per-row kernel and its merge read one list of the call's decode rows, 0 and 41");
    }
    require(fake.memory == 0 && fake.host == 0, "every device and host allocation is freed with the backend");
    require(fake.streams == 0 && fake.events == 0 && fake.modules == 0, "every stream, event and module is released with the backend");
    {
        // A card of the same device and driver at another index and PCI address computes the same bits, so a kept disk cache is adopted whichever card a member lands on.
        GpuBackend<FakeApi> first(0, false), second(1, false);
        require(second.identity() == first.identity(), "another index of the same device gives the same identity: " + second.identity());
    }
    // A device whose architecture the build left out is refused with its architecture and the build option that adds it.
    fake.no_binary = true;
    std::string refusal;
    try {
        GpuBackend<FakeApi> core(0, false);
    } catch (const std::exception& e) {
        refusal = e.what();
        std::printf("  no code object: %s\n", e.what());
    }
    fake.no_binary = false;
    require(refusal.find("fake: ") == 0 && refusal.find("fake9") != std::string::npos && refusal.find("FAKE_ARCHS") != std::string::npos,
            "a device the modules hold no code object for is refused naming its architecture and the build option");
    return failures ? 1 : 0;
}
