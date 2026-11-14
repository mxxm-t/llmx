// gpu-core: the shared GPU host core (gpu/backend_core.hpp) over a fake Api backed by host memory, with no device (docs/TESTS.md).
#include "backends/gpu/backend_core.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace backend {
namespace {

// What the fake has handed out and not yet taken back, each kind counted, and calls a test makes fail.
struct FakeState {
    std::atomic<long> memory{0}, host{0}, streams{0}, events{0}, modules{0}, launches{0};
    std::atomic<bool> refuse_malloc{false};
};
FakeState fake;

struct FakeDevice {
    int index = 0;
    DeviceCaps caps;
    std::string pci_bus = "0000:00:00.0", pci_root;
    size_t lanes_resident = 1024;
    DeviceProfile profile;
};

struct FakeEvent { std::atomic<bool> recorded{false}; };

struct FakeApi {
    using Error = int;
    using Stream = int*;
    using Event = FakeEvent*;
    using Module = int*;
    using Function = const char*;
    using CopyKind = int;
    using Device = FakeDevice;
    static constexpr Error kSuccess = 0, kNotReady = 1, kPeerAlreadyEnabled = 2;
    static constexpr int kRefused = 3;
    static constexpr CopyKind kToHost = 0, kToDevice = 0, kAnyCopy = 0;
    static constexpr const char* name = "fake";
    static inline const std::string prefix = "fake: ";
    static constexpr const unsigned char f32_image[1] = {1}, int16_image[1] = {2};
    static constexpr const unsigned char* f32_module = f32_image;
    static constexpr const unsigned char* int16_module = int16_image;
    static constexpr const unsigned char* int16_dot_module = nullptr;

    static Device device(int index) {
        FakeDevice d;
        d.index = index;
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
    static Error free(void* p) {
        std::free(p);
        --fake.memory;
        return kSuccess;
    }
    static Error host_malloc(void** p, size_t bytes, bool) {
        *p = std::calloc(1, bytes ? bytes : 1);
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
    static Error wait_word(Stream, void*, uint64_t) { return kSuccess; }
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
    static Error event_record(Event e, Stream) {
        e->recorded = true;
        return kSuccess;
    }
    static Error event_query(Event) { return kSuccess; }
    static Error event_sync(Event) { return kSuccess; }
    static Error event_elapsed(float* ms, Event, Event) {
        *ms = 0.0f;
        return kSuccess;
    }
    static Error module_load(Module* m, const void* image) {
        if (!image) return kRefused;
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
    static Error function_shared_bytes(int* v, Function) {
        *v = 1024;
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
    static Error launch(Function, unsigned, unsigned, unsigned, unsigned, unsigned, Stream, void*, size_t) {
        ++fake.launches;
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
        GpuBackend<FakeApi> core(0);
        Backend& b = core;
        require(b.identity() == "fake:0 fake 0000:00:00.0", "identity names the backend, the index, the device and its PCI address: " + b.identity());
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
    }
    require(fake.memory == 0 && fake.host == 0, "every device and host allocation is freed with the backend");
    require(fake.streams == 0 && fake.events == 0 && fake.modules == 0, "every stream, event and module is released with the backend");
    return failures ? 1 : 0;
}
