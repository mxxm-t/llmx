#include <atomic>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <thread>
#include "backends/cpu/cpu_backend.hpp"

static std::atomic<int> fail_allocation{-1};

void* operator new(std::size_t size) {
    if (fail_allocation.load() >= 0 && fail_allocation.fetch_sub(1) == 0) {
        fail_allocation.store(-1);
        throw std::bad_alloc();
    }
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
#if defined(__GNUC__)
// Keep GCC's allocation-pair warning from inlining through this test allocator.
__attribute__((noinline))
#endif
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { ::operator delete(p); }

struct AllocationFailure {
    explicit AllocationFailure(int after) { fail_allocation.store(after); }
    ~AllocationFailure() { fail_allocation.store(-1); }
};

struct Failure { int worker; };

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

static void check_interface(backend::CpuBackend& cpu) {
    cpu.set_threads(2);
    backend::Backend& api = cpu;
    api.set_threads(0);
    require(api.threads_available() == 2, "zero thread hint changed the worker count");
    cpu.run_prefill([&] {
        api.set_threads(0);
        require(api.threads_available() == 2, "zero thread hint changed an active prefill");
    });
    std::atomic<int> workers{0};
    cpu.run_parallel([&](int) { ++workers; });
    require(workers == 2, "zero thread hint changed the pool participants");
    cpu.set_threads(1);

    const std::array<unsigned char, 4> expected{0x31, 0x61, 0x90, 0xee};
    auto bytes = expected;
    unsigned char sentinel = 0x5a;
    const auto live = api.adopt(bytes.data(), bytes.size());
    const auto copy = api.alloc(bytes.size());
    api.write(*copy, 0, expected.data(), expected.size());
    api.copy(*live, 0, *copy, 0, bytes.size());
    std::array<unsigned char, 4> read{};
    api.read(*copy, 0, read.data(), read.size());
    require(read == expected && bytes == expected, "nonempty transfers changed their bytes");

    size_t valid = 0, invalid = 0;
    auto rejects = [&](auto&& work, const char* label) {
        bool rejected = false;
        try { work(); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected, label);
        ++invalid;
    };
    for (const auto& empty : {api.alloc(0), api.adopt(nullptr, 0)}) {
        api.read(*empty, 0, &sentinel, 0); ++valid;
        api.write(*empty, 0, &sentinel, 0); ++valid;
        api.write(*empty, 0, nullptr, 0); ++valid;
        api.copy(*empty, 0, *empty, 0, 0); ++valid;
        api.copy(*live, bytes.size(), *empty, 0, 0); ++valid;
        api.copy(*empty, 0, *live, bytes.size(), 0); ++valid;
        rejects([&] { api.read(*empty, 1, &sentinel, 0); }, "empty read accepted an invalid offset");
        rejects([&] { api.write(*empty, 1, nullptr, 0); }, "empty write accepted an invalid offset");
        rejects([&] { api.copy(*empty, 1, *live, 0, 0); }, "empty copy accepted an invalid destination offset");
        rejects([&] { api.copy(*live, 0, *empty, 1, 0); }, "empty copy accepted an invalid source offset");
    }
    api.read(*live, bytes.size(), &sentinel, 0); ++valid;
    api.write(*live, bytes.size(), nullptr, 0); ++valid;
    api.copy(*live, bytes.size(), *live, bytes.size(), 0); ++valid;
    rejects([&] { api.read(*live, bytes.size() + 1, &sentinel, 0); }, "read accepted an offset beyond the end");
    rejects([&] { api.write(*live, bytes.size() + 1, nullptr, 0); }, "write accepted an offset beyond the end");
    rejects([&] { api.copy(*live, bytes.size() + 1, *copy, 0, 0); }, "copy accepted an offset beyond the destination");
    rejects([&] { api.copy(*live, 0, *copy, bytes.size() + 1, 0); }, "copy accepted an offset beyond the source");
    rejects([&] { api.write(*live, 0, nullptr, 1); }, "nonempty write accepted a null source");
    require(bytes == expected && sentinel == 0x5a, "empty or rejected transfer changed storage");
    std::printf("CPU interface: %zu empty transfers, %zu rejected ranges/sources, 3 thread-hint checks passed\n", valid, invalid);
}

static void check_dispatch(backend::CpuBackend& cpu, int threads, int failing) {
    std::atomic<int> entered{0}, finished{0};
    bool caught = false;
    try {
        cpu.run_parallel([&](int worker) {
            entered.fetch_add(1);
            while (entered.load() != threads) std::this_thread::yield();
            if (worker == failing || failing == -1) {
                finished.fetch_add(1);
                throw Failure{worker};
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            finished.fetch_add(1);
        });
    } catch (const Failure& error) {
        caught = true;
        require(failing == -1 || error.worker == failing, "wrong exception payload");
        require(finished.load() == threads, "exception escaped before workers finished");
    }
    require(caught, "task exception was swallowed");
    std::atomic<int> calls{0};
    cpu.parallel_for(37, [&](int) { calls.fetch_add(1); });
    require(calls.load() == 37, "pool not reusable after exception");
}

static void check_startup() {
    bool success = false;
    int constructor_failures = 0, resize_failures = 0;
    for (int after = 0; after < 128 && !success; ++after) {
        try {
            AllocationFailure fail(after);
            backend::CpuBackend cpu;
            success = true;
        } catch (const std::bad_alloc&) { ++constructor_failures; }
    }
    require(success && constructor_failures > 1, "constructor fault sweep incomplete");
    backend::CpuBackend cpu;
    success = false;
    for (int after = 0; after < 128 && !success; ++after) {
        cpu.set_threads(1);
        try {
            AllocationFailure fail(after);
            cpu.set_threads(4);
            success = true;
        } catch (const std::bad_alloc&) {
            ++resize_failures;
            require(cpu.threads_available() == 1, "failed startup did not leave serial backend");
            check_dispatch(cpu, 1, 0);
            cpu.set_threads(4);
            check_dispatch(cpu, 4, 2);
        }
    }
    require(success && resize_failures > 1, "resize fault sweep incomplete");
    std::printf("startup allocation failures: constructor=%d resize=%d\n",
                constructor_failures, resize_failures);
}

int main() {
    try {
        check_startup();
        backend::CpuBackend cpu;
        check_interface(cpu);
        for (int threads : {1, 2, 4}) {
            cpu.set_threads(threads);
            for (int repeat = 0; repeat < 10; ++repeat)
                for (int failing = -1; failing < threads; ++failing)
                    check_dispatch(cpu, threads, failing);
            bool caught = false;
            try {
                cpu.parallel_for(64, [](int i) { if (i == 63) throw std::bad_alloc(); });
            } catch (const std::bad_alloc&) { caught = true; }
            require(caught, "allocation exception type was lost");
            check_dispatch(cpu, threads, 0);
        }
        std::puts("backend errors: passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    } catch (...) {
        std::fputs("unexpected exception\n", stderr);
        return 1;
    }
}
