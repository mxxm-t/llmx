#include <atomic>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>
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

// Every op sizes a row through quant::row_bytes, so each refuses a row that ends inside a block rather than truncating it, and row runs are checked whole before any row is computed.
static void check_contracts(backend::CpuBackend& cpu) {
    size_t refused = 0;
    auto rejects = [&](auto&& work, const char* label) {
        bool rejected = false;
        try { work(); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected, label);
        ++refused;
    };
    require(quant::row_bytes(gguf::GGML_TYPE_F32, 37) == 148 && quant::row_bytes(gguf::GGML_TYPE_Q8_0, 64) == 68 &&
            quant::row_bytes(gguf::GGML_TYPE_Q6_K, 512) == 420, "row bytes differ from the block layout");
    rejects([] { quant::row_bytes(gguf::GGML_TYPE_Q4_K, 128); }, "a row inside one K-quant block was sized");
    rejects([] { quant::row_bytes(9999, 32); }, "an unknown type was sized");
    rejects([] { quant::row_bytes(gguf::GGML_TYPE_Q8_0, std::numeric_limits<size_t>::max() / 32 * 32); },
            "a row size that wraps was returned");

    // Q8_0 rows of 48 values, a block and a half, with storage for two whole blocks a row so a truncating op would run.
    // One column takes the decode path and two the batched one.
    const size_t nin = 48, nout = 2;
    std::vector<uint8_t> w(nout * 2 * gguf::Q8_0_TYPESIZE, 0);
    std::vector<float> x(3 * 64, 1.0f), wf(nout * 64, 1.0f), y(4 * 64, 7.0f);
    const auto wb = cpu.adopt(w.data(), w.size()), wfb = cpu.adopt(wf.data(), wf.size() * sizeof(float));
    const auto xb = cpu.adopt(x.data(), x.size() * sizeof(float)), yb = cpu.adopt(y.data(), y.size() * sizeof(float));
    for (size_t nbatch : {size_t(1), size_t(2)})
        rejects([&] { cpu.matmul(gguf::GGML_TYPE_Q8_0, {wb.get(), 0}, {xb.get(), 0}, {yb.get(), 0}, nin, nout, nbatch); },
                "matmul computed a row that ends inside a block");
    const uint32_t id = 1;
    rejects([&] { cpu.embed({yb.get(), 0}, gguf::GGML_TYPE_Q8_0, {wb.get(), 0}, nin, nout, &id, 1); },
            "embed gathered a row that ends inside a block");

    // Runs that reach past the call, runs out of order that would take one path, and runs short of the call.
    // The grouped projections are Q8_0 rows of 64 values, so all-decode runs would take the grouped 8-bit dots.
    const backend::RowRun past[2] = {{3, 1}, {2, 2}}, merged[2] = {{3, 1}, {2, 1}}, short_of[1] = {{1, 1}};
    for (const backend::RowRuns runs : {backend::RowRuns{past, 2}, backend::RowRuns{merged, 2}, backend::RowRuns{short_of, 1}}) {
        rejects([&] { cpu.matmul(gguf::GGML_TYPE_F32, {wfb.get(), 0}, {xb.get(), 0}, {yb.get(), 0}, 64, nout, 2, runs); },
                "matmul accepted malformed row runs");
        rejects([&] {
            cpu.matmul_group({{gguf::GGML_TYPE_Q8_0, {wb.get(), 0}, {yb.get(), 0}, 1},
                              {gguf::GGML_TYPE_Q8_0, {wb.get(), 0}, {yb.get(), 2}, 1}},
                             {xb.get(), 0}, 64, 2, runs);
        }, "matmul_group accepted malformed row runs");
    }
    for (float v : y) require(v == 7.0f, "a refused call wrote rows");
    std::printf("backend contracts: %zu refusals, no rows written\n", refused);
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
    int constructor_failures = 0, resize_failures = 0, start_failures = 0;
    for (int after = 0; after < 128 && !success; ++after) {
        try {
            AllocationFailure fail(after);
            backend::CpuBackend cpu;
            success = true;
        } catch (const std::bad_alloc&) { ++constructor_failures; }
    }
    require(success && constructor_failures > 0, "constructor fault sweep incomplete");
    backend::CpuBackend cpu;
    cpu.set_threads(2);
    check_dispatch(cpu, 2, 1);
    success = false;
    for (int after = 0; after < 128 && !success; ++after) {
        try {
            AllocationFailure fail(after);
            cpu.set_threads(3);
            success = true;
        } catch (const std::bad_alloc&) {
            ++resize_failures;
            require(cpu.threads_available() == 2, "failed count change did not keep the count");
            check_dispatch(cpu, 2, 0);
        }
    }
    require(success && resize_failures > 0, "resize fault sweep incomplete");
    success = false;
    // The first dispatch at a count starts the pool, so that is where a start fails; the count stays, and the next dispatch starts the whole pool without a new count.
    for (int after = 0; after < 128 && !success; ++after) {
        cpu.set_threads(1);
        cpu.set_threads(4);
        try {
            AllocationFailure fail(after);
            cpu.run_parallel([](int) {});
            success = true;
        } catch (const std::bad_alloc&) {
            ++start_failures;
            require(cpu.threads_available() == 4, "a failed start changed the count");
            const size_t started = cpu.workers_started();
            std::atomic<int> participants{0};
            cpu.run_parallel([&](int) { ++participants; });
            require(participants == 4 && cpu.workers_started() == started + 3, "the dispatch after a failed start did not start the whole pool");
            check_dispatch(cpu, 4, 2);
        }
    }
    require(success && start_failures > 1, "start fault sweep incomplete");
    std::printf("startup allocation failures: constructor=%d resize=%d start=%d\n",
                constructor_failures, resize_failures, start_failures);
}

// Construction and count changes start no threads; the first dispatch at a count starts one pool of that size, which later dispatches reuse.
static void check_lazy_start() {
    std::atomic<int> participants{0};
    auto dispatch = [&](backend::CpuBackend& cpu) {
        participants = 0;
        cpu.run_parallel([&](int) { ++participants; });
        return participants.load();
    };
    {
        backend::CpuBackend cpu;
        require(cpu.workers_started() == 0, "construction started workers");
        cpu.set_threads(1);
        std::atomic<int> calls{0};
        cpu.parallel_for(1000, [&](int) { ++calls; });
        require(calls == 1000 && dispatch(cpu) == 1, "serial backend dispatch changed");
        require(cpu.workers_started() == 0, "a serial backend started workers");
    }
    backend::CpuBackend cpu;
    cpu.set_threads(3);
    cpu.set_threads(5);
    require(cpu.workers_started() == 0, "a count change before any work started workers");
    require(dispatch(cpu) == 5 && cpu.workers_started() == 4, "first dispatch did not start the pool at the count in use");
    require(dispatch(cpu) == 5 && cpu.workers_started() == 4, "a second dispatch restarted the pool");
    cpu.set_threads(0);
    cpu.set_threads(5);
    require(dispatch(cpu) == 5 && cpu.workers_started() == 4, "an unchanged count restarted the pool");
    cpu.set_threads(2);
    require(cpu.workers_started() == 4, "a count change after work started workers before a dispatch");
    require(dispatch(cpu) == 2 && cpu.workers_started() == 5, "a count change after work did not start a pool of the new size");
    cpu.set_threads(1);
    require(dispatch(cpu) == 1 && cpu.workers_started() == 5, "a serial count started workers");
    std::printf("lazy pool start: %zu workers started for pools of 5 and 2\n", cpu.workers_started());
}

int main() {
    try {
        check_startup();
        check_lazy_start();
        backend::CpuBackend cpu;
        check_interface(cpu);
        check_contracts(cpu);
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
