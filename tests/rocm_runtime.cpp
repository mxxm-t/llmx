// rocm-runtime: the HIP runtime opened at run time and device 0 as it reports itself, its memory round trip through a stream with events, and a refused allocation named; exits 77 without a runtime or a device.
#include "backends/rocm/rocm_device.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using namespace backend;

int main() {
    RocmDevice d;
    try {
        d = rocm_device(0);
    } catch (const RocmUnavailable& e) {
        std::cout << "rocm-runtime: skipped, " << e.what() << "\n";
        return 77;
    }
    int failures = 0;
    auto require = [&](bool ok, const char* what) {
        std::cout << (ok ? "PASS " : "FAIL ") << what << "\n";
        if (!ok) ++failures;
    };
    const HipFn& f = hip();
    std::cout << "device 0: " << d.caps.device << ", wave " << d.caps.subgroup_size << ", " << d.caps.compute_units << " compute units, "
              << (d.total_memory >> 20) << " MiB, bus " << d.pci_bus << " under " << (d.pci_root.empty() ? "?" : d.pci_root) << ", " << d.caps.driver << "\n";
    require(d.caps.subgroup_size == 32 || d.caps.subgroup_size == 64, "the wave width is 32 or 64");
    require(!d.arch.empty() && d.caps.compute_units > 0 && d.total_memory > 0, "the device reports its ISA, units and memory");

    hip_check(f.hipSetDevice(0), "hipSetDevice");
    hipStream_t stream = nullptr;
    hip_check(f.hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreateWithFlags");
    hipEvent_t done = nullptr;
    hip_check(f.hipEventCreateWithFlags(&done, hipEventDisableTiming), "hipEventCreateWithFlags");

    // Pinned host memory to the device and back on one stream, ordered by the stream alone; the event retires once both copies have.
    const size_t n = size_t(1) << 22;
    void* host = nullptr;
    void* dev = nullptr;
    hip_check(f.hipHostMalloc(&host, n, hipHostMallocDefault), "hipHostMalloc");
    hip_check(f.hipMalloc(&dev, n), "hipMalloc");
    std::vector<uint8_t> want(n);
    for (size_t i = 0; i < n; ++i) want[i] = uint8_t(i * 131 + 7);
    std::memcpy(host, want.data(), n);
    hip_check(f.hipMemcpyAsync(dev, host, n, hipMemcpyHostToDevice, stream), "hipMemcpyAsync");
    void* host2 = nullptr;
    hip_check(f.hipHostMalloc(&host2, n, hipHostMallocDefault), "hipHostMalloc");
    hip_check(f.hipMemcpyAsync(host2, dev, n, hipMemcpyDeviceToHost, stream), "hipMemcpyAsync");
    hip_check(f.hipEventRecord(done, stream), "hipEventRecord");
    hip_check(f.hipEventSynchronize(done), "hipEventSynchronize");
    require(f.hipEventQuery(done) == hipSuccess, "an event synchronised on reads as retired");
    require(std::memcmp(host2, want.data(), n) == 0, "4 MiB round trip host to device to host on one stream");

    // A refused allocation is a runtime error that names the call.
    bool named = false;
    void* huge = nullptr;
    try {
        hip_check(f.hipMalloc(&huge, d.total_memory * 4), "hipMalloc");
    } catch (const std::runtime_error& e) {
        named = std::strstr(e.what(), "hipMalloc") != nullptr;
    }
    require(named, "an allocation past the device's memory is refused by name");

    hip_check(f.hipFree(dev), "hipFree");
    hip_check(f.hipHostFree(host), "hipHostFree");
    hip_check(f.hipHostFree(host2), "hipHostFree");
    hip_check(f.hipEventDestroy(done), "hipEventDestroy");
    hip_check(f.hipStreamDestroy(stream), "hipStreamDestroy");
    std::cout << (failures ? "rocm-runtime: FAIL\n" : "rocm-runtime: PASS\n");
    return failures ? 1 : 0;
}
