#pragma once
// A ROCm device as HIP reports it, its wave width and units in DeviceCaps and its ISA, memory and PCI address beside them, read from the runtime and not from Vulkan (docs/ROCM.md, Hardware configuration).

#include "backends/device_profile.hpp"
#include "backends/rocm/rocm_runtime.hpp"

#include <cstdio>
#include <string>

namespace backend {

struct RocmDevice {
    int index = 0;
    DeviceCaps caps;   // the wave width and units the kernels are shaped by, and the device and runtime a measured profile is keyed by
    std::string arch, pci_bus, pci_root;
    size_t shared_per_block = 0, total_memory = 0;
};

// The device HIP numbers `index`, refused as unavailable where the runtime has no such device.
inline RocmDevice rocm_device(int index) {
    const HipFn& f = hip();
    int count = 0;
    if (f.hipGetDeviceCount(&count) != hipSuccess || count <= 0) throw RocmUnavailable("rocm: the HIP runtime sees no device");
    if (index < 0 || index >= count) throw RocmUnavailable("rocm: no device " + std::to_string(index) + " (" + std::to_string(count) + " seen)");
    hipDeviceProp_t p{};
    hip_check(f.hipGetDevicePropertiesR0600(&p, index), "hipGetDeviceProperties");
    RocmDevice d;
    d.index = index;
    d.arch = p.gcnArchName;
    d.caps.subgroup_size = (uint32_t)p.warpSize;
    d.caps.compute_units = (uint32_t)p.multiProcessorCount;
    d.caps.device = std::string(p.name) + " " + d.arch;
    d.shared_per_block = p.sharedMemPerBlock;
    d.total_memory = p.totalGlobalMem;
    char bus[32] = {};
    hip_check(f.hipDeviceGetPCIBusId(bus, sizeof(bus), index), "hipDeviceGetPCIBusId");
    d.pci_bus = bus;
    unsigned domain = 0, b = 0, dev = 0, fn = 0;
    if (std::sscanf(bus, "%x:%x:%x.%x", &domain, &b, &dev, &fn) == 4) d.pci_root = pci_root_of(domain, b, dev, fn);
    int runtime = 0;
    hip_check(f.hipRuntimeGetVersion(&runtime), "hipRuntimeGetVersion");
    d.caps.driver = "HIP " + std::to_string(runtime);
    return d;
}

} // namespace backend
