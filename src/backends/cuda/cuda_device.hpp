#pragma once
// A CUDA device as the driver reports it, its warp width and units in DeviceCaps and its SM version, memory and PCI address beside them (docs/CUDA.md, Owners and boundaries).

#include "backends/cuda/cuda_runtime.hpp"
#include "backends/device_profile.hpp"

#include <cstdio>
#include <string>

namespace backend {

struct CudaDevice {
    int index = 0;      // the driver's ordinal, which cuda:N names
    CUdevice handle = 0;
    DeviceCaps caps;    // the warp width and units the kernels are shaped by, and the device and driver a measured profile is keyed by
    std::string arch, pci_bus, pci_root;   // arch is sm_XY
    size_t total_memory = 0;
    size_t lanes_resident = 0;   // the threads a multiprocessor holds at once
    DeviceProfile profile;
};

// The device the driver numbers `index`, refused as unavailable where the driver has no such device.
inline CudaDevice cuda_device(int index) {
    const CuFn& f = cu();
    int count = 0;
    if (f.cuDeviceGetCount(&count) != CUDA_SUCCESS || count <= 0) throw CudaUnavailable("cuda: the driver sees no device");
    if (index < 0 || index >= count) throw CudaUnavailable("cuda: no device " + std::to_string(index) + " (" + std::to_string(count) + " seen)");
    CudaDevice d;
    d.index = index;
    cu_check(f.cuDeviceGet(&d.handle, index), "cuDeviceGet");
    auto attr = [&](CUdevice_attribute a, const char* what) {
        int v = 0;
        cu_check(f.cuDeviceGetAttribute(&v, a, d.handle), what);
        return v;
    };
    char name[256] = {};
    cu_check(f.cuDeviceGetName(name, sizeof(name) - 1, d.handle), "cuDeviceGetName");
    d.arch = "sm_" + std::to_string(attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, "compute capability")) +
             std::to_string(attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, "compute capability"));
    d.caps.subgroup_size = (uint32_t)attr(CU_DEVICE_ATTRIBUTE_WARP_SIZE, "warp size");
    d.caps.compute_units = (uint32_t)attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, "multiprocessor count");
    d.caps.shared_bytes = (uint32_t)attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, "shared memory per block");
    d.caps.device = std::string(name) + " " + d.arch;
    d.lanes_resident = (size_t)attr(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, "threads per multiprocessor");
    cu_check(f.cuDeviceTotalMem(&d.total_memory, d.handle), "cuDeviceTotalMem");
    const unsigned domain = (unsigned)attr(CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, "PCI domain"), bus = (unsigned)attr(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, "PCI bus"),
                   dev = (unsigned)attr(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, "PCI device");
    char text[32];
    std::snprintf(text, sizeof(text), "%04x:%02x:%02x.0", domain, bus, dev);
    d.pci_bus = text;
    d.pci_root = pci_root_of(domain, bus, dev, 0);
    int driver = 0;
    cu_check(f.cuDriverGetVersion(&driver), "cuDriverGetVersion");
    d.caps.driver = "CUDA " + std::to_string(driver);
    return d;
}

} // namespace backend
