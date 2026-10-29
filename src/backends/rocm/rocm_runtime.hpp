#pragma once
// The HIP runtime opened at run time (docs/ROCM.md, Execution and graph strategy): its calls in one table, so a build with the ROCm backend still runs on the CPU or Vulkan where no ROCm is installed.
// The SDK supplies the headers at build time; nothing links against libamdhip64.

#include <hip/hip_runtime_api.h>
#include <hip/hip_version.h>

#include <dlfcn.h>

#include <stdexcept>
#include <string>
#include <type_traits>

namespace backend {

// No HIP runtime, or no device it opens: the caller reports the device as missing, as it does a Vulkan device that cannot be opened.
struct RocmUnavailable : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The calls the backend makes, each loaded once by its exported name; the C++ header overloads several, so each function type is written out.
#define LLMX_HIP_FUNCTIONS(X) \
    X(hipInit, hipError_t(unsigned)) \
    X(hipRuntimeGetVersion, hipError_t(int*)) \
    X(hipGetDeviceCount, hipError_t(int*)) \
    X(hipGetDevicePropertiesR0600, hipError_t(hipDeviceProp_t*, int)) \
    X(hipDeviceGetPCIBusId, hipError_t(char*, int, int)) \
    X(hipSetDevice, hipError_t(int)) \
    X(hipMalloc, hipError_t(void**, size_t)) \
    X(hipFree, hipError_t(void*)) \
    X(hipHostMalloc, hipError_t(void**, size_t, unsigned)) \
    X(hipHostFree, hipError_t(void*)) \
    X(hipMemcpyAsync, hipError_t(void*, const void*, size_t, hipMemcpyKind, hipStream_t)) \
    X(hipMemsetAsync, hipError_t(void*, int, size_t, hipStream_t)) \
    X(hipMemGetInfo, hipError_t(size_t*, size_t*)) \
    X(hipStreamSynchronize, hipError_t(hipStream_t)) \
    X(hipModuleLoadData, hipError_t(hipModule_t*, const void*)) \
    X(hipModuleUnload, hipError_t(hipModule_t)) \
    X(hipModuleGetFunction, hipError_t(hipFunction_t*, hipModule_t, const char*)) \
    X(hipModuleLaunchKernel, hipError_t(hipFunction_t, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, hipStream_t, void**, void**)) \
    X(hipStreamCreateWithFlags, hipError_t(hipStream_t*, unsigned)) \
    X(hipStreamDestroy, hipError_t(hipStream_t)) \
    X(hipEventCreateWithFlags, hipError_t(hipEvent_t*, unsigned)) \
    X(hipEventDestroy, hipError_t(hipEvent_t)) \
    X(hipEventRecord, hipError_t(hipEvent_t, hipStream_t)) \
    X(hipEventQuery, hipError_t(hipEvent_t)) \
    X(hipEventSynchronize, hipError_t(hipEvent_t)) \
    X(hipGetErrorName, const char*(hipError_t)) \
    X(hipGetErrorString, const char*(hipError_t))

struct HipFn {
#define LLMX_HIP_MEMBER(name, ...) std::add_pointer_t<__VA_ARGS__> name = nullptr;
    LLMX_HIP_FUNCTIONS(LLMX_HIP_MEMBER)
#undef LLMX_HIP_MEMBER
};

// The runtime library and its table, opened once a process and kept until it ends, as the runtime keeps its own state; a test substitutes a call in it to inject a failure.
inline HipFn& hip() {
    static HipFn fn = [] {
        HipFn f;
        void* lib = dlopen("libamdhip64.so", RTLD_NOW | RTLD_LOCAL);
        // A runtime-only install has only the versioned name, the major version of the headers built against.
        if (!lib) lib = dlopen(("libamdhip64.so." + std::to_string(HIP_VERSION_MAJOR)).c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!lib) throw RocmUnavailable("rocm: no HIP runtime (libamdhip64.so) could be opened");
#define LLMX_HIP_LOAD(name, ...) \
        f.name = reinterpret_cast<std::add_pointer_t<__VA_ARGS__>>(dlsym(lib, #name)); \
        if (!f.name) throw RocmUnavailable("rocm: the HIP runtime has no " #name);
        LLMX_HIP_FUNCTIONS(LLMX_HIP_LOAD)
#undef LLMX_HIP_LOAD
        if (f.hipInit(0) != hipSuccess) throw RocmUnavailable("rocm: the HIP runtime did not initialise");
        return f;
    }();
    return fn;
}

// A failed call as a runtime error naming the call and the runtime's reason.
inline void hip_check(hipError_t e, const char* what) {
    if (e == hipSuccess) return;
    const HipFn& f = hip();
    throw std::runtime_error(std::string("rocm: ") + what + " failed with " + f.hipGetErrorName(e) + " (" + f.hipGetErrorString(e) + ")");
}

} // namespace backend
