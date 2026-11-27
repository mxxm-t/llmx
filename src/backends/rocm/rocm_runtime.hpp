#pragma once
// The HIP runtime opened at run time (docs/ROCM.md, Execution and graph strategy): its calls in one table, so a build with the ROCm backend still runs on the CPU or Vulkan where no ROCm is installed.
// The SDK supplies the headers at build time; nothing links against libamdhip64 or amdhip64.dll.

#include <hip/hip_runtime_api.h>
#include <hip/hip_deprecated.h>
#include <hip/hip_version.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

#include <cstdio>
#include <cstdlib>
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
    X(hipDeviceGetPCIBusId, hipError_t(char*, int, int)) \
    X(hipSetDevice, hipError_t(int)) \
    X(hipMalloc, hipError_t(void**, size_t)) \
    X(hipExtMallocWithFlags, hipError_t(void**, size_t, unsigned)) \
    X(hipDeviceCanAccessPeer, hipError_t(int*, int, int)) \
    X(hipDeviceEnablePeerAccess, hipError_t(int, unsigned)) \
    X(hipDeviceDisablePeerAccess, hipError_t(int)) \
    X(hipDeviceGetAttribute, hipError_t(int*, hipDeviceAttribute_t, int)) \
    X(hipFree, hipError_t(void*)) \
    X(hipHostMalloc, hipError_t(void**, size_t, unsigned)) \
    X(hipHostFree, hipError_t(void*)) \
    X(hipHostRegister, hipError_t(void*, size_t, unsigned)) \
    X(hipHostUnregister, hipError_t(void*)) \
    X(hipHostGetDevicePointer, hipError_t(void**, void*, unsigned)) \
    X(hipMemcpyAsync, hipError_t(void*, const void*, size_t, hipMemcpyKind, hipStream_t)) \
    X(hipMemsetAsync, hipError_t(void*, int, size_t, hipStream_t)) \
    X(hipMemGetInfo, hipError_t(size_t*, size_t*)) \
    X(hipStreamSynchronize, hipError_t(hipStream_t)) \
    X(hipModuleLoadData, hipError_t(hipModule_t*, const void*)) \
    X(hipModuleUnload, hipError_t(hipModule_t)) \
    X(hipModuleGetFunction, hipError_t(hipFunction_t*, hipModule_t, const char*)) \
    X(hipFuncGetAttribute, hipError_t(int*, hipFunction_attribute, hipFunction_t)) \
    X(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor, hipError_t(int*, hipFunction_t, int, size_t)) \
    X(hipModuleLaunchKernel, hipError_t(hipFunction_t, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, hipStream_t, void**, void**)) \
    X(hipStreamCreateWithFlags, hipError_t(hipStream_t*, unsigned)) \
    X(hipStreamDestroy, hipError_t(hipStream_t)) \
    X(hipEventCreateWithFlags, hipError_t(hipEvent_t*, unsigned)) \
    X(hipEventDestroy, hipError_t(hipEvent_t)) \
    X(hipEventRecord, hipError_t(hipEvent_t, hipStream_t)) \
    X(hipEventQuery, hipError_t(hipEvent_t)) \
    X(hipEventElapsedTime, hipError_t(float*, hipEvent_t, hipEvent_t)) \
    X(hipEventSynchronize, hipError_t(hipEvent_t)) \
    X(hipStreamWaitEvent, hipError_t(hipStream_t, hipEvent_t, unsigned)) \
    X(hipStreamWaitValue64, hipError_t(hipStream_t, void*, uint64_t, unsigned, uint64_t)) \
    X(hipStreamWriteValue64, hipError_t(hipStream_t, void*, uint64_t, unsigned)) \
    X(hipGetErrorName, const char*(hipError_t)) \
    X(hipGetErrorString, const char*(hipError_t))

struct HipFn {
#define LLMX_HIP_MEMBER(name, ...) std::add_pointer_t<__VA_ARGS__> name = nullptr;
    LLMX_HIP_FUNCTIONS(LLMX_HIP_MEMBER)
#undef LLMX_HIP_MEMBER
    // Device properties in the HIP 6 layout, or, where a HIP 5 runtime lacks that entry (the Windows display driver's amdhip64.dll), in the HIP 5 layout.
    hipError_t (*hipGetDevicePropertiesR0600)(hipDeviceProp_t*, int) = nullptr;
    hipError_t (*hipGetDevicePropertiesR0000)(hipDeviceProp_tR0000*, int) = nullptr;
};

// The HIP runtime library, opened once a process; a runtime-only Linux install has only the name versioned by the headers' major version, and Windows tries that version first.
inline void* hip_library() {
    static void* lib = [] {
#if defined(_WIN32)
        // The executable's own directory and System32 only, so a DLL in the working directory is never loaded; amdhip64.dll is the display driver's runtime.
        const std::wstring major = std::to_wstring(HIP_VERSION_MAJOR);
        for (const std::wstring& name : {L"amdhip64_" + major + L".dll", std::wstring(L"amdhip64_7.dll"), std::wstring(L"amdhip64_6.dll"), std::wstring(L"amdhip64.dll")}) {
            if (HMODULE h = LoadLibraryExW(name.c_str(), nullptr, LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32)) return (void*)h;
        }
        return (void*)nullptr;
#else
        void* h = dlopen("libamdhip64.so", RTLD_NOW | RTLD_LOCAL);
        if (!h) h = dlopen(("libamdhip64.so." + std::to_string(HIP_VERSION_MAJOR)).c_str(), RTLD_NOW | RTLD_LOCAL);
        return h;
#endif
    }();
    return lib;
}

// An exported function of the opened runtime, or null.
inline void* hip_symbol(const char* name) {
    void* lib = hip_library();
    if (!lib) return nullptr;
#if defined(_WIN32)
    return (void*)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

// The runtime's table, loaded once a process and kept until it ends, as the runtime keeps its own state; a test substitutes a call in it to inject a failure.
inline HipFn& hip() {
    static HipFn fn = [] {
        HipFn f;
#if defined(_WIN32)
        if (!hip_library()) throw RocmUnavailable("rocm: no HIP runtime (amdhip64_7.dll, amdhip64_6.dll or amdhip64.dll) could be opened");
#else
        if (!hip_library()) throw RocmUnavailable("rocm: no HIP runtime (libamdhip64.so) could be opened");
#endif
#define LLMX_HIP_LOAD(name, ...) \
        f.name = reinterpret_cast<std::add_pointer_t<__VA_ARGS__>>(hip_symbol(#name)); \
        if (!f.name) throw RocmUnavailable("rocm: the HIP runtime has no " #name);
        LLMX_HIP_FUNCTIONS(LLMX_HIP_LOAD)
#undef LLMX_HIP_LOAD
        f.hipGetDevicePropertiesR0600 = reinterpret_cast<decltype(f.hipGetDevicePropertiesR0600)>(hip_symbol("hipGetDevicePropertiesR0600"));
        if (!f.hipGetDevicePropertiesR0600) f.hipGetDevicePropertiesR0000 = reinterpret_cast<decltype(f.hipGetDevicePropertiesR0000)>(hip_symbol("hipGetDeviceProperties"));
        if (!f.hipGetDevicePropertiesR0600 && !f.hipGetDevicePropertiesR0000) throw RocmUnavailable("rocm: the HIP runtime has no hipGetDevicePropertiesR0600");
        if (f.hipInit(0) != hipSuccess) throw RocmUnavailable("rocm: the HIP runtime did not initialise");
        // The process ends here, before HIP's own teardown, which can free what a still-queued completion handler reads (docs/ROCM.md, Process exit).
        // Windows has no exit hook that sees the status, so it keeps the runtime's teardown.
#if !defined(_WIN32)
        on_exit([](int status, void*) { std::fflush(nullptr); _exit(status); }, nullptr);
#endif
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
