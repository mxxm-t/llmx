#pragma once
// The CUDA driver opened at run time (docs/CUDA.md, Outcome and scope): its calls in one table, so a build with the CUDA backend still runs on the CPU where no driver is installed.
// The toolkit supplies the header at build time; nothing links against libcuda or the CUDA runtime.

#include <cuda.h>

#include <dlfcn.h>

#include <stdexcept>
#include <string>

namespace backend {

// No driver, or no device it opens: the caller reports the device as missing, as it does a ROCm device.
struct CudaUnavailable : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The calls the backend makes; cuda.h maps a versioned call's name to its current symbol (cuMemAlloc to cuMemAlloc_v2), and the table loads that symbol with the header's own type.
#define LLMX_CU_FUNCTIONS(X) \
    X(cuInit) X(cuDriverGetVersion) X(cuDeviceGetCount) X(cuDeviceGet) X(cuDeviceGetName) X(cuDeviceGetAttribute) X(cuDeviceTotalMem) \
    X(cuDevicePrimaryCtxRetain) X(cuCtxSetCurrent) X(cuCtxGetCurrent) X(cuDeviceCanAccessPeer) X(cuCtxEnablePeerAccess) \
    X(cuMemAlloc) X(cuMemFree) X(cuMemHostAlloc) X(cuMemFreeHost) X(cuMemHostRegister) X(cuMemHostUnregister) X(cuMemHostGetDevicePointer) \
    X(cuMemcpyAsync) X(cuMemsetD8Async) X(cuMemGetInfo) \
    X(cuStreamCreate) X(cuStreamDestroy) X(cuStreamSynchronize) X(cuStreamWaitEvent) X(cuStreamWaitValue64) X(cuStreamWriteValue64) \
    X(cuEventCreate) X(cuEventDestroy) X(cuEventRecord) X(cuEventQuery) X(cuEventSynchronize) X(cuEventElapsedTime) \
    X(cuModuleLoadData) X(cuModuleUnload) X(cuModuleGetFunction) X(cuFuncGetAttribute) X(cuOccupancyMaxActiveBlocksPerMultiprocessor) X(cuLaunchKernel) \
    X(cuGetErrorName) X(cuGetErrorString)

#define LLMX_CU_SYMBOL_(name) #name
#define LLMX_CU_SYMBOL(name) LLMX_CU_SYMBOL_(name)

struct CuFn {
#define LLMX_CU_MEMBER(name) decltype(&::name) name = nullptr;
    LLMX_CU_FUNCTIONS(LLMX_CU_MEMBER)
#undef LLMX_CU_MEMBER
};

// The table filled from `lib`, the driver's library handle, or CudaUnavailable naming the first symbol it lacks or a driver that does not initialise.
inline CuFn cu_load(void* lib) {
    CuFn f;
#define LLMX_CU_LOAD(name) \
    f.name = reinterpret_cast<decltype(&::name)>(dlsym(lib, LLMX_CU_SYMBOL(name))); \
    if (!f.name) throw CudaUnavailable(std::string("cuda: the driver has no ") + LLMX_CU_SYMBOL(name));
    LLMX_CU_FUNCTIONS(LLMX_CU_LOAD)
#undef LLMX_CU_LOAD
    if (f.cuInit(0) != CUDA_SUCCESS) throw CudaUnavailable("cuda: the driver did not initialise");
    return f;
}

// The driver library and its table, opened once a process and kept until it ends; a test substitutes a call in it to inject a failure.
inline CuFn& cu() {
    static CuFn fn = [] {
        void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!lib) throw CudaUnavailable("cuda: no driver (libcuda.so.1) could be opened");
        return cu_load(lib);
    }();
    return fn;
}

// A failed call as a runtime error naming the call and the driver's reason.
inline void cu_check(CUresult e, const char* what) {
    if (e == CUDA_SUCCESS) return;
    const CuFn& f = cu();
    const char* name = "?";
    const char* text = "?";
    (void)f.cuGetErrorName(e, &name);
    (void)f.cuGetErrorString(e, &text);
    throw std::runtime_error(std::string("cuda: ") + what + " failed with " + name + " (" + text + ")");
}

} // namespace backend
