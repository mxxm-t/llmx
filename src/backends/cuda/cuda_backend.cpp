// The CUDA backend (docs/CUDA.md): the shared GPU core (gpu/backend_core.hpp) over the CUDA driver, its calls through the table opened at run time (cuda_runtime.hpp).
#include "backends/cuda/cuda_backend.hpp"

#include "backends/cuda/cuda_device.hpp"
#include "backends/cuda/cuda_runtime.hpp"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace backend {
namespace {

// The dense F32 kernels, SASS for each target and PTX for later ones (gpu/kernels/f32.gpu).
const unsigned char kF32Module[] = {
#include "cuda/f32.inc"
};
// The F16 class's quantized products on a block-int16 twin (gpu/kernels/int16.gpu).
const unsigned char kInt16Module[] = {
#include "cuda/int16.inc"
};

// The primary context of each device, retained once a process and made current on the calling thread, as hipSetDevice makes a device current.
CUresult set_context(int index) {
    static std::mutex m;
    static std::vector<CUcontext> contexts;
    CUcontext c = nullptr;
    {
        std::lock_guard<std::mutex> lock(m);
        if ((size_t)index >= contexts.size()) contexts.resize((size_t)index + 1, nullptr);
        if (!contexts[(size_t)index]) {
            CUdevice d = 0;
            if (CUresult e = cu().cuDeviceGet(&d, index)) return e;
            if (CUresult e = cu().cuDevicePrimaryCtxRetain(&contexts[(size_t)index], d)) return e;
        }
        c = contexts[(size_t)index];
    }
    return cu().cuCtxSetCurrent(c);
}

CUdeviceptr dptr(const void* p) { return (CUdeviceptr)(uintptr_t)p; }

// The core's calls over the CUDA driver, each through cu() at the call, so a test that substitutes a call in the table reaches the backend.
struct CudaApi {
    using Error = CUresult;
    using Stream = CUstream;
    using Event = CUevent;
    using Module = CUmodule;
    using Function = CUfunction;
    using CopyKind = int;   // the driver infers the direction from unified addresses
    using Device = CudaDevice;
    static constexpr Error kSuccess = CUDA_SUCCESS, kNotReady = CUDA_ERROR_NOT_READY, kPeerAlreadyEnabled = CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED;
    static constexpr CopyKind kToHost = 0, kToDevice = 0, kAnyCopy = 0;
    static constexpr const char* name = "cuda";
    static inline const std::string prefix = "cuda: ";
    static constexpr const unsigned char* f32_module = kF32Module;
    static constexpr const unsigned char* int16_module = kInt16Module;
    // No integer-dot module yet: its builds come with the F16 class (docs/CUDA.md, Order of work), and the core runs the plain module.
    static constexpr const unsigned char* int16_dot_module = nullptr;

    static Device device(int index) { return cuda_device(index); }
    static void check(Error e, const char* what) { cu_check(e, what); }
    static const char* error_name(Error e) {
        const char* s = "?";
        (void)cu().cuGetErrorName(e, &s);
        return s;
    }
    static Error set_device(int index) { return set_context(index); }
    static Error malloc(void** p, size_t bytes) {
        CUdeviceptr d = 0;
        const Error e = cu().cuMemAlloc(&d, bytes);
        *p = (void*)(uintptr_t)d;
        return e;
    }
    // Device memory is not held in the host's caches, so a peer's writes to it are seen as they land.
    static Error malloc_uncached(void** p, size_t bytes) { return malloc(p, bytes); }
    static Error free(void* p) { return cu().cuMemFree(dptr(p)); }
    // Pinned host memory is coherent unless write-combined, so both kinds are one allocation, mapped for every context.
    static Error host_malloc(void** p, size_t bytes, bool) { return cu().cuMemHostAlloc(p, bytes, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP); }
    static Error host_free(void* p) { return cu().cuMemFreeHost(p); }
    static Error host_device_pointer(void** device, void* host) {
        CUdeviceptr d = 0;
        const Error e = cu().cuMemHostGetDevicePointer(&d, host, 0);
        *device = (void*)(uintptr_t)d;
        return e;
    }
    static Error host_register(void* p, size_t bytes) { return cu().cuMemHostRegister(p, bytes, CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP); }
    static Error host_unregister(void* p) { return cu().cuMemHostUnregister(p); }
    static Error mem_info(size_t* free, size_t* total) { return cu().cuMemGetInfo(free, total); }
    static Error copy_async(void* dst, const void* src, size_t bytes, CopyKind, Stream s) { return cu().cuMemcpyAsync(dptr(dst), dptr(src), bytes, s); }
    static Error zero_async(void* dst, size_t bytes, Stream s) { return cu().cuMemsetD8Async(dptr(dst), 0, bytes, s); }
    static Error stream_create(Stream* s) { return cu().cuStreamCreate(s, CU_STREAM_NON_BLOCKING); }
    static Error stream_destroy(Stream s) { return cu().cuStreamDestroy(s); }
    static Error stream_sync(Stream s) { return cu().cuStreamSynchronize(s); }
    static Error stream_wait_event(Stream s, Event e) { return cu().cuStreamWaitEvent(s, e, 0); }
    // The stream waits until the word is at least `value`, or writes `value` there once the work before it is done.
    static Error wait_word(Stream s, void* word, uint64_t value) { return cu().cuStreamWaitValue64(s, dptr(word), value, CU_STREAM_WAIT_VALUE_GEQ); }
    static Error write_word(Stream s, void* word, uint64_t value) { return cu().cuStreamWriteValue64(s, dptr(word), value, CU_STREAM_WRITE_VALUE_DEFAULT); }
    static Error event_create(Event* e, bool timed) { return cu().cuEventCreate(e, timed ? CU_EVENT_DEFAULT : CU_EVENT_DISABLE_TIMING); }
    static Error event_destroy(Event e) { return cu().cuEventDestroy(e); }
    static Error event_record(Event e, Stream s) { return cu().cuEventRecord(e, s); }
    static Error event_query(Event e) { return cu().cuEventQuery(e); }
    static Error event_sync(Event e) { return cu().cuEventSynchronize(e); }
    static Error event_elapsed(float* ms, Event start, Event end) { return cu().cuEventElapsedTime(ms, start, end); }
    // A missing image is a module the build did not compile, refused as the driver refuses one with no code for the device.
    static Error module_load(Module* m, const void* image) { return image ? cu().cuModuleLoadData(m, image) : CUDA_ERROR_NO_BINARY_FOR_GPU; }
    static Error module_unload(Module m) { return cu().cuModuleUnload(m); }
    static Error module_function(Function* f, Module m, const char* name) { return cu().cuModuleGetFunction(f, m, name); }
    static Error function_shared_bytes(int* v, Function f) { return cu().cuFuncGetAttribute(v, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, f); }
    static Error function_local_bytes(int* v, Function f) { return cu().cuFuncGetAttribute(v, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, f); }
    static Error occupancy(int* blocks, Function f, unsigned threads, size_t shared) { return cu().cuOccupancyMaxActiveBlocksPerMultiprocessor(blocks, f, (int)threads, shared); }
    // A launch of `threads` lanes a block whose arguments are `size` bytes at `args`, laid out as the kernel's parameter buffer.
    static Error launch(Function f, unsigned gx, unsigned gy, unsigned gz, unsigned threads, unsigned shared, Stream s, void* args, size_t size) {
        void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, args, CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END};
        return cu().cuLaunchKernel(f, gx, gy, gz, threads, 1, 1, shared, s, nullptr, extra);
    }
    static Error can_access_peer(int* can, int device, int peer) {
        CUdevice a = 0, b = 0;
        if (Error e = cu().cuDeviceGet(&a, device)) return e;
        if (Error e = cu().cuDeviceGet(&b, peer)) return e;
        return cu().cuDeviceCanAccessPeer(can, a, b);
    }
    // The peer's primary context is made current only to name it, then the caller's is current again.
    static Error enable_peer_access(int peer) {
        CUcontext mine = nullptr, theirs = nullptr;
        if (Error e = cu().cuCtxGetCurrent(&mine)) return e;
        if (Error e = set_context(peer)) return e;
        if (Error e = cu().cuCtxGetCurrent(&theirs)) return e;
        if (Error e = cu().cuCtxSetCurrent(mine)) return e;
        return cu().cuCtxEnablePeerAccess(theirs, 0);
    }
    static Error disable_peer_access(int peer) {
        CUcontext mine = nullptr, theirs = nullptr;
        if (Error e = cu().cuCtxGetCurrent(&mine)) return e;
        if (Error e = set_context(peer)) return e;
        if (Error e = cu().cuCtxGetCurrent(&theirs)) return e;
        if (Error e = cu().cuCtxSetCurrent(mine)) return e;
        return cu().cuCtxDisablePeerAccess(theirs);
    }
    // The kernels' wall clock is the global timer, in nanoseconds.
    static Error wall_clock_khz(int* khz, int) {
        *khz = 1000000;
        return CUDA_SUCCESS;
    }
};

} // namespace
} // namespace backend

#include "backends/gpu/backend_core.hpp"

namespace backend {

using CudaBackend = GpuBackend<CudaApi>;

BackendPtr make_cuda_backend(int device) { return std::make_shared<CudaBackend>(device); }

std::map<std::string, size_t> cuda_take_launches(Backend& backend) { return dynamic_cast<CudaBackend&>(backend).take_launches(); }

} // namespace backend
