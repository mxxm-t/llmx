// The ROCm backend (docs/ROCM.md): the shared GPU core (gpu/backend_core.hpp) over HIP, its calls through the runtime table opened at run time (rocm_runtime.hpp).
#include "backends/rocm/rocm_backend.hpp"

#include "backends/rocm/rocm_device.hpp"
#include "backends/rocm/rocm_runtime.hpp"

#include <string>

namespace backend {
namespace {

// The dense F32 kernels, a code object for each target (gpu/kernels/f32.gpu).
const unsigned char kF32Module[] = {
#include "rocm/f32.inc"
};
// The F16 class's quantized products on a block-int16 twin (gpu/kernels/int16.gpu).
const unsigned char kInt16Module[] = {
#include "rocm/int16.inc"
};
// The same kernels on the 16-bit integer dot, compiled only for the targets with the instruction; a device it holds no code object for runs the module above.
const unsigned char kInt16DotModule[] = {
#include "rocm/int16dot.inc"
};

// The core's calls over HIP, each through hip() at the call, so a test that substitutes a call in the table reaches the backend.
struct HipApi {
    using Error = hipError_t;
    using Stream = hipStream_t;
    using Event = hipEvent_t;
    using Module = hipModule_t;
    using Function = hipFunction_t;
    using CopyKind = hipMemcpyKind;
    using Device = RocmDevice;
    static constexpr Error kSuccess = hipSuccess, kNotReady = hipErrorNotReady, kPeerAlreadyEnabled = hipErrorPeerAccessAlreadyEnabled;
    static constexpr CopyKind kToHost = hipMemcpyDeviceToHost, kToDevice = hipMemcpyHostToDevice, kAnyCopy = hipMemcpyDefault;
    static constexpr const char* name = "rocm";
    static inline const std::string prefix = "rocm: ";
    static constexpr const unsigned char* f32_module = kF32Module;
    static constexpr const unsigned char* int16_module = kInt16Module;
    static constexpr const unsigned char* int16_dot_module = kInt16DotModule;

    static Device device(int index) { return rocm_device(index); }
    static void check(Error e, const char* what) { hip_check(e, what); }
    static const char* error_name(Error e) { return hip().hipGetErrorName(e); }
    static Error set_device(int index) { return hip().hipSetDevice(index); }
    static Error malloc(void** p, size_t bytes) { return hip().hipMalloc(p, bytes); }
    // Device memory the host's caches do not hold, so a peer's writes to it are seen without a flush (the collective's staging).
    static Error malloc_uncached(void** p, size_t bytes) { return hip().hipExtMallocWithFlags(p, bytes, hipDeviceMallocUncached); }
    static Error free(void* p) { return hip().hipFree(p); }
    static Error host_malloc(void** p, size_t bytes, bool coherent) { return hip().hipHostMalloc(p, bytes, coherent ? hipHostMallocCoherent : hipHostMallocDefault); }
    static Error host_free(void* p) { return hip().hipHostFree(p); }
    static Error host_device_pointer(void** device, void* host) { return hip().hipHostGetDevicePointer(device, host, 0); }
    static Error host_register(void* p, size_t bytes) { return hip().hipHostRegister(p, bytes, hipHostRegisterDefault); }
    static Error host_unregister(void* p) { return hip().hipHostUnregister(p); }
    static Error mem_info(size_t* free, size_t* total) { return hip().hipMemGetInfo(free, total); }
    static Error copy_async(void* dst, const void* src, size_t bytes, CopyKind kind, Stream s) { return hip().hipMemcpyAsync(dst, src, bytes, kind, s); }
    static Error zero_async(void* dst, size_t bytes, Stream s) { return hip().hipMemsetAsync(dst, 0, bytes, s); }
    static Error stream_create(Stream* s) { return hip().hipStreamCreateWithFlags(s, hipStreamNonBlocking); }
    static Error stream_destroy(Stream s) { return hip().hipStreamDestroy(s); }
    static Error stream_sync(Stream s) { return hip().hipStreamSynchronize(s); }
    static Error stream_wait_event(Stream s, Event e) { return hip().hipStreamWaitEvent(s, e, 0); }
    // The stream waits until the word is at least `value`, or writes `value` there once the work before it is done.
    static Error wait_word(Stream s, void* word, uint64_t value) { return hip().hipStreamWaitValue64(s, word, value, hipStreamWaitValueGte, ~0ull); }
    static Error write_word(Stream s, void* word, uint64_t value) { return hip().hipStreamWriteValue64(s, word, value, 0); }
    static Error event_create(Event* e, bool timed) { return hip().hipEventCreateWithFlags(e, timed ? hipEventDefault : hipEventDisableTiming); }
    static Error event_destroy(Event e) { return hip().hipEventDestroy(e); }
    static Error event_record(Event e, Stream s) { return hip().hipEventRecord(e, s); }
    static Error event_query(Event e) { return hip().hipEventQuery(e); }
    static Error event_sync(Event e) { return hip().hipEventSynchronize(e); }
    static Error event_elapsed(float* ms, Event start, Event end) { return hip().hipEventElapsedTime(ms, start, end); }
    static Error module_load(Module* m, const void* image) { return hip().hipModuleLoadData(m, image); }
    static Error module_unload(Module m) { return hip().hipModuleUnload(m); }
    static Error module_function(Function* f, Module m, const char* name) { return hip().hipModuleGetFunction(f, m, name); }
    static Error function_shared_bytes(int* v, Function f) { return hip().hipFuncGetAttribute(v, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, f); }
    static Error function_local_bytes(int* v, Function f) { return hip().hipFuncGetAttribute(v, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, f); }
    static Error occupancy(int* blocks, Function f, unsigned threads, size_t shared) { return hip().hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(blocks, f, (int)threads, shared); }
    // A launch of `threads` lanes a block whose arguments are `size` bytes at `args`, laid out as the kernel's argument segment.
    static Error launch(Function f, unsigned gx, unsigned gy, unsigned gz, unsigned threads, unsigned shared, Stream s, void* args, size_t size) {
        void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
        return hip().hipModuleLaunchKernel(f, gx, gy, gz, threads, 1, 1, shared, s, nullptr, extra);
    }
    static Error can_access_peer(int* can, int device, int peer) { return hip().hipDeviceCanAccessPeer(can, device, peer); }
    static Error enable_peer_access(int peer) { return hip().hipDeviceEnablePeerAccess(peer, 0); }
    static Error disable_peer_access(int peer) { return hip().hipDeviceDisablePeerAccess(peer); }
    static Error wall_clock_khz(int* khz, int device) { return hip().hipDeviceGetAttribute(khz, hipDeviceAttributeWallClockRate, device); }
};

} // namespace
} // namespace backend

#include "backends/gpu/backend_core.hpp"

namespace backend {

using RocmBackend = GpuBackend<HipApi>;

BackendPtr make_rocm_backend(int device) { return std::make_shared<RocmBackend>(device); }

std::map<std::string, size_t> rocm_take_launches(Backend& backend) { return dynamic_cast<RocmBackend&>(backend).take_launches(); }
void rocm_route_only(Backend& backend, std::string build) { dynamic_cast<RocmBackend&>(backend).route_only(std::move(build)); }

DeviceProfile& rocm_profile(Backend& backend) { return dynamic_cast<RocmBackend&>(backend).profile(); }

std::unique_ptr<Collective> rocm_collective(const std::vector<Backend*>& members, size_t rows, size_t width, GpuSumForm form) {
    std::vector<RocmBackend*> group;
    for (Backend* b : members) group.push_back(&dynamic_cast<RocmBackend&>(*b));
    return std::make_unique<GpuCollective<HipApi>>(group, rows, width, true, false, form);
}
std::unique_ptr<Collective> rocm_collective(const std::vector<Backend*>& members, size_t rows, size_t width, const std::map<size_t, GpuSumForm>& forms) {
    std::vector<RocmBackend*> group;
    for (Backend* b : members) group.push_back(&dynamic_cast<RocmBackend&>(*b));
    auto c = std::make_unique<GpuCollective<HipApi>>(group, rows, width, true);
    c->use(forms);
    return c;
}

} // namespace backend
