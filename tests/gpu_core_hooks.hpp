#pragma once
// The GPU core's test hooks by device name, for the tests that take --device: the kernels a ROCm or CUDA backend launched since it was last asked, and its device profile.
#include "backends/backend.hpp"
#include "backends/device_profile.hpp"
#include "config.hpp"
#if LLMX_HAS_BACKEND_ROCM
#include "backends/rocm/rocm_backend.hpp"
#endif
#if LLMX_HAS_BACKEND_CUDA
#include "backends/cuda/cuda_backend.hpp"
#endif

#include <cstddef>
#include <map>
#include <string>

namespace gpu_core_hooks {

inline bool is_rocm(const std::string& name) { return LLMX_HAS_BACKEND_ROCM && name.rfind("rocm", 0) == 0; }
inline bool is_cuda(const std::string& name) { return LLMX_HAS_BACKEND_CUDA && name.rfind("cuda", 0) == 0; }
// A device the GPU core runs, ROCm or CUDA, in this build.
inline bool on(const std::string& name) { return is_rocm(name) || is_cuda(name); }

// The kernels launched since the last reading on a GPU-core device, and nothing elsewhere.
inline std::map<std::string, size_t> launches(backend::Backend& device, const std::string& name) {
#if LLMX_HAS_BACKEND_ROCM
    if (is_rocm(name)) return backend::rocm_take_launches(device);
#endif
#if LLMX_HAS_BACKEND_CUDA
    if (is_cuda(name)) return backend::cuda_take_launches(device);
#endif
    (void)device;
    (void)name;
    return {};
}

#if LLMX_HAS_BACKEND_ROCM || LLMX_HAS_BACKEND_CUDA
// The device profile a GPU-core device shapes its kernels by; only called where on(name).
inline backend::DeviceProfile& profile(backend::Backend& device, const std::string& name) {
#if LLMX_HAS_BACKEND_ROCM
    if (is_rocm(name)) return backend::rocm_profile(device);
#endif
#if LLMX_HAS_BACKEND_CUDA
    return backend::cuda_profile(device);
#else
    (void)name;
    return backend::rocm_profile(device);
#endif
}
#endif

} // namespace gpu_core_hooks
