#pragma once
#include "backends/backend.hpp"
#include "backends/device_profile.hpp"

#include <map>
#include <string>

// The ROCm backend (docs/ROCM.md), built only with LLMX_HAS_BACKEND_ROCM on Linux, in rocm_backend.cpp, a translation unit of its own as the Vulkan backend's is.
// The SDK supplies the headers and the kernel compiler at build time; the HIP runtime is opened at run time, so the same binary runs without ROCm.

namespace backend {

// The backend over HIP device `device`, as the runtime numbers them; it throws RocmUnavailable (rocm_runtime.hpp) when there is nothing usable to open.
BackendPtr make_rocm_backend(int device);

// The kernels a ROCm backend launched since the last reading, by name, and how often: the witness of which build, tile or exact kernel ran.
std::map<std::string, size_t> rocm_take_launches(Backend& backend);

// The profile a ROCm backend shapes its kernels by, which kv-selection reads for the crossovers it checks and changes to run each selection form at one size.
DeviceProfile& rocm_profile(Backend& backend);

} // namespace backend
