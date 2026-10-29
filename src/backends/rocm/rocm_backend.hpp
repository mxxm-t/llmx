#pragma once
#include "backends/backend.hpp"

// The ROCm backend (docs/ROCM.md), built only with LLMX_HAS_BACKEND_ROCM on Linux, in rocm_backend.cpp, a translation unit of its own as the Vulkan backend's is.
// The SDK supplies the headers and the kernel compiler at build time; the HIP runtime is opened at run time, so the same binary runs without ROCm.

namespace backend {

// The backend over HIP device `device`, as the runtime numbers them; it throws RocmUnavailable (rocm_runtime.hpp) when there is nothing usable to open.
BackendPtr make_rocm_backend(int device);

} // namespace backend
