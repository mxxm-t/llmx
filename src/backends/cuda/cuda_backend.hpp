#pragma once
#include "backends/backend.hpp"

#include <map>
#include <string>

// The CUDA backend (docs/CUDA.md), built only with LLMX_HAS_BACKEND_CUDA on Linux, in cuda_backend.cpp, a translation unit of its own that instantiates the shared GPU core (gpu/backend_core.hpp) over the CUDA driver.
// The toolkit supplies the header and the kernel compiler at build time; the driver is opened at run time, so the same binary runs without it.

namespace backend {

// The backend over CUDA device `device`, the driver's ordinal; it throws CudaUnavailable (cuda_runtime.hpp) when there is nothing usable to open.
BackendPtr make_cuda_backend(int device);

// The kernels a CUDA backend launched since the last reading, by name, and how often: the witness of which kernel ran.
std::map<std::string, size_t> cuda_take_launches(Backend& backend);

} // namespace backend
