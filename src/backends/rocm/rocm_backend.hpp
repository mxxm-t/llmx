#pragma once
#include "backends/backend.hpp"
#include "backends/device_profile.hpp"

#include <map>
#include <memory>
#include <string>

// The ROCm backend (docs/ROCM.md), built only with LLMX_HAS_BACKEND_ROCM on Linux, in rocm_backend.cpp, a translation unit of its own as the Vulkan backend's is.
// The SDK supplies the headers and the kernel compiler at build time; the HIP runtime is opened at run time, so the same binary runs without ROCm.

namespace backend {

// The backend over HIP device `device`, as the runtime numbers them; it throws RocmUnavailable (rocm_runtime.hpp) when there is nothing usable to open.
BackendPtr make_rocm_backend(int device);

// The kernels a ROCm backend launched since the last reading, by name, and how often: the witness of which build, tile or exact kernel ran.
std::map<std::string, size_t> rocm_take_launches(Backend& backend);

// A ROCm backend's routed tile calls launch only the builds whose kernel name holds `build`, every build again once it is empty: the witness of which build computes a group.
void rocm_route_only(Backend& backend, std::string build);

// The forms of a ROCm tensor group's fast sum (docs/src/backends-rocm.md, the collective), each adding every element in member order, so each gives the one-shot's bits.
enum class RocmSumForm { one_shot, two_shots, chain };

// A collective over `members` whose fast sums all take `form`, the members possibly backends of one device, never probed: the tests' witness of each form.
std::unique_ptr<Collective> rocm_collective(const std::vector<Backend*>& members, size_t rows, size_t width, RocmSumForm form);

// The profile a ROCm backend shapes its kernels by, which kv-selection reads for the crossovers it checks and changes to run each selection form at one size.
DeviceProfile& rocm_profile(Backend& backend);

} // namespace backend
