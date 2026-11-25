#pragma once
#include "backends/backend.hpp"
#include "backends/device_profile.hpp"
#include "backends/gpu/gpu_backend.hpp"

#include <chrono>
#include <map>
#include <memory>
#include <string>

// The ROCm backend (docs/ROCM.md), built only with LLMX_HAS_BACKEND_ROCM on Linux, in rocm_backend.cpp, a translation unit of its own that instantiates the shared GPU core (gpu/backend_core.hpp) over HIP.
// The SDK supplies the headers and the kernel compiler at build time; the HIP runtime is opened at run time, so the same binary runs without ROCm.

namespace backend {

// The backend over HIP device `device`, as the runtime numbers them; it throws RocmUnavailable (rocm_runtime.hpp) when there is nothing usable to open.
// With `diagnostics`, its kernels, copies and fills are timed for `Backend::kernel_times`.
BackendPtr make_rocm_backend(int device, bool diagnostics = false);

// The kernels a ROCm backend launched since the last reading, by name, and how often: the witness of which build, tile or exact kernel ran.
std::map<std::string, size_t> rocm_take_launches(Backend& backend);

// A ROCm backend's routed tile calls launch only the builds whose kernel name holds `build`, every build again once it is empty: the witness of which build computes a group.
void rocm_route_only(Backend& backend, std::string build);
// A diagnostics ROCm backend sleeps `delay` between each timed op's start event and its launch: the tests' witness that a host falling behind its device shows as idle, not as op time.
void rocm_delay_launches(Backend& backend, std::chrono::microseconds delay);

// A collective over `members` whose fast sums all take `form` over `transport`, the members possibly backends of one device, never probed: the tests' witness of each form and transport.
std::unique_ptr<Collective> rocm_collective(const std::vector<Backend*>& members, size_t rows, size_t width, GpuSumForm form, GroupTransport transport = GroupTransport::peer_writes);
// The same with each fast sum's transport and form by its size, `forms` from each size up as the join's timing keeps them, staging every transport they name: the tests' witness of the picks following one another.
std::unique_ptr<Collective> rocm_collective(const std::vector<Backend*>& members, size_t rows, size_t width, const std::map<size_t, GpuSumPick>& forms);

// The profile a ROCm backend shapes its kernels by, which kv-selection reads for the crossovers it checks and changes to run each selection form at one size.
DeviceProfile& rocm_profile(Backend& backend);

// The capabilities a ROCm backend's rules read at each call, which sparse_tile_pair changes to take each few-row build at another unit count.
DeviceCaps& rocm_caps(Backend& backend);

} // namespace backend
