#pragma once
#include "backends/backend.hpp"
#include "backends/device_profile.hpp"
#include "backends/gpu/gpu_backend.hpp"

#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <vector>

// The CUDA backend (docs/CUDA.md), built only with LLMX_HAS_BACKEND_CUDA on Linux, in cuda_backend.cpp, a translation unit of its own that instantiates the shared GPU core (gpu/backend_core.hpp) over the CUDA driver.
// The toolkit supplies the header and the kernel compiler at build time; the driver is opened at run time, so the same binary runs without it.

namespace backend {

// The backend over CUDA device `device`, the driver's ordinal; it throws CudaUnavailable (cuda_runtime.hpp) when there is nothing usable to open.
// With `diagnostics`, its kernels, copies and fills are timed for `Backend::kernel_times`.
BackendPtr make_cuda_backend(int device, bool diagnostics = false);

// The kernels a CUDA backend launched since the last reading, by name, and how often: the witness of which kernel ran.
std::map<std::string, size_t> cuda_take_launches(Backend& backend);

// Limits the routed tile builds a CUDA backend's calls may take to those whose name holds `build`, or lifts the limit when it is empty: the routed builds' witness.
void cuda_route_only(Backend& backend, std::string build);
// A diagnostics CUDA backend sleeps `delay` between each timed op's start event and its launch: the tests' witness that a host falling behind its device shows as idle, not as op time.
void cuda_delay_launches(Backend& backend, std::chrono::microseconds delay);

// The device numbers a CUDA backend shapes its kernels by (backends/device_profile.hpp), which its witnesses read.
DeviceProfile& cuda_profile(Backend& backend);

// The capabilities a CUDA backend's rules read at each call, which sparse_tile_pair changes to take each few-row build at another unit count.
DeviceCaps& cuda_caps(Backend& backend);

// A collective over `members` whose fast sums all take `form` over `transport`, never probed: the tests' witness of each form and transport.
std::unique_ptr<Collective> cuda_collective(const std::vector<Backend*>& members, size_t rows, size_t width, GpuSumForm form, GroupTransport transport);
// The same with each fast sum's transport and form by its size, `forms` from each size up as the join's timing keeps them, staging every transport they name: the tests' witness of sums switching transport.
std::unique_ptr<Collective> cuda_collective(const std::vector<Backend*>& members, size_t rows, size_t width, const std::map<size_t, GpuSumPick>& forms);

} // namespace backend
