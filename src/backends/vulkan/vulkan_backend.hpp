#pragma once
#include <stdexcept>
#include <string>
#include "backends/device_profile.hpp"
#include <utility>
#include <vector>
#include "backends/backend.hpp"

// The Vulkan backend (docs/VULKAN.md), built only with LLMX_HAS_BACKEND_VULKAN, in vulkan_backend.cpp, the one translation unit outside the header-only runtime.
// The SDK supplies the headers and shader compiler at build time; the loader (vulkan-1.dll or libvulkan.so.1) is loaded at run time, so the same binary runs without Vulkan.

namespace backend {

// No loader, no device, no device at the requested index, or a device that lacks a feature the kernels need, such as a software device with narrow subgroups.
// A test skips on it; the CLI reports it.
struct VulkanUnavailable : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The backend over physical device `device`, counted as the loader lists them; it throws VulkanUnavailable when there is nothing usable to open.
// With `diagnostics`, the driver's internal representations of the kernels are captured for `vulkan_kernel_representations`, and on a queue that timestamps the dispatches are timed for `vulkan_kernel_times`.
BackendPtr make_vulkan_backend(int device, bool diagnostics = false);

// The ids one workgroup reduces of a row whose argmax is taken in slices (Backend::argmax_rows); a row of at most this many ids is reduced whole.
inline constexpr size_t kVulkanArgmaxSlice = 4096;

// The device's name as the driver reports it, for the CLI and the tests.
std::string vulkan_device_name(const Backend& backend);

// What this backend measured or was told about its device (backends/device_profile.hpp).
// The test reads it so that it predicts the same kernel the backend will pick, rather than assuming the numbers a particular device happens to want.
DeviceProfile vulkan_device_profile(const Backend& backend);

// Has a test's backend take the integer-dot kernels, as a profile preferring them would, on a device whose profile does not; false where the device lacks the integer dot.
// It is called before the backend's first matmul, so the test reaches on any such device the kernels an integer-dot profile picks.
bool vulkan_prefer_integer_dot(Backend& backend);

// Whether the device may flush a subnormal F32 result of a matrix module to zero, having no guaranteed 32-bit denormal preservation; false for another backend.
bool vulkan_may_flush_float32(const Backend& backend);

// Whether the row kernel the backend picks for `type` at `dtype` reads the activations as F32 rather than their rounded twin, so a test feeds its reference what that row reads.
bool vulkan_row_reads_float(const Backend& backend, uint32_t type, Dtype dtype);

// The driver's statistics for every kernel the backend has compiled, one line each (registers, scratch, occupancy on AMD), for the test's report.
// Empty when the device does not report them.
std::string vulkan_kernel_statistics(const Backend& backend);

// The driver's internal representations (the ISA on AMD) of every kernel a diagnostics backend has compiled, as (kernel name, text), empty for a backend opened without diagnostics or a device that does not serve them.
// Q8_0 decode builds start with `; q8_decode_build cols=C rows=R steps=S tree=T half=H halves=V`;
// two-row row builds start with `; row_build cols=C rows=R tree=T repair=P` (docs/VULKAN.md).
std::vector<std::pair<std::string, std::string>> vulkan_kernel_representations(const Backend& backend);

// Device time per kernel in milliseconds since the last call, for a diagnostics backend on a queue that timestamps, the output head's dispatches (Backend::matmul_logits) under their kernel's name with " head" after it.
// Reading it waits for the queue, so it is a diagnostic and not something a pass does.
std::vector<std::pair<std::string, double>> vulkan_kernel_times(Backend& backend);

// Each kernel's bindings, true for one it writes, as the backend derives them from the kernel's SPIR-V to drop the twins a dispatch overwrites; the test holds them to the shaders' qualifiers.
std::vector<std::pair<std::string, std::vector<bool>>> vulkan_kernel_writes();

// How many dispatches the last reading covered. The query pool bounds it, so a long interval is sampled from its first dispatches.
size_t vulkan_timed_dispatches(const Backend& backend);

// The last reading's device time between each timed op's end and the next one's start, copies included, in milliseconds, 0 where they overlap.
std::vector<double> vulkan_timed_gaps(const Backend& backend);

} // namespace backend
