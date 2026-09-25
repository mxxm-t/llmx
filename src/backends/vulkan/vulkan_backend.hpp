#pragma once
#include <stdexcept>
#include <string>
#include "backends/device_profile.hpp"
#include <utility>
#include <vector>
#include "backends/backend.hpp"

// The Vulkan backend (docs/VULKAN.md), built only with LLMX_HAS_BACKEND_VULKAN.
// The SDK supplies the headers and shader compiler at build time; the loader (vulkan-1.dll or libvulkan.so.1) is loaded at run time, so the same binary runs without Vulkan.
// The implementation lives in vulkan_backend.cpp, the one translation unit outside the header-only runtime.

namespace backend {

// No loader, no device, no device at the requested index, or a device that lacks a feature the kernels need, such as a software device with narrow subgroups.
// A test skips on it; the CLI reports it.
struct VulkanUnavailable : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The backend over physical device `device`, counted as the loader lists them.
// Throws VulkanUnavailable when there is nothing usable to open.
// With `diagnostics`, the driver's internal representations of the kernels are captured for `vulkan_kernel_representations`, and on a queue that timestamps the dispatches are timed for `vulkan_kernel_times`.
BackendPtr make_vulkan_backend(int device, bool diagnostics = false);

// The device's name as the driver reports it, for the CLI and the tests.
std::string vulkan_device_name(const Backend& backend);

// What this backend measured or was told about its device (backends/device_profile.hpp).
// The test reads it so that it predicts the same kernel the backend will pick, rather than assuming the numbers a particular device happens to want.
DeviceProfile vulkan_device_profile(const Backend& backend);

// The driver's statistics for every kernel the backend has compiled, one line each (registers, scratch, occupancy on AMD), for the test's report.
// Empty when the device does not report them.
std::string vulkan_kernel_statistics(const Backend& backend);

// The driver's internal representations (the ISA on AMD) of every kernel a diagnostics backend has compiled, as (kernel name, text).
// Empty for a backend opened without diagnostics or a device that does not serve them.
std::vector<std::pair<std::string, std::string>> vulkan_kernel_representations(const Backend& backend);

// Device time per kernel in milliseconds since the last call, for a diagnostics backend on a queue that timestamps.
// Reading it waits for the queue, so it is a diagnostic and not something a pass does.
std::vector<std::pair<std::string, double>> vulkan_kernel_times(Backend& backend);

// How many dispatches the last reading covered. The query pool bounds it, so a long interval is sampled from its first dispatches.
size_t vulkan_timed_dispatches(const Backend& backend);

} // namespace backend
