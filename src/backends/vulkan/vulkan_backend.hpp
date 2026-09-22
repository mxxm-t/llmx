#pragma once
#include <stdexcept>
#include <string>
#include "backends/backend.hpp"

// The Vulkan backend (docs/VULKAN.md). Built only with LLMX_HAS_BACKEND_VULKAN;
// the SDK supplies the headers and the shader compiler at build time, while
// the loader, vulkan-1.dll or libvulkan.so.1, is loaded at run time so the
// same binary runs on a machine without Vulkan. The implementation lives in
// vulkan_backend.cpp, the one translation unit outside the header-only
// runtime, because it is device code with a large header behind it.
//
// Sub-step 1 of the page: instance, device and queue; buffers in device and
// host-visible memory; adopt, read, write and copy through a staging buffer;
// submit, wait and sync over a timeline semaphore. The compute ops arrive in
// the following sub-steps and throw until then, naming the sub-step.

namespace backend {

// No loader, no device, no device at the requested index, or a device
// that lacks a feature the kernels need, such as a software device with
// narrow subgroups. A test skips on it; the CLI reports it.
struct VulkanUnavailable : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The backend over physical device `device`, counted as the loader lists
// them. Throws VulkanUnavailable when there is nothing usable to open.
BackendPtr make_vulkan_backend(int device);

// The device's name as the driver reports it, for the CLI and the tests.
std::string vulkan_device_name(const Backend& backend);

} // namespace backend
