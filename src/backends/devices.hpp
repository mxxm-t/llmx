#pragma once
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include "config.hpp"
#include "core/list.hpp"
#include "backends/backend.hpp"
#include "backends/cpu/cpu_backend.hpp"
#if LLMX_HAS_BACKEND_VULKAN
#include "backends/vulkan/vulkan_backend.hpp"
#endif

// The backends a device spec names: `cpu` or `vulkan:N` (`vulkan` alone is `vulkan:0`), one or a comma-separated list, and the backend each names.
namespace backend {

// A device spec in its one spelling, so `vulkan`, `vulkan:0` and `vulkan:00` are the same device.
inline std::string canonical_device(const std::string& spec) {
    if (spec == "cpu") return spec;
    const size_t colon = spec.find(':');
    const std::string name = spec.substr(0, colon);
    if (name != "vulkan") throw std::runtime_error("--device: unknown backend '" + name + "' (cpu, vulkan:N)");
    unsigned long index = 0;
    if (colon != std::string::npos) {
        const std::string rest = spec.substr(colon + 1);
        if (rest.empty() || rest.size() > 6 || rest.find_first_not_of("0123456789") != std::string::npos)
            throw std::runtime_error("--device: invalid device index in '" + spec + "'");
        index = std::stoul(rest);
    }
    return name + ":" + std::to_string(index);
}

// A device list: one device, or several separated by commas for a model split by layers over them in that order, each in its canonical spelling.
// A device listed twice would have two stages driving one backend and its free memory counted twice, so it is refused.
inline std::vector<std::string> device_specs(const std::string& value) {
    std::vector<std::string> specs;
    for (const auto& entry : core::comma_list(value)) {
        if (entry.empty()) throw std::runtime_error("--device: an empty entry in '" + value + "'");
        const std::string spec = canonical_device(entry);
        if (std::find(specs.begin(), specs.end(), spec) != specs.end())
            throw std::runtime_error("--device: " + spec + " is listed twice");
        specs.push_back(spec);
    }
    return specs;
}

// The backend a spec names; `diagnostics` asks a device backend to time its kernels (bench --profile).
inline BackendPtr make_backend(const std::string& spec, bool diagnostics = false) {
    const std::string device = canonical_device(spec);
    if (device == "cpu") return make_cpu_backend();
#if LLMX_HAS_BACKEND_VULKAN
    return make_vulkan_backend(std::atoi(device.c_str() + device.find(':') + 1), diagnostics);
#else
    (void)diagnostics;
    throw std::runtime_error("--device vulkan: this build has no Vulkan backend (LLMX_HAS_BACKEND_VULKAN)");
#endif
}

// The backends of a device list, in its order.
inline std::vector<BackendPtr> make_backends(const std::vector<std::string>& specs, bool diagnostics = false) {
    std::vector<BackendPtr> backends;
    for (const auto& spec : specs) backends.push_back(make_backend(spec, diagnostics));
    return backends;
}

} // namespace backend
