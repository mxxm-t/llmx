#pragma once

// llmx build configuration. This header is the single place build-time knobs
// live. CMake regenerates it via configure_file into the build directory (and
// adds that dir to the include path); the checked-in default here is what the
// plain build.bat path uses. Keep this minimal on purpose:
//
//   - Backends: the only things that NEED compile-time config, because GPU
//     backends pull in heavyweight SDKs (ROCm/CUDA/Vulkan). CPU is always on
//     (no external deps).
//   - Model architectures: NOT here. All archs are compiled in and selected at
//     runtime from the model file's metadata.
//   - Split mode / node count: NOT here. They are runtime parameters (chosen at
//     launch), not compile-time.

#define LLMX_VERSION_MAJOR 0
#define LLMX_VERSION_MINOR 1
#define LLMX_VERSION_PATCH 0
#define LLMX_VERSION_STRING "0.1.0"

// Backends compiled in. CPU is mandatory and has no SDK dependency. GPU
// backends are opt-in and require their SDK at build time.
#define LLMX_HAS_BACKEND_CPU    1
#define LLMX_HAS_BACKEND_ROCM   0
#define LLMX_HAS_BACKEND_CUDA   0
#define LLMX_HAS_BACKEND_VULKAN 0

// Default worker-thread hint for CPU backends. 0 = auto (hardware_concurrency).
#define LLMX_DEFAULT_THREADS 0
