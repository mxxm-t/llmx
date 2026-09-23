#pragma once

// Build configuration: the single place build-time knobs live.
// CMake regenerates it into the build directory; the checked-in default serves the plain build.bat path.
// Only backends are compile-time, since GPU backends pull in heavy SDKs; architectures and split mode are chosen at run time.

#define LLMX_VERSION_MAJOR 0
#define LLMX_VERSION_MINOR 1
#define LLMX_VERSION_PATCH 0
#define LLMX_RELEASE_VERSION "0.1.0"
#if __has_include("llmx-build-info.hpp")
#include "llmx-build-info.hpp"
#endif
#ifndef LLMX_BUILD_REVISION
#define LLMX_BUILD_REVISION "unknown"
#endif
#define LLMX_VERSION_STRING LLMX_RELEASE_VERSION "+" LLMX_BUILD_REVISION

// Backends compiled in.
// CPU is mandatory and has no SDK dependency.
// GPU backends are opt-in and require their SDK at build time.
#define LLMX_HAS_BACKEND_CPU    1
#define LLMX_HAS_BACKEND_ROCM   0
#define LLMX_HAS_BACKEND_CUDA   0
#define LLMX_HAS_BACKEND_SYCL   0
#define LLMX_HAS_BACKEND_VULKAN 0

// Default worker-thread hint for CPU backends. 0 = auto (hardware_concurrency).
#define LLMX_DEFAULT_THREADS 0
