#pragma once

// Build configuration: the single place build-time knobs live.
// CMake regenerates it into the build directory; the checked-in default serves the plain build.bat path.
// Only backends are compile-time, since GPU backends pull in heavy SDKs; architectures and split mode are chosen at run time.

#define LLMX_RELEASE_VERSION "0.1.0"
#if __has_include("llmx-build-info.hpp")
#include "llmx-build-info.hpp"
#endif
#ifndef LLMX_BUILD_REVISION
#define LLMX_BUILD_REVISION "unknown"
#endif
#define LLMX_VERSION_STRING LLMX_RELEASE_VERSION "+" LLMX_BUILD_REVISION

// Backends compiled in besides the CPU, which is always built: the Vulkan backend is opt-in and needs its SDK at build time.
#define LLMX_HAS_BACKEND_VULKAN 0
