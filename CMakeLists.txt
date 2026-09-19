cmake_minimum_required(VERSION 3.16)
project(llmx VERSION 0.1.0 LANGUAGES CXX)

# llmx - ground-up GGUF Q8_0 CLI. Cross-platform build (Windows/Linux/macOS).
# CPU is always built; GPU backends are opt-in via LLMX_HAS_BACKEND_* and require
# their SDK. Split mode and node count are runtime parameters, NOT build options.

option(LLMX_HAS_BACKEND_ROCM   "Build the ROCm (HIP) backend"   OFF)
option(LLMX_HAS_BACKEND_CUDA   "Build the CUDA backend"         OFF)
option(LLMX_HAS_BACKEND_SYCL   "Build the SYCL (Intel oneAPI) backend" OFF)
option(LLMX_HAS_BACKEND_VULKAN "Build the Vulkan backend"       OFF)
set(LLMX_DEFAULT_THREADS 0 CACHE STRING "Default worker threads for CPU (0 = auto)")

add_executable(llmx
    src/cli/main.cpp
)

target_include_directories(llmx PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_include_directories(llmx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Generate src/config.hpp from src/config.hpp.in-style template. The committed
# src/config.hpp is the fallback for plain build.bat; CMake produces a fresh one
# with the cache options baked in.
set(LLMX_CPU 1)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/llmx-config.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/config.hpp
    @ONLY
)

# The current x86 build requires AVX2/FMA/F16C; per-kernel runtime checks do not make the binary portable to older CPUs.
if(MSVC)
    target_compile_options(llmx PRIVATE /std:c++17 /O2 /arch:AVX2)
else()
    target_compile_options(llmx PRIVATE -std=c++17 -O3)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        target_compile_options(llmx PRIVATE -mavx2 -mfma -mf16c)
    endif()
endif()

if(MSVC)
    target_compile_options(llmx PRIVATE /W4)
else()
    target_compile_options(llmx PRIVATE -Wall -Wextra)
endif()

find_package(Threads REQUIRED)
target_link_libraries(llmx PRIVATE Threads::Threads)
