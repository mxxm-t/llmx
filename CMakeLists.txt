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

add_custom_target(llmx-build-info
    COMMAND ${CMAKE_COMMAND}
        "-DSOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}"
        "-DOUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/generated/llmx-build-info.hpp"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake"
    BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/generated/llmx-build-info.hpp"
    VERBATIM
)
add_dependencies(llmx llmx-build-info)

include(CTest)
set(LLMX_TARGETS llmx)
if(BUILD_TESTING)
    add_executable(llmx-cli-output-test tests/cli_output.cpp)
    target_include_directories(llmx-cli-output-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-cli-output-test)
    add_test(NAME cli-output COMMAND llmx-cli-output-test)
    set_tests_properties(cli-output PROPERTIES TIMEOUT 30)
    add_executable(llmx-load-progress-test tests/load_progress.cpp)
    target_include_directories(llmx-load-progress-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-load-progress-test)
    add_test(NAME load-progress COMMAND llmx-load-progress-test ${CMAKE_CURRENT_BINARY_DIR}/load-progress-fixture.gguf)
    set_tests_properties(load-progress PROPERTIES TIMEOUT 30)
    add_executable(llmx-generation-stream-test tests/generation_stream.cpp)
    target_include_directories(llmx-generation-stream-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-generation-stream-test)
    add_test(NAME generation-stream COMMAND llmx-generation-stream-test)
    set_tests_properties(generation-stream PROPERTIES TIMEOUT 30)
    add_executable(llmx-backend-group-test tests/backend_group.cpp)
    target_include_directories(llmx-backend-group-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-backend-group-test)
    add_test(NAME backend-group COMMAND llmx-backend-group-test)
    set_tests_properties(backend-group PROPERTIES TIMEOUT 60)
    add_executable(llmx-backend-errors-test tests/backend_errors.cpp)
    target_include_directories(llmx-backend-errors-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-backend-errors-test)
    add_test(NAME backend-errors COMMAND llmx-backend-errors-test)
    set_tests_properties(backend-errors PROPERTIES TIMEOUT 30)
    add_executable(llmx-chat-template-test tests/chat_template.cpp)
    target_include_directories(llmx-chat-template-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-chat-template-test)
    add_test(NAME chat-template COMMAND llmx-chat-template-test
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/baseline_chat_template.json)
    set_tests_properties(chat-template PROPERTIES TIMEOUT 30)
    add_executable(llmx-kv-cache-test tests/kv_cache.cpp)
    target_include_directories(llmx-kv-cache-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-kv-cache-test)
    add_test(NAME kv-cache COMMAND llmx-kv-cache-test)
    set_tests_properties(kv-cache PROPERTIES TIMEOUT 30)
endif()

find_package(Threads REQUIRED)
foreach(target IN LISTS LLMX_TARGETS)
    # The current x86 build requires AVX2/FMA/F16C; per-kernel runtime checks do not make the binary portable to older CPUs.
    if(MSVC)
        target_compile_options(${target} PRIVATE /std:c++17 /O2 /arch:AVX2)
    else()
        target_compile_options(${target} PRIVATE -std=c++17 -O3)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
            target_compile_options(${target} PRIVATE -mavx2 -mfma -mf16c)
        endif()
    endif()

    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra)
    endif()

    target_link_libraries(${target} PRIVATE Threads::Threads)
endforeach()
