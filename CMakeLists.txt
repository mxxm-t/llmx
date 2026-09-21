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
# with the cache options baked in. Options are ON/OFF; the header wants 0/1.
set(LLMX_CPU 1)
foreach(gpu ROCM CUDA SYCL VULKAN)
    if(LLMX_HAS_BACKEND_${gpu})
        set(LLMX_BACKEND_${gpu}_FLAG 1)
    else()
        set(LLMX_BACKEND_${gpu}_FLAG 0)
    endif()
endforeach()
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

# The Vulkan backend (docs/VULKAN.md). The SDK supplies the headers at build
# time; the loader is loaded at run time, so nothing links against vulkan-1.
# One static library so the CLI and the test compile the device code once.
if(LLMX_HAS_BACKEND_VULKAN)
    find_package(Vulkan REQUIRED COMPONENTS glslc)
    # Shaders are GLSL in the tree, compiled to SPIR-V at build time and
    # embedded as uint32_t arrays, so the binary carries its kernels.
    set(LLMX_VK_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/vulkan/shaders)
    set(LLMX_VK_SHADERS add silu_mul gather_rows rms_norm_rows norm_rope_rows embed matmul_row
        kv_write attention attention_merge matmul_tile norm_rope_kv attention_tile)
    # A variant is the same source compiled with a define, "name=source=DEFINE":
    # the row kernel is one module per family of types because the register
    # demand of every family's branches in one module cost the Q8_0 path 40
    # percent of its decode throughput.
    set(LLMX_VK_VARIANTS matmul_row_q4=matmul_row=LLMX_Q4 matmul_row_k4=matmul_row=LLMX_K4 matmul_row_k5=matmul_row=LLMX_K5
        matmul_row_k=matmul_row=LLMX_KQUANT
        kv_write_k16=kv_write=LLMX_K16 kv_write_v16=kv_write=LLMX_V16 kv_write_kv16=kv_write=LLMX_K16+LLMX_V16
        attention_k16=attention=LLMX_K16 attention_v16=attention=LLMX_V16 attention_kv16=attention=LLMX_K16+LLMX_V16
        attention_tile_k16=attention_tile=LLMX_K16 attention_tile_v16=attention_tile=LLMX_V16
        attention_tile_kv16=attention_tile=LLMX_K16+LLMX_V16
        norm_rope_kv_k16=norm_rope_kv=LLMX_K16 norm_rope_kv_v16=norm_rope_kv=LLMX_V16
        norm_rope_kv_kv16=norm_rope_kv=LLMX_K16+LLMX_V16)
    set(LLMX_VK_SPV)
    foreach(entry IN LISTS LLMX_VK_SHADERS LLMX_VK_VARIANTS)
        string(REPLACE "=" ";" parts ${entry})
        list(GET parts 0 shader)
        set(source ${shader})
        set(define)
        list(LENGTH parts nparts)
        if(nparts EQUAL 3)
            list(GET parts 1 source)
            list(GET parts 2 define)
            string(REPLACE "+" ";-D" define "-D${define}")
        endif()
        set(spv ${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan/${shader}.inc)
        add_custom_command(OUTPUT ${spv}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan
            COMMAND ${Vulkan_GLSLC_EXECUTABLE} -fshader-stage=compute --target-env=vulkan1.2 -O
                    -mfmt=num -I ${LLMX_VK_SHADER_DIR} ${define} -o ${spv} ${LLMX_VK_SHADER_DIR}/${source}.comp
            DEPENDS ${LLMX_VK_SHADER_DIR}/${source}.comp ${LLMX_VK_SHADER_DIR}/q.glsl
                    ${LLMX_VK_SHADER_DIR}/qdecode.glsl ${LLMX_VK_SHADER_DIR}/f16.glsl
            COMMENT "glslc ${shader}"
            VERBATIM)
        list(APPEND LLMX_VK_SPV ${spv})
    endforeach()
    add_custom_target(llmx-vulkan-shaders DEPENDS ${LLMX_VK_SPV})
    add_library(llmx-vulkan STATIC src/backends/vulkan/vulkan_backend.cpp)
    add_dependencies(llmx-vulkan llmx-vulkan-shaders)
    target_include_directories(llmx-vulkan
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR}/generated
        PRIVATE ${Vulkan_INCLUDE_DIRS})
    target_link_libraries(llmx-vulkan PUBLIC ${CMAKE_DL_LIBS})
    target_link_libraries(llmx PRIVATE llmx-vulkan)
endif()

include(CTest)
set(LLMX_TARGETS llmx)
if(LLMX_HAS_BACKEND_VULKAN)
    list(APPEND LLMX_TARGETS llmx-vulkan)
endif()
if(BUILD_TESTING)
    add_executable(llmx-hub-manifest-test tests/hub_manifest.cpp)
    target_include_directories(llmx-hub-manifest-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-hub-manifest-test)
    add_test(NAME hub-manifest COMMAND llmx-hub-manifest-test)
    set_tests_properties(hub-manifest PROPERTIES TIMEOUT 30)
    add_executable(llmx-hub-pull-test tests/hub_pull.cpp)
    target_include_directories(llmx-hub-pull-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-hub-pull-test)
    add_test(NAME hub-pull COMMAND llmx-hub-pull-test ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(hub-pull PROPERTIES TIMEOUT 60)
    add_executable(llmx-hub-transport-test tests/hub_transport.cpp)
    target_include_directories(llmx-hub-transport-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-hub-transport-test)
    add_test(NAME hub-transport COMMAND llmx-hub-transport-test)
    set_tests_properties(hub-transport PROPERTIES TIMEOUT 60)
    add_executable(llmx-gguf-shards-test tests/gguf_shards.cpp)
    target_include_directories(llmx-gguf-shards-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-gguf-shards-test)
    add_test(NAME gguf-shards COMMAND llmx-gguf-shards-test ${CMAKE_CURRENT_BINARY_DIR}/gguf-shards-fixtures)
    set_tests_properties(gguf-shards PROPERTIES TIMEOUT 30)
    add_executable(llmx-model-validation-test tests/model_validation.cpp)
    target_include_directories(llmx-model-validation-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-model-validation-test)
    add_test(NAME model-validation COMMAND llmx-model-validation-test)
    set_tests_properties(model-validation PROPERTIES TIMEOUT 30)
    add_executable(llmx-gguf-validation-test tests/gguf_validation.cpp)
    target_include_directories(llmx-gguf-validation-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-gguf-validation-test)
    add_test(NAME gguf-validation COMMAND llmx-gguf-validation-test ${CMAKE_CURRENT_BINARY_DIR}/gguf-validation-fixture.gguf)
    set_tests_properties(gguf-validation PROPERTIES TIMEOUT 30)
    add_executable(llmx-json-test tests/json.cpp)
    target_include_directories(llmx-json-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-json-test)
    add_test(NAME json COMMAND llmx-json-test)
    set_tests_properties(json PROPERTIES TIMEOUT 30)
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
    if(NOT MSVC AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # Make scalar-tail rounding deterministic for the exact ordered-FMA oracle.
        target_compile_options(llmx-backend-group-test PRIVATE -ffp-contract=off)
    endif()
    list(APPEND LLMX_TARGETS llmx-backend-group-test)
    add_test(NAME backend-group COMMAND llmx-backend-group-test)
    set_tests_properties(backend-group PROPERTIES TIMEOUT 60)
    add_executable(llmx-fused-dot-overflow-test tests/fused_dot_overflow.cpp)
    target_include_directories(llmx-fused-dot-overflow-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-fused-dot-overflow-test)
    add_test(NAME fused-dot-overflow COMMAND llmx-fused-dot-overflow-test)
    set_tests_properties(fused-dot-overflow PROPERTIES TIMEOUT 30)
    add_executable(llmx-backend-errors-test tests/backend_errors.cpp)
    target_include_directories(llmx-backend-errors-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-backend-errors-test)
    add_test(NAME backend-errors COMMAND llmx-backend-errors-test)
    set_tests_properties(backend-errors PROPERTIES TIMEOUT 30)
    add_executable(llmx-prefill-scope-test tests/prefill_scope.cpp)
    target_include_directories(llmx-prefill-scope-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-prefill-scope-test)
    add_test(NAME prefill-scope COMMAND llmx-prefill-scope-test)
    set_tests_properties(prefill-scope PROPERTIES TIMEOUT 60)
    if(WIN32)
        add_executable(llmx-prefill-placement-test tests/prefill_placement.cpp)
        target_include_directories(llmx-prefill-placement-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        list(APPEND LLMX_TARGETS llmx-prefill-placement-test)
        add_test(NAME prefill-placement COMMAND llmx-prefill-placement-test)
        set_tests_properties(prefill-placement PROPERTIES TIMEOUT 60)
    endif()
    add_executable(llmx-chat-template-test tests/chat_template.cpp)
    target_include_directories(llmx-chat-template-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-chat-template-test)
    add_test(NAME chat-template COMMAND llmx-chat-template-test
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/baseline_chat_template.json)
    set_tests_properties(chat-template PROPERTIES TIMEOUT 30)
    add_executable(llmx-fp16-test tests/fp16.cpp)
    target_include_directories(llmx-fp16-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-fp16-test)
    add_test(NAME fp16 COMMAND llmx-fp16-test)
    set_tests_properties(fp16 PROPERTIES TIMEOUT 60)

    add_executable(llmx-kv-cache-test tests/kv_cache.cpp)
    target_include_directories(llmx-kv-cache-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-kv-cache-test)
    add_test(NAME kv-cache COMMAND llmx-kv-cache-test)
    set_tests_properties(kv-cache PROPERTIES TIMEOUT 30)

    add_executable(llmx-placement-test tests/placement.cpp)
    target_include_directories(llmx-placement-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-placement-test)
    add_test(NAME placement COMMAND llmx-placement-test)
    set_tests_properties(placement PROPERTIES TIMEOUT 30)

    if(LLMX_HAS_BACKEND_VULKAN)
        # Exits 77 when there is no loader or no device, which CTest reports
        # as skipped rather than passed.
        add_executable(llmx-backend-vulkan-test tests/backend_vulkan.cpp)
        target_link_libraries(llmx-backend-vulkan-test PRIVATE llmx-vulkan)
        list(APPEND LLMX_TARGETS llmx-backend-vulkan-test)
        add_test(NAME backend-vulkan COMMAND llmx-backend-vulkan-test)
        set_tests_properties(backend-vulkan PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endif()
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
