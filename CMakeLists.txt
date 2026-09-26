cmake_minimum_required(VERSION 3.16)
project(llmx VERSION 0.1.0 LANGUAGES CXX)

# llmx: a dependency-free GGUF runtime, built on Windows, Linux and macOS.
# The CPU backend is always built; the Vulkan backend is opt-in through LLMX_HAS_BACKEND_VULKAN and needs its SDK.
# Split mode and node count are runtime parameters, not build options.

option(LLMX_HAS_BACKEND_VULKAN "Build the Vulkan backend" OFF)

add_executable(llmx
    src/cli/main.cpp
)

target_include_directories(llmx PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_include_directories(llmx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Generate src/config.hpp with the cache options baked in, as 0/1; the committed src/config.hpp is the fallback for plain build.bat.
if(LLMX_HAS_BACKEND_VULKAN)
    set(LLMX_BACKEND_VULKAN_FLAG 1)
else()
    set(LLMX_BACKEND_VULKAN_FLAG 0)
endif()
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

# The Vulkan backend (docs/VULKAN.md): the SDK supplies the headers at build time and the loader is opened at run time, so nothing links against vulkan-1.
# One static library, so the CLI and the tests compile the device code once.
if(LLMX_HAS_BACKEND_VULKAN)
    # A build needs only the headers and the shader compiler; FindVulkan would insist on the library too.
    find_path(LLMX_VULKAN_INCLUDE_DIR vulkan/vulkan.h HINTS $ENV{VULKAN_SDK}/Include $ENV{VULKAN_SDK}/include)
    find_program(LLMX_GLSLC glslc HINTS $ENV{VULKAN_SDK}/Bin $ENV{VULKAN_SDK}/bin)
    if(NOT LLMX_VULKAN_INCLUDE_DIR OR NOT LLMX_GLSLC)
        message(FATAL_ERROR "LLMX_HAS_BACKEND_VULKAN needs vulkan/vulkan.h and glslc: install the Vulkan headers and shaderc, or set VULKAN_SDK")
    endif()
    # Shaders are GLSL in the tree, compiled to SPIR-V at build time and embedded as uint32_t arrays, so the binary carries its kernels.
    set(LLMX_VK_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/vulkan/shaders)
    set(LLMX_VK_SHADERS add silu_mul gather_rows rms_norm_rows norm_rope_rows embed matmul_row
        kv_write attention attention_merge matmul_tile norm_rope_kv attention_tile quantize_x
        quantize_x8 matmul_tile_q matmul_reduce matmul_vec_q8 moe_route moe_combine moe_group attention_vec)
    # A variant is the same source compiled with a define, "name=source=DEFINE"; the row kernel is one module per family of types, since one module for all of them raised every path's registers.
    set(LLMX_VK_VARIANTS matmul_row_q8w=matmul_row=LLMX_Q8W matmul_row_q4=matmul_row=LLMX_Q4 matmul_row_k4=matmul_row=LLMX_K4 matmul_row_k5=matmul_row=LLMX_K5
        matmul_row_k=matmul_row=LLMX_KQUANT
        matmul_row_q4_dot=matmul_row=LLMX_Q4+LLMX_DOT
        matmul_row_q4_dot8=matmul_row=LLMX_Q4+LLMX_DOT+LLMX_X8
        matmul_row_k4_dot=matmul_row=LLMX_K4+LLMX_DOT+LLMX_X8
        matmul_row_k5_dot=matmul_row=LLMX_K5+LLMX_DOT+LLMX_X8
        matmul_row_k_dot=matmul_row=LLMX_KQUANT+LLMX_DOT
        matmul_row_k_dot8=matmul_row=LLMX_KQUANT+LLMX_DOT+LLMX_X8
        kv_write_k16=kv_write=LLMX_K16 kv_write_v16=kv_write=LLMX_V16 kv_write_kv16=kv_write=LLMX_K16+LLMX_V16
        attention_k16=attention=LLMX_K16 attention_v16=attention=LLMX_V16 attention_kv16=attention=LLMX_K16+LLMX_V16
        attention_g4=attention=LLMX_G4 attention_k16_g4=attention=LLMX_K16+LLMX_G4 attention_v16_g4=attention=LLMX_V16+LLMX_G4
        attention_kv16_g4=attention=LLMX_K16+LLMX_V16+LLMX_G4
        attention_vec_k16=attention_vec=LLMX_K16 attention_vec_v16=attention_vec=LLMX_V16 attention_vec_kv16=attention_vec=LLMX_K16+LLMX_V16
        attention_vec_g4=attention_vec=LLMX_G4 attention_vec_k16_g4=attention_vec=LLMX_K16+LLMX_G4
        attention_vec_v16_g4=attention_vec=LLMX_V16+LLMX_G4 attention_vec_kv16_g4=attention_vec=LLMX_K16+LLMX_V16+LLMX_G4
        attention_tile_k16=attention_tile=LLMX_K16 attention_tile_v16=attention_tile=LLMX_V16
        attention_tile_kv16=attention_tile=LLMX_K16+LLMX_V16
        norm_rope_kv_k16=norm_rope_kv=LLMX_K16 norm_rope_kv_v16=norm_rope_kv=LLMX_V16
        norm_rope_kv_kv16=norm_rope_kv=LLMX_K16+LLMX_V16
        matmul_tile_q6=matmul_tile_q=LLMX_Q6 matmul_tile_q8=matmul_tile_q=LLMX_Q8)
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
            COMMAND ${LLMX_GLSLC} -fshader-stage=compute --target-env=vulkan1.2 -O
                    -mfmt=num -I ${LLMX_VK_SHADER_DIR} ${define} -o ${spv} ${LLMX_VK_SHADER_DIR}/${source}.comp
            DEPENDS ${LLMX_VK_SHADER_DIR}/${source}.comp ${LLMX_VK_SHADER_DIR}/q.glsl
                    ${LLMX_VK_SHADER_DIR}/qdecode.glsl ${LLMX_VK_SHADER_DIR}/f16.glsl ${LLMX_VK_SHADER_DIR}/xquant.glsl
                    ${LLMX_VK_SHADER_DIR}/views.glsl
            COMMENT "glslc ${shader}"
            VERBATIM)
        list(APPEND LLMX_VK_SPV ${spv})
    endforeach()
    add_custom_target(llmx-vulkan-shaders DEPENDS ${LLMX_VK_SPV})
    add_library(llmx-vulkan STATIC src/backends/vulkan/vulkan_backend.cpp)
    add_dependencies(llmx-vulkan llmx-vulkan-shaders)
    # The generated directory comes first, so every target linking the backend reads the build's config.hpp and not the checked-in fallback.
    target_include_directories(llmx-vulkan
        PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/generated ${CMAKE_CURRENT_SOURCE_DIR}/src
        PRIVATE ${LLMX_VULKAN_INCLUDE_DIR})
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
    # It builds hundreds of small models, which once started a CPU worker pool each and ran past 30 seconds on a host with no idle cores.
    set_tests_properties(model-validation PROPERTIES TIMEOUT 120)
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
    add_executable(llmx-file-reader-test tests/file_reader.cpp)
    target_include_directories(llmx-file-reader-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-file-reader-test)
    add_test(NAME file-reader COMMAND llmx-file-reader-test ${CMAKE_CURRENT_BINARY_DIR}/file-reader-fixtures)
    set_tests_properties(file-reader PROPERTIES TIMEOUT 60)
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

    add_executable(llmx-q8-dots-test tests/q8_dots.cpp)
    target_include_directories(llmx-q8-dots-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-q8-dots-test)
    add_test(NAME q8-dots COMMAND llmx-q8-dots-test)
    set_tests_properties(q8-dots PROPERTIES TIMEOUT 60)
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
    add_executable(llmx-sampler-test tests/sampler.cpp)
    target_include_directories(llmx-sampler-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-sampler-test)
    add_test(NAME sampler COMMAND llmx-sampler-test)
    set_tests_properties(sampler PROPERTIES TIMEOUT 30)

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

    add_executable(llmx-http-test tests/http.cpp)
    target_include_directories(llmx-http-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(WIN32)
        target_link_libraries(llmx-http-test PRIVATE ws2_32)
    endif()
    list(APPEND LLMX_TARGETS llmx-http-test)
    add_test(NAME http COMMAND llmx-http-test)
    set_tests_properties(http PROPERTIES TIMEOUT 30)
    add_executable(llmx-server-utf8-test tests/server_utf8.cpp)
    target_include_directories(llmx-server-utf8-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-server-utf8-test)
    add_test(NAME server-utf8 COMMAND llmx-server-utf8-test)
    set_tests_properties(server-utf8 PROPERTIES TIMEOUT 30)

    # Not a CTest: a model on one device against it split by layers over several, raw logits compared bit for bit; the suite's `split` component runs it over CPU backends, beside the executable.
    # It reads the build's config.hpp first, so it takes Vulkan devices exactly when the build has them.
    add_executable(llmx-split-check tools/split_check.cpp)
    target_include_directories(llmx-split-check PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(LLMX_HAS_BACKEND_VULKAN)
        target_link_libraries(llmx-split-check PRIVATE llmx-vulkan)
    endif()
    list(APPEND LLMX_TARGETS llmx-split-check)

    if(LLMX_HAS_BACKEND_VULKAN)
        # Exits 77 when there is no loader or no device, which CTest reports as skipped rather than passed.
        add_executable(llmx-backend-vulkan-test tests/backend_vulkan.cpp)
        target_link_libraries(llmx-backend-vulkan-test PRIVATE llmx-vulkan)
        list(APPEND LLMX_TARGETS llmx-backend-vulkan-test)
        add_test(NAME backend-vulkan COMMAND llmx-backend-vulkan-test)
        # It checks about 25 million kernel outputs against the CPU backend, which takes one to six minutes on a shared card.
        set_tests_properties(backend-vulkan PROPERTIES TIMEOUT 900 SKIP_RETURN_CODE 77)
        # Includes the private implementation to inject resource failures without runtime hooks.
        add_executable(llmx-vulkan-lifetime-test tests/vulkan_buffer.cpp)
        add_dependencies(llmx-vulkan-lifetime-test llmx-vulkan-shaders)
        target_include_directories(llmx-vulkan-lifetime-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_BINARY_DIR}/generated ${LLMX_VULKAN_INCLUDE_DIR})
        target_link_libraries(llmx-vulkan-lifetime-test PRIVATE ${CMAKE_DL_LIBS})
        list(APPEND LLMX_TARGETS llmx-vulkan-lifetime-test)
        add_test(NAME vulkan-buffer COMMAND llmx-vulkan-lifetime-test)
        add_test(NAME vulkan-lifetime COMMAND llmx-vulkan-lifetime-test --queue)
        set_tests_properties(vulkan-buffer vulkan-lifetime PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
        # Not a test: times the routed expert projections against dense matmuls of the same bytes.
        add_executable(llmx-moe-kernel-bench tools/moe_kernel_bench.cpp)
        target_link_libraries(llmx-moe-kernel-bench PRIVATE llmx-vulkan)
        list(APPEND LLMX_TARGETS llmx-moe-kernel-bench)
        # Not a test: the device queries and handoff timings of phase 0 in docs/MULTI-DEVICE.md, on the loader alone.
        add_executable(llmx-vk-handoff tools/vulkan_handoff.cpp)
        target_include_directories(llmx-vk-handoff PRIVATE ${LLMX_VULKAN_INCLUDE_DIR})
        target_link_libraries(llmx-vk-handoff PRIVATE ${CMAKE_DL_LIBS})
        list(APPEND LLMX_TARGETS llmx-vk-handoff)
        # Not a test: phase 0's concurrency, pipeline, group sum and expert exchange measurements through the backend.
        add_executable(llmx-multi-device-bench tools/multi_device_bench.cpp)
        target_link_libraries(llmx-multi-device-bench PRIVATE llmx-vulkan)
        list(APPEND LLMX_TARGETS llmx-multi-device-bench)
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
