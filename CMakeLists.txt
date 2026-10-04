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

# Generate config.hpp into the build tree from cmake/llmx-config.hpp.in, with the release version and the Vulkan backend flag as 0/1; the committed src/config.hpp is the fallback for plain build.bat.
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
    set(LLMX_VK_SHADERS copy_mxfp4 add silu_mul gather_rows rms_norm_rows norm_rope_partial embed matmul_row
        kv_write attention attention_merge matmul_tile norm_rope_kv attention_tile quantize_x
        quantize_xw matmul_tile_q matmul_reduce matmul_vec_q8 moe_route moe_combine moe_group attention_vec
        sigmoid_mul gated_rms_norm causal_conv_silu delta_rule argmax_rows)
    # A variant is the same source compiled with a define, "name=source=DEFINE"; the row kernel is one module per family of types, since one module for all of them raised every path's registers.
    set(LLMX_VK_VARIANTS matmul_row_mxfp4=matmul_row=LLMX_MXFP4+LLMX_PRESERVE_F32
        matmul_row_mxfp4_dot=matmul_row=LLMX_MXFP4+LLMX_DOT+LLMX_PRESERVE_F32
        matmul_row_mxfp4_float_x=matmul_row=LLMX_MXFP4+LLMX_FLOAT_X+LLMX_PRESERVE_F32
        matmul_tile_q8mx=matmul_tile_q=LLMX_Q8+LLMX_Q8MX+LLMX_PRESERVE_F32
        embed_mxfp4=embed=LLMX_MXFP4 matmul_tile_mxfp4=matmul_tile=LLMX_MXFP4+LLMX_PRESERVE_F32
        matmul_tile_mxfp4_bf16=matmul_tile=LLMX_MXFP4+LLMX_PRESERVE_F32+LLMX_BF16 matmul_reduce_mxfp4=matmul_reduce=LLMX_MXFP4+LLMX_PRESERVE_F32
        matmul_row_q8w=matmul_row=LLMX_Q8W matmul_row_q4=matmul_row=LLMX_Q4 matmul_row_k4=matmul_row=LLMX_K4 matmul_row_k5=matmul_row=LLMX_K5
        matmul_row_float_x=matmul_row=LLMX_FLOAT_X
        matmul_row_float_x_preserve=matmul_row=LLMX_FLOAT_X+LLMX_PRESERVE_F32
        matmul_row_preserve=matmul_row=LLMX_PRESERVE_F32
        matmul_tile_preserve=matmul_tile=LLMX_PRESERVE_F32
        matmul_tile_bf16=matmul_tile=LLMX_BF16
        matmul_tile_bf16_preserve=matmul_tile=LLMX_BF16+LLMX_PRESERVE_F32
        matmul_tile_q_preserve=matmul_tile_q=LLMX_PRESERVE_F32
        matmul_tile_q6_preserve=matmul_tile_q=LLMX_Q6+LLMX_PRESERVE_F32
        matmul_reduce_preserve=matmul_reduce=LLMX_PRESERVE_F32
        matmul_row_q8w_preserve=matmul_row=LLMX_Q8W+LLMX_PRESERVE_F32
        matmul_vec_q8_preserve=matmul_vec_q8=LLMX_PRESERVE_F32
        matmul_row_q4_preserve=matmul_row=LLMX_Q4+LLMX_PRESERVE_F32
        matmul_row_q4_dot_preserve=matmul_row=LLMX_Q4+LLMX_DOT+LLMX_PRESERVE_F32
        matmul_row_k4_dot_preserve=matmul_row=LLMX_K4+LLMX_DOT+LLMX_PRESERVE_F32
        matmul_row_k5_preserve=matmul_row=LLMX_K5+LLMX_PRESERVE_F32
        matmul_row_k5_dot_preserve=matmul_row=LLMX_K5+LLMX_DOT+LLMX_PRESERVE_F32
        matmul_row_k_preserve=matmul_row=LLMX_KQUANT+LLMX_PRESERVE_F32
        matmul_row_k_dot_preserve=matmul_row=LLMX_KQUANT+LLMX_DOT+LLMX_PRESERVE_F32
        matmul_row_k=matmul_row=LLMX_KQUANT
        matmul_row_q4_dot=matmul_row=LLMX_Q4+LLMX_DOT
        matmul_row_k4_dot=matmul_row=LLMX_K4+LLMX_DOT
        matmul_row_k5_dot=matmul_row=LLMX_K5+LLMX_DOT
        matmul_row_k_dot=matmul_row=LLMX_KQUANT+LLMX_DOT
        kv_write_k16=kv_write=LLMX_K16 kv_write_v16=kv_write=LLMX_V16 kv_write_kv16=kv_write=LLMX_K16+LLMX_V16
        attention_k16=attention=LLMX_K16 attention_v16=attention=LLMX_V16 attention_kv16=attention=LLMX_K16+LLMX_V16
        attention_g4=attention=LLMX_G4 attention_k16_g4=attention=LLMX_K16+LLMX_G4 attention_v16_g4=attention=LLMX_V16+LLMX_G4
        attention_kv16_g4=attention=LLMX_K16+LLMX_V16+LLMX_G4
        attention_vec_k16=attention_vec=LLMX_K16 attention_vec_v16=attention_vec=LLMX_V16 attention_vec_kv16=attention_vec=LLMX_K16+LLMX_V16
        attention_vec_g4=attention_vec=LLMX_G4 attention_vec_k16_g4=attention_vec=LLMX_K16+LLMX_G4
        attention_vec_v16_g4=attention_vec=LLMX_V16+LLMX_G4 attention_vec_kv16_g4=attention_vec=LLMX_K16+LLMX_V16+LLMX_G4
        attention_tile_k16=attention_tile=LLMX_K16 attention_tile_v16=attention_tile=LLMX_V16
        attention_tile_kv16=attention_tile=LLMX_K16+LLMX_V16
        attention_tile_d256=attention_tile=LLMX_D256 attention_tile_d256_k16=attention_tile=LLMX_D256+LLMX_K16
        attention_tile_d256_v16=attention_tile=LLMX_D256+LLMX_V16 attention_tile_d256_kv16=attention_tile=LLMX_D256+LLMX_K16+LLMX_V16
        attention_vec_d256=attention_vec=LLMX_D256 attention_vec_d256_k16=attention_vec=LLMX_D256+LLMX_K16
        attention_vec_d256_v16=attention_vec=LLMX_D256+LLMX_V16 attention_vec_d256_kv16=attention_vec=LLMX_D256+LLMX_K16+LLMX_V16
        attention_vec_d256_g4=attention_vec=LLMX_D256+LLMX_G4 attention_vec_d256_k16_g4=attention_vec=LLMX_D256+LLMX_K16+LLMX_G4
        attention_vec_d256_v16_g4=attention_vec=LLMX_D256+LLMX_V16+LLMX_G4 attention_vec_d256_kv16_g4=attention_vec=LLMX_D256+LLMX_K16+LLMX_V16+LLMX_G4
        norm_rope_kv_k16=norm_rope_kv=LLMX_K16 norm_rope_kv_v16=norm_rope_kv=LLMX_V16
        norm_rope_kv_kv16=norm_rope_kv=LLMX_K16+LLMX_V16
        matmul_tile_q6=matmul_tile_q=LLMX_Q6 matmul_tile_q8=matmul_tile_q=LLMX_Q8
        embed_ids=embed=LLMX_DEVICE_IDS embed_ids_mxfp4=embed=LLMX_DEVICE_IDS+LLMX_MXFP4
        delta_rule_short=delta_rule=LLMX_SHORT)
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
                    ${LLMX_VK_SHADER_DIR}/views.glsl ${LLMX_VK_SHADER_DIR}/float_controls.glsl ${LLMX_VK_SHADER_DIR}/dot16.glsl
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
    add_executable(llmx-format-output-test tests/format_output.cpp)
    target_include_directories(llmx-format-output-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-format-output-test)
    add_test(NAME format-output COMMAND llmx-format-output-test ${CMAKE_CURRENT_BINARY_DIR}/format-output-test)
    set_tests_properties(format-output PROPERTIES TIMEOUT 30)
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
    add_test(NAME model-validation COMMAND llmx-model-validation-test
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/model_refusals.txt)
    # It builds hundreds of small models, so it gets a longer timeout than the other unit tests.
    set_tests_properties(model-validation PROPERTIES TIMEOUT 120)
    add_executable(llmx-gguf-validation-test tests/gguf_validation.cpp)
    target_include_directories(llmx-gguf-validation-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-gguf-validation-test)
    add_test(NAME gguf-validation COMMAND llmx-gguf-validation-test ${CMAKE_CURRENT_BINARY_DIR}/gguf-validation-fixture.gguf
        ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/vulkan/shaders/q.glsl)
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
        # Without contraction only a multiply-add written as an FMA fuses, so the ordered-FMA oracle fails a prompt dot whose tail leaves its FMAs to the compiler.
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

    add_executable(llmx-quantize-range-test tests/quantize_range.cpp)
    target_include_directories(llmx-quantize-range-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-quantize-range-test)
    add_test(NAME quantize-range COMMAND llmx-quantize-range-test)
    set_tests_properties(quantize-range PROPERTIES TIMEOUT 30)

    add_executable(llmx-q8-dots-test tests/q8_dots.cpp)
    target_include_directories(llmx-q8-dots-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-q8-dots-test)
    add_test(NAME q8-dots COMMAND llmx-q8-dots-test)
    set_tests_properties(q8-dots PROPERTIES TIMEOUT 60)
    add_executable(llmx-qwen35-ops-test tests/qwen35_ops.cpp)
    target_include_directories(llmx-qwen35-ops-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-qwen35-ops-test)
    add_test(NAME qwen35-ops COMMAND llmx-qwen35-ops-test)
    set_tests_properties(qwen35-ops PROPERTIES TIMEOUT 60)
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
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/baseline_chat_template.json
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/qwen35-0.8b/baseline_chat.json
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/qwen35-4b/baseline_chat.json)
    set_tests_properties(chat-template PROPERTIES TIMEOUT 30)
    add_executable(llmx-mxfp4-test tests/mxfp4.cpp)
    target_include_directories(llmx-mxfp4-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-mxfp4-test)
    add_test(NAME mxfp4 COMMAND llmx-mxfp4-test)
    set_tests_properties(mxfp4 PROPERTIES TIMEOUT 60)
    add_executable(llmx-dtype-test tests/dtype.cpp)
    target_include_directories(llmx-dtype-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-dtype-test)
    add_test(NAME dtype COMMAND llmx-dtype-test)

    add_executable(llmx-fp16-test tests/fp16.cpp)
    target_include_directories(llmx-fp16-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-fp16-test)
    add_test(NAME fp16 COMMAND llmx-fp16-test)
    set_tests_properties(fp16 PROPERTIES TIMEOUT 60)
    add_executable(llmx-cpus-test tests/cpus.cpp)
    target_include_directories(llmx-cpus-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-cpus-test)
    add_test(NAME cpus COMMAND llmx-cpus-test)
    set_tests_properties(cpus PROPERTIES TIMEOUT 30)
    add_executable(llmx-host-memory-test tests/host_memory.cpp)
    target_include_directories(llmx-host-memory-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-host-memory-test)
    add_test(NAME host-memory COMMAND llmx-host-memory-test)
    set_tests_properties(host-memory PROPERTIES TIMEOUT 30)
    add_executable(llmx-sampler-test tests/sampler.cpp)
    target_include_directories(llmx-sampler-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-sampler-test)
    add_test(NAME sampler COMMAND llmx-sampler-test)
    set_tests_properties(sampler PROPERTIES TIMEOUT 30)
    add_executable(llmx-logprobs-test tests/logprobs.cpp)
    target_include_directories(llmx-logprobs-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-logprobs-test)
    add_test(NAME logprobs COMMAND llmx-logprobs-test)
    set_tests_properties(logprobs PROPERTIES TIMEOUT 60)
    # Reads the build's config.hpp first, so a build with Vulkan also runs its device case on device 0 when there is one.
    add_executable(llmx-server-resume-test tests/server_resume.cpp)
    target_include_directories(llmx-server-resume-test PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(LLMX_HAS_BACKEND_VULKAN)
        target_link_libraries(llmx-server-resume-test PRIVATE llmx-vulkan)
    endif()
    list(APPEND LLMX_TARGETS llmx-server-resume-test)
    add_test(NAME server-resume COMMAND llmx-server-resume-test)
    set_tests_properties(server-resume PROPERTIES TIMEOUT 300)
    add_executable(llmx-server-spec-test tests/server_spec.cpp)
    target_include_directories(llmx-server-spec-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-server-spec-test)
    add_test(NAME server-spec COMMAND llmx-server-spec-test)
    set_tests_properties(server-spec PROPERTIES TIMEOUT 300)
    add_executable(llmx-server-passes-test tests/server_passes.cpp)
    target_include_directories(llmx-server-passes-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-server-passes-test)
    add_test(NAME server-passes COMMAND llmx-server-passes-test)
    set_tests_properties(server-passes PROPERTIES TIMEOUT 300)
    add_executable(llmx-sampling-pool-test tests/sampling_pool.cpp)
    target_include_directories(llmx-sampling-pool-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-sampling-pool-test)
    add_test(NAME sampling-pool COMMAND llmx-sampling-pool-test)
    set_tests_properties(sampling-pool PROPERTIES TIMEOUT 60)
    add_executable(llmx-server-passes-cpu-test tests/server_passes_cpu.cpp)
    target_include_directories(llmx-server-passes-cpu-test PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(LLMX_HAS_BACKEND_VULKAN)
        target_link_libraries(llmx-server-passes-cpu-test PRIVATE llmx-vulkan)
    endif()
    list(APPEND LLMX_TARGETS llmx-server-passes-cpu-test)
    add_test(NAME server-passes-cpu COMMAND llmx-server-passes-cpu-test)
    set_tests_properties(server-passes-cpu PROPERTIES TIMEOUT 600)

    add_executable(llmx-spec-test tests/spec.cpp)
    target_include_directories(llmx-spec-test PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(LLMX_HAS_BACKEND_VULKAN)
        target_link_libraries(llmx-spec-test PRIVATE llmx-vulkan)
    endif()
    list(APPEND LLMX_TARGETS llmx-spec-test)
    add_test(NAME spec COMMAND llmx-spec-test)
    set_tests_properties(spec PROPERTIES TIMEOUT 600)

    add_executable(llmx-kv-cache-test tests/kv_cache.cpp)
    target_include_directories(llmx-kv-cache-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-kv-cache-test)
    add_test(NAME kv-cache COMMAND llmx-kv-cache-test)
    set_tests_properties(kv-cache PROPERTIES TIMEOUT 30)

    add_executable(llmx-arch-qwen35-test tests/arch_qwen35.cpp)
    target_include_directories(llmx-arch-qwen35-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-arch-qwen35-test)
    add_test(NAME arch-qwen35 COMMAND llmx-arch-qwen35-test)
    set_tests_properties(arch-qwen35 PROPERTIES TIMEOUT 60)

    add_executable(llmx-placement-test tests/placement.cpp)
    target_include_directories(llmx-placement-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-placement-test)
    add_test(NAME placement COMMAND llmx-placement-test)
    set_tests_properties(placement PROPERTIES TIMEOUT 120)

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

    # The suite compares original CPU F32 activations against independent HF fixtures beside quantized decode.
    add_executable(llmx-cpu-f32-check tools/cpu_f32_check.cpp)
    target_include_directories(llmx-cpu-f32-check PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(LLMX_HAS_BACKEND_VULKAN)
        target_link_libraries(llmx-cpu-f32-check PRIVATE llmx-vulkan)
    endif()
    list(APPEND LLMX_TARGETS llmx-cpu-f32-check)

    # Not a CTest: a model on one device against it split by layers over several, raw logits compared bit for bit; the suite's `split` component runs it over CPU backends, beside the executable.
    # It reads the build's config.hpp first, so it takes Vulkan devices exactly when the build has them.
    add_executable(llmx-split-check tools/split_check.cpp)
    target_include_directories(llmx-split-check PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(LLMX_HAS_BACKEND_VULKAN)
        target_link_libraries(llmx-split-check PRIVATE llmx-vulkan)
    endif()
    list(APPEND LLMX_TARGETS llmx-split-check)
    # Not a CTest either: a fixture's forced reply read through the decode path of a real model, run by hand (AGENTS.md, Tests).
    add_executable(llmx-decode-probe tools/decode_probe.cpp)
    target_include_directories(llmx-decode-probe PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(LLMX_HAS_BACKEND_VULKAN)
        target_link_libraries(llmx-decode-probe PRIVATE llmx-vulkan)
    endif()
    list(APPEND LLMX_TARGETS llmx-decode-probe)
    # Not a CTest: a model's MTP blocks split out into a drafter file beside it or embedded back, bytes unchanged; the suite's `drafters` component runs it on tiny fixtures (docs/SPECULATIVE.md, step 6).
    add_executable(llmx-drafter-pack tools/drafter_pack.cpp)
    target_include_directories(llmx-drafter-pack PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated ${CMAKE_CURRENT_SOURCE_DIR}/src)
    list(APPEND LLMX_TARGETS llmx-drafter-pack)
    add_executable(llmx-model-logits tools/model_logits.cpp)
    add_dependencies(llmx-model-logits llmx-build-info)
    target_include_directories(llmx-model-logits PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(LLMX_HAS_BACKEND_VULKAN)
        target_link_libraries(llmx-model-logits PRIVATE llmx-vulkan)
    endif()
    list(APPEND LLMX_TARGETS llmx-model-logits)


    if(LLMX_HAS_BACKEND_VULKAN)
        # Exits 77 when there is no loader or no device, which CTest reports as skipped rather than passed.
        add_executable(llmx-mxfp4-vulkan-test tests/mxfp4_vulkan.cpp)
        target_link_libraries(llmx-mxfp4-vulkan-test PRIVATE llmx-vulkan)
        list(APPEND LLMX_TARGETS llmx-mxfp4-vulkan-test)
        add_test(NAME mxfp4-vulkan COMMAND llmx-mxfp4-vulkan-test)
        set_tests_properties(mxfp4-vulkan PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
        add_executable(llmx-backend-vulkan-test tests/backend_vulkan.cpp)
        target_link_libraries(llmx-backend-vulkan-test PRIVATE llmx-vulkan)
        list(APPEND LLMX_TARGETS llmx-backend-vulkan-test)
        add_test(NAME backend-vulkan COMMAND llmx-backend-vulkan-test)
        # It checks millions of kernel outputs against the CPU backend, so it gets the longest timeout.
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
        # Reads the private packed activation buffers, before consumer float arithmetic can flush them.
        add_executable(llmx-vulkan-quantization-test tests/vulkan_quantization.cpp)
        add_dependencies(llmx-vulkan-quantization-test llmx-vulkan-shaders)
        target_include_directories(llmx-vulkan-quantization-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_BINARY_DIR}/generated ${LLMX_VULKAN_INCLUDE_DIR})
        target_link_libraries(llmx-vulkan-quantization-test PRIVATE ${CMAKE_DL_LIBS})
        list(APPEND LLMX_TARGETS llmx-vulkan-quantization-test)
        add_test(NAME vulkan-quantization COMMAND llmx-vulkan-quantization-test)
        set_tests_properties(vulkan-quantization PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
        # Not a test: times the routed expert projections against dense matmuls of the same bytes.
        add_executable(llmx-moe-kernel-bench tools/moe_kernel_bench.cpp)
        target_link_libraries(llmx-moe-kernel-bench PRIVATE llmx-vulkan)
        list(APPEND LLMX_TARGETS llmx-moe-kernel-bench)
        # Not a test: the device queries and handoff timings of phase 0 in docs/MULTI-DEVICE.md, on the loader alone.
        add_executable(llmx-vk-handoff tools/vulkan_handoff.cpp)
        target_include_directories(llmx-vk-handoff PRIVATE ${LLMX_VULKAN_INCLUDE_DIR})
        target_link_libraries(llmx-vk-handoff PRIVATE ${CMAKE_DL_LIBS})
        list(APPEND LLMX_TARGETS llmx-vk-handoff)
        # Not a test: phase 0's concurrency, pipeline, group sum and expert exchange measurements through the backend, and a split model's stage times for the layer split's phase 3.
        add_executable(llmx-multi-device-bench tools/multi_device_bench.cpp)
        target_link_libraries(llmx-multi-device-bench PRIVATE llmx-vulkan)
        list(APPEND LLMX_TARGETS llmx-multi-device-bench)
    endif()
endif()

find_package(Threads REQUIRED)
foreach(target IN LISTS LLMX_TARGETS)
    # The x86 build requires AVX2/FMA/F16C, which the CPU kernels use with no runtime check; without them cpu_backend.hpp stops the compile.
    if(MSVC)
        target_compile_options(${target} PRIVATE /std:c++17 /O2 /arch:AVX2)
        get_target_property(target_kind ${target} TYPE)
        if(target_kind STREQUAL "EXECUTABLE")
            target_sources(${target} PRIVATE cmake/windows.manifest)
        endif()
    else()
        target_compile_options(${target} PRIVATE -std=c++17 -O3)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
            target_compile_options(${target} PRIVATE -mavx2 -mfma -mf16c)
        endif()
    endif()

    # An unused function of one translation unit, or an unused local, fails the build, since nothing reaches it (AGENTS.md, Dead code and stale docs).
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /we4505 /we4101 /we4189)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Werror=unused-function -Werror=unused-variable -Werror=unused-but-set-variable)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(${target} PRIVATE -Werror=unused-private-field)
        endif()
    endif()

    target_link_libraries(${target} PRIVATE Threads::Threads)
endforeach()
