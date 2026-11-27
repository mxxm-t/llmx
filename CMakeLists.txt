cmake_minimum_required(VERSION 3.16)
project(llmx VERSION 0.2.1 LANGUAGES CXX)

# llmx: a dependency-free GGUF runtime, built on Windows, Linux and macOS.
# The CPU backend is always built; the Vulkan backend is opt-in through LLMX_HAS_BACKEND_VULKAN and needs its SDK, the ROCm backend through LLMX_HAS_BACKEND_ROCM, on Linux or Windows, with the ROCm SDK, and the CUDA backend through LLMX_HAS_BACKEND_CUDA, on Linux, with the CUDA toolkit.
# Split mode and node count are runtime parameters, not build options.

option(LLMX_HAS_BACKEND_VULKAN "Build the Vulkan backend" OFF)
option(LLMX_HAS_BACKEND_ROCM "Build the ROCm backend (Linux, Windows)" OFF)
option(LLMX_HAS_BACKEND_CUDA "Build the CUDA backend (Linux)" OFF)
option(LLMX_NIGHTLY "Mark the build identifier as a nightly build of its commit (docs/RELEASE.md)" OFF)

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
if(LLMX_HAS_BACKEND_ROCM)
    set(LLMX_BACKEND_ROCM_FLAG 1)
else()
    set(LLMX_BACKEND_ROCM_FLAG 0)
endif()
if(LLMX_HAS_BACKEND_CUDA)
    set(LLMX_BACKEND_CUDA_FLAG 1)
else()
    set(LLMX_BACKEND_CUDA_FLAG 0)
endif()
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/llmx-config.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/config.hpp
    @ONLY
)

# What the build identifier carries beyond the revision, for the disk tier's identity (docs/DISK-TIER.md, The entry file): the compiler, the flags of the configuration built and the shader compiler.
set(LLMX_GLSLC_VERSION "")
if(LLMX_HAS_BACKEND_VULKAN)
    find_program(LLMX_GLSLC glslc HINTS $ENV{VULKAN_SDK}/Bin $ENV{VULKAN_SDK}/bin)
    if(LLMX_GLSLC)
        execute_process(COMMAND ${LLMX_GLSLC} --version OUTPUT_VARIABLE LLMX_GLSLC_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        string(REGEX REPLACE "[\r\n]+" ", " LLMX_GLSLC_VERSION "${LLMX_GLSLC_VERSION}")
    endif()
endif()
add_custom_target(llmx-build-info
    COMMAND ${CMAKE_COMMAND}
        "-DSOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}"
        "-DCOMPILER=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
        "-DBUILD_CONFIG=$<CONFIG>"
        "-DCXX_FLAGS=${CMAKE_CXX_FLAGS}"
        "-DCXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}"
        "-DCXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}"
        "-DCXX_FLAGS_RELWITHDEBINFO=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}"
        "-DCXX_FLAGS_MINSIZEREL=${CMAKE_CXX_FLAGS_MINSIZEREL}"
        "-DVULKAN=${LLMX_BACKEND_VULKAN_FLAG}"
        "-DIPO=${CMAKE_INTERPROCEDURAL_OPTIMIZATION}"
        "-DGLSLC_VERSION=${LLMX_GLSLC_VERSION}"
        "-DNIGHTLY=${LLMX_NIGHTLY}"
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
    # Shaders include shared files transitively; keep the full include set current when files are added.
    file(GLOB LLMX_VK_SHADER_INCLUDES CONFIGURE_DEPENDS "${LLMX_VK_SHADER_DIR}/*.glsl")
    set(LLMX_VK_SHADERS copy_mxfp4 add sum_rows silu_mul gather_rows rms_norm_rows norm_rope_partial embed matmul_row
        kv_write attention attention_merge matmul_tile matmul_tile_wide norm_rope_kv attention_tile quantize_x
        quantize_xw matmul_tile_q matmul_reduce matmul_vec_q8 moe_route moe_combine moe_group moe_step_route moe_step_act expert_mark expert_wave expert_pull expert_round attention_vec
        sigmoid_mul gated_rms_norm causal_conv_silu dilated_conv_silu_add copy_rows softcap l2_clip query_mix tanh_gate_mix depthwise_ring attention_rel_sink ring_store delta_rule argmax_rows
        group_rms_norm silu_scaled mix_streams inject_streams signed_sqrt_gate
        kv_index_pool kv_index_scores_tile kv_index_select kv_index_select_wide add_bias gelu layer_norm_rows mix_rows hold flag_push flag_recv
        hc_coeffs hc_collapse_norm hc_expand sparse_softmax sparse_columns topk_rows compress_pool indexer_scores indexer_scores_tile raw_index_scores pool_write model_format_qdq route_biased
        unit_rows sigmoid_gate_rows matmul_at_ids chain_pick tap_weighted_rows)
    # A variant is the same source compiled with a define, "name=source=DEFINE"; the row kernel is one module per family of types, since one module for all of them raised every path's registers.
    set(LLMX_VK_VARIANTS attention_rel_sink_tile=attention_rel_sink=LLMX_TILE moe_route_s1=moe_route=LLMX_SLOTS1 moe_route_s2=moe_route=LLMX_SLOTS2 moe_route_s4=moe_route=LLMX_SLOTS4 moe_route_s8=moe_route=LLMX_SLOTS8
        matmul_row_iq4_dot=matmul_row=LLMX_IQ4+LLMX_DOT+LLMX_PRESERVE_F32
        matmul_row_iq4_xs_dot=matmul_row=LLMX_IQ4+LLMX_IQ4_XS+LLMX_DOT+LLMX_PRESERVE_F32
        matmul_row_mxfp4=matmul_row=LLMX_MXFP4+LLMX_PRESERVE_F32
        matmul_row_mxfp4_dot=matmul_row=LLMX_MXFP4+LLMX_DOT+LLMX_PRESERVE_F32
        matmul_row_mxfp4_float_x=matmul_row=LLMX_MXFP4+LLMX_FLOAT_X+LLMX_PRESERVE_F32
        matmul_tile_iq4=matmul_tile_q=LLMX_Q8+LLMX_IQ4+LLMX_PRESERVE_F32
        matmul_tile_iq4_xs=matmul_tile_q=LLMX_Q8+LLMX_IQ4+LLMX_IQ4_XS+LLMX_PRESERVE_F32
        matmul_tile_q8mx=matmul_tile_q=LLMX_Q8+LLMX_Q8MX+LLMX_PRESERVE_F32
        embed_mxfp4=embed=LLMX_MXFP4 matmul_tile_mxfp4=matmul_tile=LLMX_MXFP4+LLMX_PRESERVE_F32
        matmul_tile_mxfp4_narrow=matmul_tile=LLMX_MXFP4+LLMX_PRESERVE_F32+LLMX_TILE_NARROW
        matmul_tile_mxfp4_bf16=matmul_tile=LLMX_MXFP4+LLMX_PRESERVE_F32+LLMX_BF16 matmul_reduce_mxfp4=matmul_reduce=LLMX_MXFP4+LLMX_PRESERVE_F32 matmul_reduce_float=matmul_reduce=LLMX_FLOAT_REPLAY
        matmul_row_f32_only=matmul_row=LLMX_ROW_F32 matmul_row_q8w=matmul_row=LLMX_Q8W matmul_row_q4=matmul_row=LLMX_Q4 matmul_row_k4=matmul_row=LLMX_K4 matmul_row_k5=matmul_row=LLMX_K5
        matmul_tile_iq1_s_dot=matmul_tile_q=LLMX_Q8+LLMX_IQ1+LLMX_IQ1_S+LLMX_PRESERVE_F32
        matmul_tile_iq1_m_dot=matmul_tile_q=LLMX_Q8+LLMX_IQ1+LLMX_IQ1_M+LLMX_PRESERVE_F32
        matmul_tile_iq2_s_dot=matmul_tile_q=LLMX_Q8+LLMX_IQ2_S+LLMX_PRESERVE_F32
        matmul_tile_iq3_xxs_dot=matmul_tile_q=LLMX_Q8+LLMX_IQ3_XXS+LLMX_PRESERVE_F32
        matmul_tile_iq2_xxs_dot=matmul_tile_q=LLMX_Q8+LLMX_IQ2_SMALL+LLMX_IQ2_XXS+LLMX_PRESERVE_F32
        matmul_tile_iq2_xs_dot=matmul_tile_q=LLMX_Q8+LLMX_IQ2_SMALL+LLMX_IQ2_XS+LLMX_PRESERVE_F32
        matmul_row_iq1_s_dot=matmul_row=LLMX_IQ1+LLMX_IQ1_S+LLMX_DOT+LLMX_PRESERVE_F32
        matmul_row_iq1_m_dot=matmul_row=LLMX_IQ1+LLMX_IQ1_M+LLMX_DOT+LLMX_PRESERVE_F32
        matmul_row_iq2_s_dot=matmul_row=LLMX_IQ2_S+LLMX_DOT+LLMX_PRESERVE_F32
        matmul_row_iq3_xxs_dot=matmul_row=LLMX_IQ3_XXS+LLMX_DOT+LLMX_PRESERVE_F32
        matmul_row_iq2_xxs_dot=matmul_row=LLMX_IQ2_SMALL+LLMX_IQ2_XXS+LLMX_DOT+LLMX_PRESERVE_F32
        matmul_row_iq2_xs_dot=matmul_row=LLMX_IQ2_SMALL+LLMX_IQ2_XS+LLMX_DOT+LLMX_PRESERVE_F32
        matmul_tile_iq3_s_dot=matmul_tile_q=LLMX_Q8+LLMX_IQ3_S+LLMX_PRESERVE_F32
        matmul_row_iq3_s_dot=matmul_row=LLMX_IQ3_S+LLMX_DOT+LLMX_PRESERVE_F32
        matmul_row_iq1_s=matmul_row=LLMX_FLOAT_X+LLMX_IQ1+LLMX_IQ1_S
        matmul_row_iq1_m=matmul_row=LLMX_FLOAT_X+LLMX_IQ1+LLMX_IQ1_M
        matmul_row_iq2_s=matmul_row=LLMX_FLOAT_X+LLMX_IQ2_S
        matmul_row_iq3_xxs=matmul_row=LLMX_FLOAT_X+LLMX_IQ3_XXS
        matmul_row_iq2_xxs=matmul_row=LLMX_FLOAT_X+LLMX_IQ2_SMALL+LLMX_IQ2_XXS
        matmul_row_iq2_xs=matmul_row=LLMX_FLOAT_X+LLMX_IQ2_SMALL+LLMX_IQ2_XS
        matmul_row_iq2_s_preserve=matmul_row=LLMX_FLOAT_X+LLMX_IQ2_S+LLMX_PRESERVE_F32
        matmul_row_iq3_xxs_preserve=matmul_row=LLMX_FLOAT_X+LLMX_IQ3_XXS+LLMX_PRESERVE_F32
        matmul_row_iq2_xxs_preserve=matmul_row=LLMX_FLOAT_X+LLMX_IQ2_SMALL+LLMX_IQ2_XXS+LLMX_PRESERVE_F32
        matmul_row_iq2_xs_preserve=matmul_row=LLMX_FLOAT_X+LLMX_IQ2_SMALL+LLMX_IQ2_XS+LLMX_PRESERVE_F32
        matmul_tile_iq2_s=matmul_tile=LLMX_IQ2_S
        matmul_tile_iq3_xxs=matmul_tile=LLMX_IQ3_XXS
        matmul_tile_iq2_xxs=matmul_tile=LLMX_IQ2_SMALL+LLMX_IQ2_XXS
        matmul_tile_iq2_xs=matmul_tile=LLMX_IQ2_SMALL+LLMX_IQ2_XS
        matmul_tile_iq2_s_preserve=matmul_tile=LLMX_IQ2_S+LLMX_PRESERVE_F32
        matmul_tile_iq3_xxs_preserve=matmul_tile=LLMX_IQ3_XXS+LLMX_PRESERVE_F32
        matmul_tile_iq2_xxs_preserve=matmul_tile=LLMX_IQ2_SMALL+LLMX_IQ2_XXS+LLMX_PRESERVE_F32
        matmul_tile_iq2_xs_preserve=matmul_tile=LLMX_IQ2_SMALL+LLMX_IQ2_XS+LLMX_PRESERVE_F32
        matmul_tile_iq2_s_bf16=matmul_tile=LLMX_IQ2_S+LLMX_BF16
        matmul_tile_iq3_xxs_bf16=matmul_tile=LLMX_IQ3_XXS+LLMX_BF16
        matmul_tile_iq2_xxs_bf16=matmul_tile=LLMX_IQ2_SMALL+LLMX_IQ2_XXS+LLMX_BF16
        matmul_tile_iq2_xs_bf16=matmul_tile=LLMX_IQ2_SMALL+LLMX_IQ2_XS+LLMX_BF16
        matmul_row_iq1_s_preserve=matmul_row=LLMX_FLOAT_X+LLMX_IQ1+LLMX_IQ1_S+LLMX_PRESERVE_F32
        matmul_row_iq1_m_preserve=matmul_row=LLMX_FLOAT_X+LLMX_IQ1+LLMX_IQ1_M+LLMX_PRESERVE_F32
        matmul_tile_iq1_s=matmul_tile=LLMX_IQ1+LLMX_IQ1_S
        matmul_tile_iq1_m=matmul_tile=LLMX_IQ1+LLMX_IQ1_M
        matmul_tile_iq1_s_preserve=matmul_tile=LLMX_IQ1+LLMX_IQ1_S+LLMX_PRESERVE_F32
        matmul_tile_iq1_m_preserve=matmul_tile=LLMX_IQ1+LLMX_IQ1_M+LLMX_PRESERVE_F32
        matmul_tile_iq1_s_bf16=matmul_tile=LLMX_IQ1+LLMX_IQ1_S+LLMX_BF16
        matmul_tile_iq1_m_bf16=matmul_tile=LLMX_IQ1+LLMX_IQ1_M+LLMX_BF16
        matmul_tile_iq1_s_bf16_preserve=matmul_tile=LLMX_IQ1+LLMX_IQ1_S+LLMX_BF16+LLMX_PRESERVE_F32
        matmul_tile_iq1_m_bf16_preserve=matmul_tile=LLMX_IQ1+LLMX_IQ1_M+LLMX_BF16+LLMX_PRESERVE_F32
        matmul_tile_iq2_s_bf16_preserve=matmul_tile=LLMX_IQ2_S+LLMX_BF16+LLMX_PRESERVE_F32
        matmul_tile_iq3_xxs_bf16_preserve=matmul_tile=LLMX_IQ3_XXS+LLMX_BF16+LLMX_PRESERVE_F32
        matmul_tile_iq2_xxs_bf16_preserve=matmul_tile=LLMX_IQ2_SMALL+LLMX_IQ2_XXS+LLMX_BF16+LLMX_PRESERVE_F32
        matmul_tile_iq2_xs_bf16_preserve=matmul_tile=LLMX_IQ2_SMALL+LLMX_IQ2_XS+LLMX_BF16+LLMX_PRESERVE_F32
        matmul_row_iq3_s=matmul_row=LLMX_FLOAT_X+LLMX_IQ3_S
        matmul_row_iq3_s_preserve=matmul_row=LLMX_FLOAT_X+LLMX_IQ3_S+LLMX_PRESERVE_F32
        matmul_tile_iq3_s=matmul_tile=LLMX_IQ3_S
        matmul_tile_iq3_s_preserve=matmul_tile=LLMX_IQ3_S+LLMX_PRESERVE_F32
        matmul_tile_iq3_s_bf16=matmul_tile=LLMX_IQ3_S+LLMX_BF16
        matmul_tile_iq3_s_bf16_preserve=matmul_tile=LLMX_IQ3_S+LLMX_BF16+LLMX_PRESERVE_F32
        matmul_row_float_x=matmul_row=LLMX_FLOAT_X
        matmul_row_float_x_preserve=matmul_row=LLMX_FLOAT_X+LLMX_PRESERVE_F32
        matmul_row_nvfp4=matmul_row=LLMX_FLOAT_X+LLMX_NVFP4
        matmul_row_nvfp4_preserve=matmul_row=LLMX_FLOAT_X+LLMX_NVFP4+LLMX_PRESERVE_F32
        matmul_row_fp8=matmul_row=LLMX_FLOAT_X+LLMX_FP8
        matmul_row_fp8_preserve=matmul_row=LLMX_FLOAT_X+LLMX_FP8+LLMX_PRESERVE_F32
        matmul_row_fp8_f16=matmul_row=LLMX_FLOAT_X+LLMX_FP8+LLMX_INPUT_F16+LLMX_PRESERVE_F32
        matmul_row_q1_0=matmul_row=LLMX_FLOAT_X+LLMX_Q1_0
        matmul_row_q2_0=matmul_row=LLMX_FLOAT_X+LLMX_Q2_0
        matmul_row_q1_0_preserve=matmul_row=LLMX_FLOAT_X+LLMX_Q1_0+LLMX_PRESERVE_F32
        matmul_row_q2_0_preserve=matmul_row=LLMX_FLOAT_X+LLMX_Q2_0+LLMX_PRESERVE_F32
        matmul_row_float_x_q2=matmul_row=LLMX_FLOAT_X+LLMX_Q2
        matmul_row_float_x_q2_preserve=matmul_row=LLMX_FLOAT_X+LLMX_Q2+LLMX_PRESERVE_F32
        matmul_row_float_x_q3=matmul_row=LLMX_FLOAT_X+LLMX_Q3
        matmul_row_float_x_q3_preserve=matmul_row=LLMX_FLOAT_X+LLMX_Q3+LLMX_PRESERVE_F32
        matmul_row_q2_dot=matmul_row=LLMX_Q2+LLMX_DOT
        matmul_row_q2_dot_preserve=matmul_row=LLMX_Q2+LLMX_DOT+LLMX_PRESERVE_F32
        matmul_row_q3_dot=matmul_row=LLMX_Q3+LLMX_DOT
        matmul_row_q3_dot_preserve=matmul_row=LLMX_Q3+LLMX_DOT+LLMX_PRESERVE_F32
        matmul_row_preserve=matmul_row=LLMX_PRESERVE_F32
        matmul_tile_preserve=matmul_tile=LLMX_PRESERVE_F32
        matmul_tile_bf16=matmul_tile=LLMX_BF16
        matmul_tile_bf16_preserve=matmul_tile=LLMX_BF16+LLMX_PRESERVE_F32
        matmul_tile_nvfp4=matmul_tile=LLMX_NVFP4
        matmul_tile_nvfp4_preserve=matmul_tile=LLMX_NVFP4+LLMX_PRESERVE_F32
        matmul_tile_nvfp4_f16=matmul_tile=LLMX_NVFP4+LLMX_INPUT_F16+LLMX_PRESERVE_F32
        matmul_tile_nvfp4_bf16=matmul_tile=LLMX_NVFP4+LLMX_BF16
        matmul_tile_nvfp4_bf16_preserve=matmul_tile=LLMX_NVFP4+LLMX_BF16+LLMX_PRESERVE_F32
        matmul_tile_fp8=matmul_tile=LLMX_FP8
        matmul_tile_fp8_preserve=matmul_tile=LLMX_FP8+LLMX_PRESERVE_F32
        matmul_tile_fp8_f16=matmul_tile=LLMX_FP8+LLMX_INPUT_F16+LLMX_PRESERVE_F32
        matmul_tile_fp8_bf16=matmul_tile=LLMX_FP8+LLMX_BF16
        matmul_tile_fp8_bf16_preserve=matmul_tile=LLMX_FP8+LLMX_BF16+LLMX_PRESERVE_F32
        matmul_tile_q_preserve=matmul_tile_q=LLMX_PRESERVE_F32
        matmul_tile_q6_preserve=matmul_tile_q=LLMX_Q6+LLMX_PRESERVE_F32
        matmul_tile_q3=matmul_tile_q=LLMX_Q3
        matmul_tile_q3_preserve=matmul_tile_q=LLMX_Q3+LLMX_PRESERVE_F32
        matmul_tile_q2=matmul_tile_q=LLMX_Q2
        matmul_tile_q2_preserve=matmul_tile_q=LLMX_Q2+LLMX_PRESERVE_F32
        matmul_reduce_preserve=matmul_reduce=LLMX_PRESERVE_F32
        matmul_reduce_float_preserve=matmul_reduce=LLMX_FLOAT_REPLAY+LLMX_PRESERVE_F32
        matmul_row_q8w_preserve=matmul_row=LLMX_Q8W+LLMX_PRESERVE_F32
        matmul_vec_q8_preserve=matmul_vec_q8=LLMX_PRESERVE_F32
        silu_scaled_preserve=silu_scaled=LLMX_PRESERVE_F32 mix_streams_preserve=mix_streams=LLMX_PRESERVE_F32
        inject_streams_preserve=inject_streams=LLMX_PRESERVE_F32
        moe_combine_preserve=moe_combine=LLMX_PRESERVE_F32
        moe_step_route_s1=moe_step_route=LLMX_SLOTS1
        moe_step_route_s2=moe_step_route=LLMX_SLOTS2
        moe_step_route_s4=moe_step_route=LLMX_SLOTS4
        moe_step_route_s8=moe_step_route=LLMX_SLOTS8
        moe_step_up_k4q8=moe_step_up=LLMX_K4+LLMX_DOT+LLMX_SHARED_Q8
        moe_step_up_k4q8_i8=moe_step_up=LLMX_K4+LLMX_DOT+LLMX_SHARED_Q8+LLMX_I8
        moe_step_up_k4q8_preserve=moe_step_up=LLMX_K4+LLMX_DOT+LLMX_SHARED_Q8+LLMX_PRESERVE_F32
        moe_step_up_k4q8_i8_preserve=moe_step_up=LLMX_K4+LLMX_DOT+LLMX_SHARED_Q8+LLMX_I8+LLMX_PRESERVE_F32
        moe_step_up_k4=moe_step_up=LLMX_K4+LLMX_DOT
        moe_step_up_k4_i8=moe_step_up=LLMX_K4+LLMX_DOT+LLMX_I8
        moe_step_up_k4_preserve=moe_step_up=LLMX_K4+LLMX_DOT+LLMX_PRESERVE_F32
        moe_step_up_k4_i8_preserve=moe_step_up=LLMX_K4+LLMX_DOT+LLMX_I8+LLMX_PRESERVE_F32
        moe_step_down_q8=moe_step_down=LLMX_ROUTED_Q8
        moe_step_down_q8_i8=moe_step_down=LLMX_ROUTED_Q8+LLMX_Q8_I8
        moe_step_down_q8_preserve=moe_step_down=LLMX_ROUTED_Q8+LLMX_PRESERVE_F32
        moe_step_down_q8_i8_preserve=moe_step_down=LLMX_ROUTED_Q8+LLMX_Q8_I8+LLMX_PRESERVE_F32
        moe_step_down_q5q8=moe_step_down=LLMX_Q4+LLMX_Q5+LLMX_DOT+LLMX_SHARED_Q8
        moe_step_down_q5q8_i8=moe_step_down=LLMX_Q4+LLMX_Q5+LLMX_DOT+LLMX_SHARED_Q8+LLMX_I8
        moe_step_down_q5q8_preserve=moe_step_down=LLMX_Q4+LLMX_Q5+LLMX_DOT+LLMX_SHARED_Q8+LLMX_PRESERVE_F32
        moe_step_down_q5q8_i8_preserve=moe_step_down=LLMX_Q4+LLMX_Q5+LLMX_DOT+LLMX_SHARED_Q8+LLMX_I8+LLMX_PRESERVE_F32
        moe_step_down_q8q8=moe_step_down=LLMX_ROUTED_Q8+LLMX_SHARED_Q8
        moe_step_down_q8q8_i8=moe_step_down=LLMX_ROUTED_Q8+LLMX_SHARED_Q8+LLMX_Q8_I8
        moe_step_down_q8q8_preserve=moe_step_down=LLMX_ROUTED_Q8+LLMX_SHARED_Q8+LLMX_PRESERVE_F32
        moe_step_down_q8q8_i8_preserve=moe_step_down=LLMX_ROUTED_Q8+LLMX_SHARED_Q8+LLMX_Q8_I8+LLMX_PRESERVE_F32
        moe_step_down_kq8=moe_step_down=LLMX_KQUANT+LLMX_DOT+LLMX_SHARED_Q8
        moe_step_down_kq8_i8=moe_step_down=LLMX_KQUANT+LLMX_DOT+LLMX_SHARED_Q8+LLMX_I8
        moe_step_down_kq8_preserve=moe_step_down=LLMX_KQUANT+LLMX_DOT+LLMX_SHARED_Q8+LLMX_PRESERVE_F32
        moe_step_down_kq8_i8_preserve=moe_step_down=LLMX_KQUANT+LLMX_DOT+LLMX_SHARED_Q8+LLMX_I8+LLMX_PRESERVE_F32
        moe_step_down_k5q8=moe_step_down=LLMX_K5+LLMX_DOT+LLMX_SHARED_Q8
        moe_step_down_k5q8_i8=moe_step_down=LLMX_K5+LLMX_DOT+LLMX_SHARED_Q8+LLMX_I8
        moe_step_down_k5q8_preserve=moe_step_down=LLMX_K5+LLMX_DOT+LLMX_SHARED_Q8+LLMX_PRESERVE_F32
        moe_step_down_k5q8_i8_preserve=moe_step_down=LLMX_K5+LLMX_DOT+LLMX_SHARED_Q8+LLMX_I8+LLMX_PRESERVE_F32
        moe_step_down_k4=moe_step_down=LLMX_K4+LLMX_DOT+LLMX_SHARED_ROW
        moe_step_down_k4_i8=moe_step_down=LLMX_K4+LLMX_DOT+LLMX_SHARED_ROW+LLMX_I8
        moe_step_down_k4_preserve=moe_step_down=LLMX_K4+LLMX_DOT+LLMX_SHARED_ROW+LLMX_PRESERVE_F32
        moe_step_down_k4_i8_preserve=moe_step_down=LLMX_K4+LLMX_DOT+LLMX_SHARED_ROW+LLMX_I8+LLMX_PRESERVE_F32
        moe_step_down_k=moe_step_down=LLMX_KQUANT+LLMX_DOT+LLMX_SHARED_ROW
        moe_step_down_k_i8=moe_step_down=LLMX_KQUANT+LLMX_DOT+LLMX_SHARED_ROW+LLMX_I8
        moe_step_down_k_preserve=moe_step_down=LLMX_KQUANT+LLMX_DOT+LLMX_SHARED_ROW+LLMX_PRESERVE_F32
        moe_step_down_k_i8_preserve=moe_step_down=LLMX_KQUANT+LLMX_DOT+LLMX_SHARED_ROW+LLMX_I8+LLMX_PRESERVE_F32
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
        sparse_attention=attention=LLMX_SPARSE
        sparse_attention_k16=attention=LLMX_SPARSE+LLMX_K16
        sparse_attention_v16=attention=LLMX_SPARSE+LLMX_V16
        sparse_attention_kv16=attention=LLMX_SPARSE+LLMX_K16+LLMX_V16
        sparse_attention_vec=attention_vec=LLMX_SPARSE
        sparse_attention_vec_k16=attention_vec=LLMX_SPARSE+LLMX_K16
        sparse_attention_vec_v16=attention_vec=LLMX_SPARSE+LLMX_V16
        sparse_attention_vec_kv16=attention_vec=LLMX_SPARSE+LLMX_K16+LLMX_V16
        sparse_attention_vec_d256=attention_vec=LLMX_SPARSE+LLMX_D256
        sparse_attention_vec_d256_k16=attention_vec=LLMX_SPARSE+LLMX_D256+LLMX_K16
        sparse_attention_vec_d256_v16=attention_vec=LLMX_SPARSE+LLMX_D256+LLMX_V16
        sparse_attention_vec_d256_kv16=attention_vec=LLMX_SPARSE+LLMX_D256+LLMX_K16+LLMX_V16
        sparse_attention_vec_d256_g2=attention_vec=LLMX_SPARSE+LLMX_D256+LLMX_G2
        sparse_attention_vec_d256_k16_g2=attention_vec=LLMX_SPARSE+LLMX_D256+LLMX_K16+LLMX_G2
        sparse_attention_vec_d256_v16_g2=attention_vec=LLMX_SPARSE+LLMX_D256+LLMX_V16+LLMX_G2
        sparse_attention_vec_d256_kv16_g2=attention_vec=LLMX_SPARSE+LLMX_D256+LLMX_K16+LLMX_V16+LLMX_G2
        sparse_attention_vec_d256_g3=attention_vec=LLMX_SPARSE+LLMX_D256+LLMX_G3
        sparse_attention_vec_d256_k16_g3=attention_vec=LLMX_SPARSE+LLMX_D256+LLMX_K16+LLMX_G3
        sparse_attention_vec_d256_v16_g3=attention_vec=LLMX_SPARSE+LLMX_D256+LLMX_V16+LLMX_G3
        sparse_attention_vec_d256_kv16_g3=attention_vec=LLMX_SPARSE+LLMX_D256+LLMX_K16+LLMX_V16+LLMX_G3
        sparse_attention_vec_d256_g4=attention_vec=LLMX_SPARSE+LLMX_D256+LLMX_G4
        sparse_attention_vec_d256_k16_g4=attention_vec=LLMX_SPARSE+LLMX_D256+LLMX_K16+LLMX_G4
        sparse_attention_vec_d256_v16_g4=attention_vec=LLMX_SPARSE+LLMX_D256+LLMX_V16+LLMX_G4
        sparse_attention_vec_d256_kv16_g4=attention_vec=LLMX_SPARSE+LLMX_D256+LLMX_K16+LLMX_V16+LLMX_G4
        kv_write_k16=kv_write=LLMX_K16 kv_write_v16=kv_write=LLMX_V16 kv_write_kv16=kv_write=LLMX_K16+LLMX_V16
        attention_k16=attention=LLMX_K16 attention_v16=attention=LLMX_V16 attention_kv16=attention=LLMX_K16+LLMX_V16
        attention_g4=attention=LLMX_G4 attention_k16_g4=attention=LLMX_K16+LLMX_G4 attention_v16_g4=attention=LLMX_V16+LLMX_G4
        attention_kv16_g4=attention=LLMX_K16+LLMX_V16+LLMX_G4
        attention_full=attention=LLMX_FULL norm_rope_partial_streams=norm_rope_partial=LLMX_STREAMS
        norm_rope_partial_form=norm_rope_partial=LLMX_STREAMS+LLMX_FORM silu_mul_clamp=silu_mul=LLMX_CLAMP
        attention_full_tile_d64=attention_flash=LLMX_D64 attention_full_tile_d72=attention_flash=LLMX_D72
        attention_vec_k16=attention_vec=LLMX_K16 attention_vec_v16=attention_vec=LLMX_V16 attention_vec_kv16=attention_vec=LLMX_K16+LLMX_V16
        attention_vec_g2=attention_vec=LLMX_G2 attention_vec_k16_g2=attention_vec=LLMX_K16+LLMX_G2
        attention_vec_v16_g2=attention_vec=LLMX_V16+LLMX_G2 attention_vec_kv16_g2=attention_vec=LLMX_K16+LLMX_V16+LLMX_G2
        attention_vec_g3=attention_vec=LLMX_G3 attention_vec_k16_g3=attention_vec=LLMX_K16+LLMX_G3
        attention_vec_v16_g3=attention_vec=LLMX_V16+LLMX_G3 attention_vec_kv16_g3=attention_vec=LLMX_K16+LLMX_V16+LLMX_G3
        attention_vec_g4=attention_vec=LLMX_G4 attention_vec_k16_g4=attention_vec=LLMX_K16+LLMX_G4
        attention_vec_v16_g4=attention_vec=LLMX_V16+LLMX_G4 attention_vec_kv16_g4=attention_vec=LLMX_K16+LLMX_V16+LLMX_G4
        attention_tile_k16=attention_tile=LLMX_K16 attention_tile_v16=attention_tile=LLMX_V16
        attention_tile_kv16=attention_tile=LLMX_K16+LLMX_V16
        attention_tile_d256=attention_tile=LLMX_D256 attention_tile_d256_k16=attention_tile=LLMX_D256+LLMX_K16
        attention_tile_d256_v16=attention_tile=LLMX_D256+LLMX_V16 attention_tile_d256_kv16=attention_tile=LLMX_D256+LLMX_K16+LLMX_V16
        attention_vec_d256=attention_vec=LLMX_D256 attention_vec_d256_k16=attention_vec=LLMX_D256+LLMX_K16
        attention_window=attention=LLMX_WINDOW attention_k16_window=attention=LLMX_WINDOW+LLMX_K16
        attention_v16_window=attention=LLMX_WINDOW+LLMX_V16 attention_kv16_window=attention=LLMX_WINDOW+LLMX_K16+LLMX_V16
        attention_vec_window=attention_vec=LLMX_WINDOW attention_vec_k16_window=attention_vec=LLMX_WINDOW+LLMX_K16
        attention_vec_v16_window=attention_vec=LLMX_WINDOW+LLMX_V16 attention_vec_kv16_window=attention_vec=LLMX_WINDOW+LLMX_K16+LLMX_V16
        attention_vec_d256_window=attention_vec=LLMX_D256+LLMX_WINDOW attention_vec_d256_k16_window=attention_vec=LLMX_D256+LLMX_WINDOW+LLMX_K16
        attention_vec_d256_v16_window=attention_vec=LLMX_D256+LLMX_WINDOW+LLMX_V16 attention_vec_d256_kv16_window=attention_vec=LLMX_D256+LLMX_WINDOW+LLMX_K16+LLMX_V16
        attention_vec_d256_v16=attention_vec=LLMX_D256+LLMX_V16 attention_vec_d256_kv16=attention_vec=LLMX_D256+LLMX_K16+LLMX_V16
        attention_vec_d256_g2=attention_vec=LLMX_D256+LLMX_G2 attention_vec_d256_k16_g2=attention_vec=LLMX_D256+LLMX_K16+LLMX_G2
        attention_vec_d256_v16_g2=attention_vec=LLMX_D256+LLMX_V16+LLMX_G2 attention_vec_d256_kv16_g2=attention_vec=LLMX_D256+LLMX_K16+LLMX_V16+LLMX_G2
        attention_vec_d256_g3=attention_vec=LLMX_D256+LLMX_G3 attention_vec_d256_k16_g3=attention_vec=LLMX_D256+LLMX_K16+LLMX_G3
        attention_vec_d256_v16_g3=attention_vec=LLMX_D256+LLMX_V16+LLMX_G3 attention_vec_d256_kv16_g3=attention_vec=LLMX_D256+LLMX_K16+LLMX_V16+LLMX_G3
        attention_vec_d256_g4=attention_vec=LLMX_D256+LLMX_G4 attention_vec_d256_k16_g4=attention_vec=LLMX_D256+LLMX_K16+LLMX_G4
        attention_vec_d256_v16_g4=attention_vec=LLMX_D256+LLMX_V16+LLMX_G4 attention_vec_d256_kv16_g4=attention_vec=LLMX_D256+LLMX_K16+LLMX_V16+LLMX_G4
        norm_rope_kv_k16=norm_rope_kv=LLMX_K16 norm_rope_kv_v16=norm_rope_kv=LLMX_V16
        norm_rope_kv_kv16=norm_rope_kv=LLMX_K16+LLMX_V16
        matmul_tile_q6=matmul_tile_q=LLMX_Q6 matmul_tile_q8=matmul_tile_q=LLMX_Q8
        matmul_tile_qi8=matmul_tile_q=LLMX_I8 matmul_tile_q6i8=matmul_tile_q=LLMX_Q6+LLMX_I8 matmul_tile_q8i8=matmul_tile_q=LLMX_Q8+LLMX_I8
        matmul_tile_qi8_preserve=matmul_tile_q=LLMX_I8+LLMX_PRESERVE_F32 matmul_tile_q6i8_preserve=matmul_tile_q=LLMX_Q6+LLMX_I8+LLMX_PRESERVE_F32
        matmul_row_q4_i8=matmul_row=LLMX_Q4+LLMX_DOT+LLMX_I8 matmul_row_k4_i8=matmul_row=LLMX_K4+LLMX_DOT+LLMX_I8
        matmul_row_k5_i8=matmul_row=LLMX_K5+LLMX_DOT+LLMX_I8 matmul_row_k_i8=matmul_row=LLMX_KQUANT+LLMX_DOT+LLMX_I8
        matmul_row_q4_i8_preserve=matmul_row=LLMX_Q4+LLMX_DOT+LLMX_I8+LLMX_PRESERVE_F32
        matmul_row_k4_i8_preserve=matmul_row=LLMX_K4+LLMX_DOT+LLMX_I8+LLMX_PRESERVE_F32
        matmul_row_k5_i8_preserve=matmul_row=LLMX_K5+LLMX_DOT+LLMX_I8+LLMX_PRESERVE_F32
        matmul_row_k_i8_preserve=matmul_row=LLMX_KQUANT+LLMX_DOT+LLMX_I8+LLMX_PRESERVE_F32
        matmul_vec_q8_i8=matmul_vec_q8=LLMX_I8 matmul_vec_q8_i8_preserve=matmul_vec_q8=LLMX_I8+LLMX_PRESERVE_F32
        embed_ids=embed=LLMX_DEVICE_IDS embed_ids_mxfp4=embed=LLMX_DEVICE_IDS+LLMX_MXFP4
        matmul_row_q5=matmul_row=LLMX_Q4+LLMX_Q5
        matmul_row_q50=matmul_row=LLMX_Q4+LLMX_Q5+LLMX_Q50
        matmul_tile_q5=matmul_tile_q=LLMX_Q5
        matmul_tile_q50=matmul_tile_q=LLMX_Q5+LLMX_Q50
        matmul_tile_q5_preserve=matmul_tile_q=LLMX_Q5+LLMX_PRESERVE_F32
        matmul_tile_q50_preserve=matmul_tile_q=LLMX_Q5+LLMX_Q50+LLMX_PRESERVE_F32
        matmul_tile_q5i8=matmul_tile_q=LLMX_Q5+LLMX_I8
        matmul_tile_q50i8=matmul_tile_q=LLMX_Q5+LLMX_Q50+LLMX_I8
        matmul_tile_q5i8_preserve=matmul_tile_q=LLMX_Q5+LLMX_I8+LLMX_PRESERVE_F32
        matmul_tile_q50i8_preserve=matmul_tile_q=LLMX_Q5+LLMX_Q50+LLMX_I8+LLMX_PRESERVE_F32
        matmul_row_q5_dot=matmul_row=LLMX_Q4+LLMX_Q5+LLMX_DOT
        matmul_row_q50_dot=matmul_row=LLMX_Q4+LLMX_Q5+LLMX_Q50+LLMX_DOT
        matmul_row_q5_i8=matmul_row=LLMX_Q4+LLMX_Q5+LLMX_DOT+LLMX_I8
        matmul_row_q50_i8=matmul_row=LLMX_Q4+LLMX_Q5+LLMX_Q50+LLMX_DOT+LLMX_I8
        matmul_row_q5_preserve=matmul_row=LLMX_Q4+LLMX_Q5+LLMX_PRESERVE_F32
        matmul_row_q50_preserve=matmul_row=LLMX_Q4+LLMX_Q5+LLMX_Q50+LLMX_PRESERVE_F32
        matmul_row_q5_dot_preserve=matmul_row=LLMX_Q4+LLMX_Q5+LLMX_DOT+LLMX_PRESERVE_F32
        matmul_row_q50_dot_preserve=matmul_row=LLMX_Q4+LLMX_Q5+LLMX_Q50+LLMX_DOT+LLMX_PRESERVE_F32
        matmul_row_q5_i8_preserve=matmul_row=LLMX_Q4+LLMX_Q5+LLMX_DOT+LLMX_I8+LLMX_PRESERVE_F32
        matmul_row_q50_i8_preserve=matmul_row=LLMX_Q4+LLMX_Q5+LLMX_Q50+LLMX_DOT+LLMX_I8+LLMX_PRESERVE_F32
        delta_rule_short=delta_rule=LLMX_SHORT
        embed_wf16=embed=LLMX_WEIGHT_F16 embed_wbf16=embed=LLMX_WEIGHT_BF16
        embed_ids_wf16=embed=LLMX_DEVICE_IDS+LLMX_WEIGHT_F16 embed_ids_wbf16=embed=LLMX_DEVICE_IDS+LLMX_WEIGHT_BF16
        matmul_row_wf16=matmul_row=LLMX_WEIGHT_F16 matmul_row_wbf16=matmul_row=LLMX_WEIGHT_BF16
        matmul_tile_wf32=matmul_tile=LLMX_WEIGHT_F32 matmul_tile_wf32_preserve=matmul_tile=LLMX_WEIGHT_F32+LLMX_PRESERVE_F32
        matmul_tile_wf16=matmul_tile=LLMX_WEIGHT_F16
        matmul_tile_wf16_preserve=matmul_tile=LLMX_WEIGHT_F16+LLMX_PRESERVE_F32
        matmul_tile_wf16_bf16=matmul_tile=LLMX_WEIGHT_F16+LLMX_BF16
        matmul_tile_wf16_bf16_preserve=matmul_tile=LLMX_WEIGHT_F16+LLMX_BF16+LLMX_PRESERVE_F32
        matmul_tile_wbf16=matmul_tile=LLMX_WEIGHT_BF16
        matmul_tile_wide_preserve=matmul_tile_wide=LLMX_PRESERVE_F32
        matmul_tile_wide_bf16=matmul_tile_wide=LLMX_BF16 matmul_tile_wide_bf16_preserve=matmul_tile_wide=LLMX_BF16+LLMX_PRESERVE_F32
        matmul_tile_wide_wf16=matmul_tile_wide=LLMX_WEIGHT_F16 matmul_tile_wide_wf16_preserve=matmul_tile_wide=LLMX_WEIGHT_F16+LLMX_PRESERVE_F32
        matmul_tile_wide_wf16_bf16=matmul_tile_wide=LLMX_WEIGHT_F16+LLMX_BF16 matmul_tile_wide_wf16_bf16_preserve=matmul_tile_wide=LLMX_WEIGHT_F16+LLMX_BF16+LLMX_PRESERVE_F32
        matmul_tile_wide_wbf16=matmul_tile_wide=LLMX_WEIGHT_BF16 matmul_tile_wide_wbf16_preserve=matmul_tile_wide=LLMX_WEIGHT_BF16+LLMX_PRESERVE_F32
        matmul_tile_wide_wbf16_bf16=matmul_tile_wide=LLMX_WEIGHT_BF16+LLMX_BF16 matmul_tile_wide_wbf16_bf16_preserve=matmul_tile_wide=LLMX_WEIGHT_BF16+LLMX_BF16+LLMX_PRESERVE_F32
        matmul_tile_wbf16_preserve=matmul_tile=LLMX_WEIGHT_BF16+LLMX_PRESERVE_F32
        matmul_tile_wbf16_bf16=matmul_tile=LLMX_WEIGHT_BF16+LLMX_BF16
        matmul_tile_wbf16_bf16_preserve=matmul_tile=LLMX_WEIGHT_BF16+LLMX_BF16+LLMX_PRESERVE_F32
        matmul_tile_tq1_0_dot=matmul_tile_q=LLMX_Q8+LLMX_TQ+LLMX_TQ1_0+LLMX_PRESERVE_F32
        matmul_tile_tq2_0_dot=matmul_tile_q=LLMX_Q8+LLMX_TQ+LLMX_TQ2_0+LLMX_PRESERVE_F32
        matmul_tile_q1_0_dot=matmul_tile_q=LLMX_Q8+LLMX_LOW_BITS_DOT+LLMX_Q1_0+LLMX_PRESERVE_F32
        matmul_tile_q2_0_dot=matmul_tile_q=LLMX_Q8+LLMX_LOW_BITS_DOT+LLMX_PRESERVE_F32
        matmul_row_q1_0_dot=matmul_row=LLMX_DOT+LLMX_LOW_BITS_DOT+LLMX_Q1_0+LLMX_PRESERVE_F32
        matmul_row_q2_0_dot=matmul_row=LLMX_DOT+LLMX_LOW_BITS_DOT+LLMX_PRESERVE_F32
        matmul_row_tq1_0_dot=matmul_row=LLMX_TQ+LLMX_TQ1_0+LLMX_DOT+LLMX_PRESERVE_F32
        matmul_row_tq2_0_dot=matmul_row=LLMX_TQ+LLMX_TQ2_0+LLMX_DOT+LLMX_PRESERVE_F32
        matmul_row_tq1_0=matmul_row=LLMX_FLOAT_X+LLMX_TQ+LLMX_TQ1_0
        matmul_row_tq2_0=matmul_row=LLMX_FLOAT_X+LLMX_TQ+LLMX_TQ2_0
        matmul_row_tq1_0_preserve=matmul_row=LLMX_FLOAT_X+LLMX_TQ+LLMX_TQ1_0+LLMX_PRESERVE_F32
        matmul_row_tq2_0_preserve=matmul_row=LLMX_FLOAT_X+LLMX_TQ+LLMX_TQ2_0+LLMX_PRESERVE_F32
        matmul_tile_tq1_0=matmul_tile=LLMX_TQ+LLMX_TQ1_0
        matmul_tile_tq2_0=matmul_tile=LLMX_TQ+LLMX_TQ2_0
        matmul_tile_tq1_0_preserve=matmul_tile=LLMX_TQ+LLMX_TQ1_0+LLMX_PRESERVE_F32
        matmul_tile_tq2_0_preserve=matmul_tile=LLMX_TQ+LLMX_TQ2_0+LLMX_PRESERVE_F32
        matmul_tile_tq1_0_bf16=matmul_tile=LLMX_TQ+LLMX_TQ1_0+LLMX_BF16
        matmul_tile_tq2_0_bf16=matmul_tile=LLMX_TQ+LLMX_TQ2_0+LLMX_BF16
        matmul_tile_tq1_0_bf16_preserve=matmul_tile=LLMX_TQ+LLMX_TQ1_0+LLMX_BF16+LLMX_PRESERVE_F32
        matmul_tile_tq2_0_bf16_preserve=matmul_tile=LLMX_TQ+LLMX_TQ2_0+LLMX_BF16+LLMX_PRESERVE_F32)
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
            DEPENDS ${LLMX_VK_SHADER_DIR}/${source}.comp ${LLMX_VK_SHADER_INCLUDES}
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

# The shared GPU kernels (docs/src/backends-gpu.md), compiled by each GPU backend's own compiler.
set(LLMX_GPU_KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/gpu/kernels)
include(cmake/embed_modules.cmake)
# The memory the build may take, in GiB: the machine's, or a container's lower limit, which sets how many GPU targets compile at once (docs/BUILD.md, GPU architectures).
cmake_host_system_information(RESULT LLMX_BUILD_MEMORY_MIB QUERY TOTAL_PHYSICAL_MEMORY)
foreach(limit_file /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes)
    if(EXISTS ${limit_file})
        file(READ ${limit_file} limit)
        string(STRIP "${limit}" limit)
        if(limit MATCHES "^[0-9]+$" AND limit LESS 1099511627776)
            math(EXPR limit "${limit} / 1048576")
            if(limit LESS LLMX_BUILD_MEMORY_MIB)
                set(LLMX_BUILD_MEMORY_MIB ${limit})
            endif()
        endif()
    endif()
endforeach()
math(EXPR LLMX_BUILD_MEMORY_GIB "${LLMX_BUILD_MEMORY_MIB} / 1024")
# How many targets of one kernel file compile at once: one each, as far as the memory holds jobs of GIB_PER_JOB while both kernel files compile.
function(llmx_gpu_jobs out count gib_per_job)
    math(EXPR jobs "${LLMX_BUILD_MEMORY_GIB} / (2 * ${gib_per_job})")
    if(jobs LESS 1)
        set(jobs 1)
    elseif(jobs GREATER count)
        set(jobs ${count})
    endif()
    set(${out} ${jobs} PARENT_SCOPE)
endfunction()

# The ROCm backend (docs/ROCM.md): the SDK supplies the headers and the compiler at build time and the HIP runtime is opened at run time, so nothing links against libamdhip64.
if(LLMX_HAS_BACKEND_ROCM)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT WIN32)
        message(FATAL_ERROR "LLMX_HAS_BACKEND_ROCM is Linux and Windows only")
    endif()
    # ROCM_PATH on Linux; on Windows HIP_PATH, which the HIP SDK installer and TheRock's rocm-sdk wheels both set (docs/BUILD.md).
    set(LLMX_HIP_ROOTS $ENV{ROCM_PATH})
    if(WIN32)
        file(TO_CMAKE_PATH "$ENV{HIP_PATH}" LLMX_HIP_ROOTS)
    endif()
    list(TRANSFORM LLMX_HIP_ROOTS APPEND /include OUTPUT_VARIABLE LLMX_HIP_INCLUDE_HINTS)
    set(LLMX_HIP_CLANG_HINTS)
    foreach(root IN LISTS LLMX_HIP_ROOTS)
        list(APPEND LLMX_HIP_CLANG_HINTS ${root}/llvm/bin ${root}/lib/llvm/bin)
        # The HIP SDK installer keeps its clang++ in bin.
        if(WIN32)
            list(APPEND LLMX_HIP_CLANG_HINTS ${root}/bin)
        endif()
    endforeach()
    find_path(LLMX_HIP_INCLUDE_DIR hip/hip_runtime_api.h HINTS ${LLMX_HIP_INCLUDE_HINTS} /opt/rocm/include)
    find_program(LLMX_HIP_CLANG clang++ HINTS ${LLMX_HIP_CLANG_HINTS} /opt/rocm/llvm/bin NO_DEFAULT_PATH)
    if(NOT LLMX_HIP_INCLUDE_DIR OR NOT LLMX_HIP_CLANG)
        message(FATAL_ERROR "LLMX_HAS_BACKEND_ROCM needs hip/hip_runtime_api.h and the ROCm clang++: install the ROCm SDK, or set ROCM_PATH (HIP_PATH on Windows)")
    endif()
    # Kernels are HIP in the tree, compiled to a code object for each listed target and linked in as bytes (llmx_embed_modules); only gfx906 is gated on hardware (docs/BUILD.md).
    # Windows lists the targets a Windows HIP runtime runs: no Instinct target has one, and gfx906 runs on the display driver's HIP 5 runtime (docs/ROCM.md, Windows).
    if(WIN32)
        set(LLMX_ROCM_DEFAULT_ARCHS "gfx906;gfx1030;gfx1031;gfx1032;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201")
    else()
        set(LLMX_ROCM_DEFAULT_ARCHS "gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201")
    endif()
    set(LLMX_ROCM_ARCHS "${LLMX_ROCM_DEFAULT_ARCHS}" CACHE STRING "AMD GPU architectures the ROCm kernels are compiled for (docs/BUILD.md)")
    if(NOT LLMX_ROCM_ARCHS)
        message(FATAL_ERROR "LLMX_ROCM_ARCHS names no architecture")
    endif()
    list(TRANSFORM LLMX_ROCM_ARCHS PREPEND --offload-arch= OUTPUT_VARIABLE LLMX_ROCM_ARCH_FLAGS)
    # A target of the int16 kernels compiles in about 0.6 GB.
    list(LENGTH LLMX_ROCM_ARCHS LLMX_ROCM_ARCH_COUNT)
    llmx_gpu_jobs(LLMX_ROCM_JOBS ${LLMX_ROCM_ARCH_COUNT} 1)
    list(APPEND LLMX_ROCM_ARCH_FLAGS -parallel-jobs=${LLMX_ROCM_JOBS})
    # The compiler reads the SDK it was found with, never another HIP installed beside it.
    get_filename_component(LLMX_ROCM_ROOT ${LLMX_HIP_INCLUDE_DIR} DIRECTORY)
    # The Windows driver's HIP 5 runtime refuses gfx906 code objects past version 4 (docs/ROCM.md, Windows).
    set(LLMX_ROCM_WINDOWS_FLAGS)
    if(WIN32 AND "gfx906" IN_LIST LLMX_ROCM_ARCHS)
        set(LLMX_ROCM_WINDOWS_FLAGS -Xarch_gfx906 -mcode-object-version=4)
    endif()
    # TheRock keeps the device libraries under lib/llvm, where --rocm-path does not look.
    if(WIN32)
        find_path(LLMX_HIP_BITCODE_DIR ocml.bc HINTS ${LLMX_ROCM_ROOT}/amdgcn/bitcode ${LLMX_ROCM_ROOT}/lib/llvm/amdgcn/bitcode NO_DEFAULT_PATH)
        if(LLMX_HIP_BITCODE_DIR)
            list(APPEND LLMX_ROCM_WINDOWS_FLAGS --rocm-device-lib-path=${LLMX_HIP_BITCODE_DIR})
        endif()
    endif()
    set(LLMX_ROCM_KERNELS f32 int16)
    set(LLMX_ROCM_OBJECTS)
    # Every kernel compiles without contraction, so a build of a body cannot fuse a product another build of it keeps apart; a fused multiply-add is written as fmaf (docs/src/backends-gpu.md).
    set(LLMX_HIP_FLAGS -x hip -std=c++17 -O3 -ffp-contract=off --rocm-path=${LLMX_ROCM_ROOT} --cuda-device-only ${LLMX_ROCM_WINDOWS_FLAGS})
    foreach(kernel IN LISTS LLMX_ROCM_KERNELS)
        set(bundle ${CMAKE_CURRENT_BINARY_DIR}/generated/rocm/${kernel}.hipfb)
        add_custom_command(OUTPUT ${bundle}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/generated/rocm
            COMMAND ${LLMX_HIP_CLANG} ${LLMX_HIP_FLAGS} ${LLMX_ROCM_ARCH_FLAGS}
                    -I${CMAKE_CURRENT_SOURCE_DIR}/src -c ${LLMX_GPU_KERNEL_DIR}/${kernel}.gpu -o ${bundle}
            DEPENDS ${LLMX_GPU_KERNEL_DIR}/${kernel}.gpu ${LLMX_GPU_KERNEL_DIR}/quant.h ${LLMX_GPU_KERNEL_DIR}/port.h ${LLMX_GPU_KERNEL_DIR}/shapes.h ${LLMX_GPU_KERNEL_DIR}/group.h ${LLMX_GPU_KERNEL_DIR}/rows.h ${CMAKE_CURRENT_SOURCE_DIR}/src/quant/types.hpp
            COMMENT "hip ${kernel}"
            VERBATIM)
        list(APPEND LLMX_ROCM_OBJECTS ${bundle})
    endforeach()
    # The int16 kernels again, their products on the 16-bit integer dot, for gfx906 alone, the one target that has run them; a build without gfx906 has no such module, and a device it holds no code object for runs the plain module (docs/src/backends-gpu.md).
    set(LLMX_ROCM_MODULES llmx_rocm_f32_module ${CMAKE_CURRENT_BINARY_DIR}/generated/rocm/f32.hipfb llmx_rocm_int16_module ${CMAKE_CURRENT_BINARY_DIR}/generated/rocm/int16.hipfb)
    if("gfx906" IN_LIST LLMX_ROCM_ARCHS)
        set(bundle ${CMAKE_CURRENT_BINARY_DIR}/generated/rocm/int16dot.hipfb)
        add_custom_command(OUTPUT ${bundle}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/generated/rocm
            COMMAND ${LLMX_HIP_CLANG} ${LLMX_HIP_FLAGS} --offload-arch=gfx906 -DLLMX_DOT16
                    -I${CMAKE_CURRENT_SOURCE_DIR}/src -c ${LLMX_GPU_KERNEL_DIR}/int16.gpu -o ${bundle}
            DEPENDS ${LLMX_GPU_KERNEL_DIR}/int16.gpu ${LLMX_GPU_KERNEL_DIR}/quant.h ${LLMX_GPU_KERNEL_DIR}/port.h ${LLMX_GPU_KERNEL_DIR}/shapes.h ${LLMX_GPU_KERNEL_DIR}/group.h ${LLMX_GPU_KERNEL_DIR}/rows.h ${CMAKE_CURRENT_SOURCE_DIR}/src/quant/types.hpp
            COMMENT "hip int16dot"
            VERBATIM)
        list(APPEND LLMX_ROCM_OBJECTS ${bundle})
        list(APPEND LLMX_ROCM_MODULES llmx_rocm_int16dot_module ${bundle})
    endif()
    add_custom_target(llmx-rocm-kernels DEPENDS ${LLMX_ROCM_OBJECTS})
    add_library(llmx-rocm STATIC src/backends/rocm/rocm_backend.cpp)
    add_dependencies(llmx-rocm llmx-rocm-kernels)
    set(LLMX_EMBED_ASSEMBLER ${LLMX_HIP_CLANG})
    llmx_embed_modules(llmx-rocm rocm ${LLMX_ROCM_MODULES})
    if("gfx906" IN_LIST LLMX_ROCM_ARCHS)
        target_compile_definitions(llmx-rocm PRIVATE LLMX_ROCM_DOT_MODULE)
    endif()
    # The generated directory comes first, so the backend reads the build's config.hpp and its embedded kernels.
    target_include_directories(llmx-rocm
        PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/generated ${CMAKE_CURRENT_SOURCE_DIR}/src
        PRIVATE ${LLMX_HIP_INCLUDE_DIR})
    target_compile_definitions(llmx-rocm PRIVATE __HIP_PLATFORM_AMD__)
    target_link_libraries(llmx-rocm PUBLIC ${CMAKE_DL_LIBS})
    target_link_libraries(llmx PRIVATE llmx-rocm)
endif()

# The CUDA backend (docs/CUDA.md): the toolkit supplies cuda.h and nvcc at build time and the driver is opened at run time, so nothing links against libcuda or the CUDA runtime.
if(LLMX_HAS_BACKEND_CUDA)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "LLMX_HAS_BACKEND_CUDA is Linux only")
    endif()
    find_program(LLMX_NVCC nvcc HINTS $ENV{CUDA_PATH}/bin /usr/local/cuda/bin)
    find_path(LLMX_CUDA_INCLUDE_DIR cuda.h HINTS $ENV{CUDA_PATH}/include /usr/local/cuda/include)
    if(NOT LLMX_NVCC OR NOT LLMX_CUDA_INCLUDE_DIR)
        message(FATAL_ERROR "LLMX_HAS_BACKEND_CUDA needs nvcc and cuda.h: install the CUDA toolkit 12.8 or newer, or set CUDA_PATH")
    endif()
    # SASS for every listed target; 12.8 is the first toolkit with sm_120, and 13.0 the first without sm_70, which the default then leaves out (docs/BUILD.md).
    execute_process(COMMAND ${LLMX_NVCC} --version OUTPUT_VARIABLE LLMX_NVCC_VERSION)
    string(REGEX MATCH "release ([0-9]+)[.]([0-9]+)" LLMX_NVCC_VERSION "${LLMX_NVCC_VERSION}")
    if(CMAKE_MATCH_1 LESS 12 OR (CMAKE_MATCH_1 EQUAL 12 AND CMAKE_MATCH_2 LESS 8))
        message(FATAL_ERROR "LLMX_HAS_BACKEND_CUDA needs the CUDA toolkit 12.8 or newer, found ${LLMX_NVCC_VERSION}")
    endif()
    if(CMAKE_MATCH_1 LESS 13)
        set(LLMX_CUDA_DEFAULT_ARCHS "70;75;80;86;89;90;100;120")
    else()
        set(LLMX_CUDA_DEFAULT_ARCHS "75;80;86;89;90;100;120")
    endif()
    set(LLMX_CUDA_ARCHS "${LLMX_CUDA_DEFAULT_ARCHS}" CACHE STRING "NVIDIA compute capabilities the CUDA kernels are compiled for, as 86 for sm_86 (docs/BUILD.md)")
    # PTX of the lowest listed target with the integer tensor cores (sm_80), else of the lowest, which later GPUs compile at load.
    set(LLMX_CUDA_PTX)
    set(LLMX_CUDA_PTX_TENSOR)
    foreach(a IN LISTS LLMX_CUDA_ARCHS)
        if(NOT a MATCHES "^[0-9]+$")
            message(FATAL_ERROR "LLMX_CUDA_ARCHS takes compute capabilities as numbers, such as 86 for sm_86, not ${a}")
        endif()
        if(NOT LLMX_CUDA_PTX OR a LESS LLMX_CUDA_PTX)
            set(LLMX_CUDA_PTX ${a})
        endif()
        if(a GREATER_EQUAL 80 AND (NOT LLMX_CUDA_PTX_TENSOR OR a LESS LLMX_CUDA_PTX_TENSOR))
            set(LLMX_CUDA_PTX_TENSOR ${a})
        endif()
    endforeach()
    if(LLMX_CUDA_PTX_TENSOR)
        set(LLMX_CUDA_PTX ${LLMX_CUDA_PTX_TENSOR})
    endif()
    # ptxas takes about 2.4 GB for a target of the int16 kernels.
    list(LENGTH LLMX_CUDA_ARCHS LLMX_CUDA_ARCH_COUNT)
    llmx_gpu_jobs(LLMX_CUDA_JOBS ${LLMX_CUDA_ARCH_COUNT} 3)
    if(NOT LLMX_CUDA_PTX)
        message(FATAL_ERROR "LLMX_CUDA_ARCHS names no compute capability")
    endif()
    set(LLMX_CUDA_OBJECTS)
    foreach(kernel f32 int16)
        set(fatbin ${CMAKE_CURRENT_BINARY_DIR}/generated/cuda/${kernel}.fatbin)
        add_custom_command(OUTPUT ${fatbin} ${CMAKE_CURRENT_BINARY_DIR}/generated/cuda/${kernel}.ptxas
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/generated/cuda
            COMMAND ${CMAKE_COMMAND} -DNVCC=${LLMX_NVCC} -DSRC=${LLMX_GPU_KERNEL_DIR}/${kernel}.gpu -DOUT=${fatbin}
                    -DREPORT=${CMAKE_CURRENT_BINARY_DIR}/generated/cuda/${kernel}.ptxas -DINCLUDE=${CMAKE_CURRENT_SOURCE_DIR}/src
                    "-DARCHS=${LLMX_CUDA_ARCHS}" -DPTX=${LLMX_CUDA_PTX} -DJOBS=${LLMX_CUDA_JOBS} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/nvcc.cmake
            DEPENDS ${LLMX_GPU_KERNEL_DIR}/${kernel}.gpu ${LLMX_GPU_KERNEL_DIR}/quant.h ${LLMX_GPU_KERNEL_DIR}/port.h ${LLMX_GPU_KERNEL_DIR}/shapes.h ${LLMX_GPU_KERNEL_DIR}/group.h ${LLMX_GPU_KERNEL_DIR}/rows.h ${CMAKE_CURRENT_SOURCE_DIR}/src/quant/types.hpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/nvcc.cmake
            COMMENT "nvcc ${kernel}"
            VERBATIM)
        list(APPEND LLMX_CUDA_OBJECTS ${fatbin})
    endforeach()
    add_custom_target(llmx-cuda-kernels DEPENDS ${LLMX_CUDA_OBJECTS})
    add_library(llmx-cuda STATIC src/backends/cuda/cuda_backend.cpp)
    add_dependencies(llmx-cuda llmx-cuda-kernels)
    llmx_embed_modules(llmx-cuda cuda llmx_cuda_f32_module ${CMAKE_CURRENT_BINARY_DIR}/generated/cuda/f32.fatbin llmx_cuda_int16_module ${CMAKE_CURRENT_BINARY_DIR}/generated/cuda/int16.fatbin)
    target_include_directories(llmx-cuda
        PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/generated ${CMAKE_CURRENT_SOURCE_DIR}/src
        PRIVATE ${LLMX_CUDA_INCLUDE_DIR})
    target_link_libraries(llmx-cuda PUBLIC ${CMAKE_DL_LIBS})
    target_link_libraries(llmx PRIVATE llmx-cuda)
endif()

include(CTest)
set(LLMX_TARGETS llmx)
if(LLMX_HAS_BACKEND_VULKAN)
    list(APPEND LLMX_TARGETS llmx-vulkan)
endif()
if(LLMX_HAS_BACKEND_ROCM)
    list(APPEND LLMX_TARGETS llmx-rocm)
endif()
if(LLMX_HAS_BACKEND_CUDA)
    list(APPEND LLMX_TARGETS llmx-cuda)
endif()
if(BUILD_TESTING)
    include(cmake/tests.cmake)
endif()

find_package(Threads REQUIRED)
foreach(target IN LISTS LLMX_TARGETS)
    # The x86 build requires AVX2/FMA/F16C, which the CPU kernels use with no runtime check; without them cpu_backend.hpp stops the compile.
    # /bigobj lifts COFF's section count for the one translation unit the runtime compiles in, changing the object format and not the code.
    if(MSVC)
        target_compile_options(${target} PRIVATE /std:c++17 /O2 /arch:AVX2 /bigobj)
        get_target_property(target_kind ${target} TYPE)
        if(target_kind STREQUAL "EXECUTABLE")
            target_sources(${target} PRIVATE cmake/windows.manifest)
        endif()
    else()
        # A multiply-add fuses only where the code writes an FMA, as under MSVC, so no build's bits rest on what its compiler version fuses (docs/BUILD.md).
        target_compile_options(${target} PRIVATE -std=c++17 -O3 -ffp-contract=off)
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
