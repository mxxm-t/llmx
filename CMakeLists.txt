cmake_minimum_required(VERSION 3.16)
project(llmx VERSION 0.1.0 LANGUAGES CXX)

# llmx: a dependency-free GGUF runtime, built on Windows, Linux and macOS.
# The CPU backend is always built; the Vulkan backend is opt-in through LLMX_HAS_BACKEND_VULKAN and needs its SDK, the ROCm backend through LLMX_HAS_BACKEND_ROCM, on Linux, with the ROCm SDK, and the CUDA backend through LLMX_HAS_BACKEND_CUDA, on Linux, with the CUDA toolkit.
# Split mode and node count are runtime parameters, not build options.

option(LLMX_HAS_BACKEND_VULKAN "Build the Vulkan backend" OFF)
option(LLMX_HAS_BACKEND_ROCM "Build the ROCm backend (Linux)" OFF)
option(LLMX_HAS_BACKEND_CUDA "Build the CUDA backend (Linux)" OFF)

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
        quantize_xw matmul_tile_q matmul_reduce matmul_vec_q8 moe_route moe_combine moe_group moe_step_route moe_step_act attention_vec
        sigmoid_mul gated_rms_norm causal_conv_silu dilated_conv_silu_add delta_rule argmax_rows
        group_rms_norm silu_scaled mix_streams inject_streams signed_sqrt_gate
        kv_index_pool kv_index_scores_tile kv_index_select kv_index_select_wide add_bias gelu layer_norm_rows mix_rows hold
        hc_coeffs hc_collapse_norm hc_expand sparse_softmax sparse_columns topk_rows compress_pool indexer_scores indexer_scores_tile pool_write model_format_qdq route_sqrt_softplus)
    # A variant is the same source compiled with a define, "name=source=DEFINE"; the row kernel is one module per family of types, since one module for all of them raised every path's registers.
    set(LLMX_VK_VARIANTS moe_route_s1=moe_route=LLMX_SLOTS1 moe_route_s2=moe_route=LLMX_SLOTS2 moe_route_s4=moe_route=LLMX_SLOTS4 moe_route_s8=moe_route=LLMX_SLOTS8
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
        matmul_tile_mxfp4_bf16=matmul_tile=LLMX_MXFP4+LLMX_PRESERVE_F32+LLMX_BF16 matmul_reduce_mxfp4=matmul_reduce=LLMX_MXFP4+LLMX_PRESERVE_F32
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
        matmul_tile_q_preserve=matmul_tile_q=LLMX_PRESERVE_F32
        matmul_tile_q6_preserve=matmul_tile_q=LLMX_Q6+LLMX_PRESERVE_F32
        matmul_tile_q3=matmul_tile_q=LLMX_Q3
        matmul_tile_q3_preserve=matmul_tile_q=LLMX_Q3+LLMX_PRESERVE_F32
        matmul_tile_q2=matmul_tile_q=LLMX_Q2
        matmul_tile_q2_preserve=matmul_tile_q=LLMX_Q2+LLMX_PRESERVE_F32
        matmul_reduce_preserve=matmul_reduce=LLMX_PRESERVE_F32
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

# The ROCm backend (docs/ROCM.md): the SDK supplies the headers and the compiler at build time and the HIP runtime is opened at run time, so nothing links against libamdhip64.
if(LLMX_HAS_BACKEND_ROCM)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "LLMX_HAS_BACKEND_ROCM is Linux only")
    endif()
    find_path(LLMX_HIP_INCLUDE_DIR hip/hip_runtime_api.h HINTS $ENV{ROCM_PATH}/include /opt/rocm/include)
    find_program(LLMX_HIP_CLANG clang++ HINTS $ENV{ROCM_PATH}/llvm/bin /opt/rocm/llvm/bin NO_DEFAULT_PATH)
    if(NOT LLMX_HIP_INCLUDE_DIR OR NOT LLMX_HIP_CLANG)
        message(FATAL_ERROR "LLMX_HAS_BACKEND_ROCM needs hip/hip_runtime_api.h and the ROCm clang++: install the ROCm SDK, or set ROCM_PATH")
    endif()
    # Kernels are HIP in the tree, compiled to a code object for each target and embedded as byte arrays, as the Vulkan SPIR-V is; gfx1151 is compile coverage until hardware qualifies it.
    # The compiler reads the SDK it was found with, never another HIP installed beside it.
    get_filename_component(LLMX_ROCM_ROOT ${LLMX_HIP_INCLUDE_DIR} DIRECTORY)
    set(LLMX_ROCM_KERNELS f32 int16)
    set(LLMX_ROCM_OBJECTS)
    # Every kernel compiles without contraction, so a build of a body cannot fuse a product another build of it keeps apart; a fused multiply-add is written as fmaf (docs/src/backends-gpu.md).
    set(LLMX_HIP_FLAGS -x hip -std=c++17 -O3 -ffp-contract=off --rocm-path=${LLMX_ROCM_ROOT} --cuda-device-only)
    foreach(kernel IN LISTS LLMX_ROCM_KERNELS)
        set(bundle ${CMAKE_CURRENT_BINARY_DIR}/generated/rocm/${kernel}.hipfb)
        set(inc ${CMAKE_CURRENT_BINARY_DIR}/generated/rocm/${kernel}.inc)
        add_custom_command(OUTPUT ${inc}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/generated/rocm
            COMMAND ${LLMX_HIP_CLANG} ${LLMX_HIP_FLAGS} --offload-arch=gfx906 --offload-arch=gfx1151
                    -I${CMAKE_CURRENT_SOURCE_DIR}/src -c ${LLMX_GPU_KERNEL_DIR}/${kernel}.gpu -o ${bundle}
            COMMAND ${CMAKE_COMMAND} -DIN=${bundle} -DOUT=${inc} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
            DEPENDS ${LLMX_GPU_KERNEL_DIR}/${kernel}.gpu ${LLMX_GPU_KERNEL_DIR}/quant.h ${LLMX_GPU_KERNEL_DIR}/port.h ${CMAKE_CURRENT_SOURCE_DIR}/src/quant/types.hpp ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
            COMMENT "hip ${kernel}"
            VERBATIM)
        list(APPEND LLMX_ROCM_OBJECTS ${inc})
    endforeach()
    # The int16 kernels again, their products on the 16-bit integer dot, for the targets that have the instruction; a device whose code object is missing from it runs the plain module (docs/src/backends-gpu.md).
    set(bundle ${CMAKE_CURRENT_BINARY_DIR}/generated/rocm/int16dot.hipfb)
    set(inc ${CMAKE_CURRENT_BINARY_DIR}/generated/rocm/int16dot.inc)
    add_custom_command(OUTPUT ${inc}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/generated/rocm
        COMMAND ${LLMX_HIP_CLANG} ${LLMX_HIP_FLAGS} --offload-arch=gfx906 -DLLMX_DOT16
                -I${CMAKE_CURRENT_SOURCE_DIR}/src -c ${LLMX_GPU_KERNEL_DIR}/int16.gpu -o ${bundle}
        COMMAND ${CMAKE_COMMAND} -DIN=${bundle} -DOUT=${inc} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
        DEPENDS ${LLMX_GPU_KERNEL_DIR}/int16.gpu ${LLMX_GPU_KERNEL_DIR}/quant.h ${LLMX_GPU_KERNEL_DIR}/port.h ${CMAKE_CURRENT_SOURCE_DIR}/src/quant/types.hpp ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
        COMMENT "hip int16dot"
        VERBATIM)
    list(APPEND LLMX_ROCM_OBJECTS ${inc})
    add_custom_target(llmx-rocm-kernels DEPENDS ${LLMX_ROCM_OBJECTS})
    add_library(llmx-rocm STATIC src/backends/rocm/rocm_backend.cpp)
    add_dependencies(llmx-rocm llmx-rocm-kernels)
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
    # SASS for every target the backend claims and PTX for later ones; 12.8 is the first toolkit with sm_120.
    set(LLMX_CUDA_ARCHS 80 86 89 90 120)
    set(LLMX_CUDA_OBJECTS)
    foreach(kernel f32 int16)
        set(fatbin ${CMAKE_CURRENT_BINARY_DIR}/generated/cuda/${kernel}.fatbin)
        set(inc ${CMAKE_CURRENT_BINARY_DIR}/generated/cuda/${kernel}.inc)
        add_custom_command(OUTPUT ${inc} ${fatbin} ${CMAKE_CURRENT_BINARY_DIR}/generated/cuda/${kernel}.ptxas
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/generated/cuda
            COMMAND ${CMAKE_COMMAND} -DNVCC=${LLMX_NVCC} -DSRC=${LLMX_GPU_KERNEL_DIR}/${kernel}.gpu -DOUT=${fatbin}
                    -DREPORT=${CMAKE_CURRENT_BINARY_DIR}/generated/cuda/${kernel}.ptxas -DINCLUDE=${CMAKE_CURRENT_SOURCE_DIR}/src
                    "-DARCHS=${LLMX_CUDA_ARCHS}" -DPTX=80 -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/nvcc.cmake
            COMMAND ${CMAKE_COMMAND} -DIN=${fatbin} -DOUT=${inc} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
            DEPENDS ${LLMX_GPU_KERNEL_DIR}/${kernel}.gpu ${LLMX_GPU_KERNEL_DIR}/quant.h ${LLMX_GPU_KERNEL_DIR}/port.h ${CMAKE_CURRENT_SOURCE_DIR}/src/quant/types.hpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/nvcc.cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
            COMMENT "nvcc ${kernel}"
            VERBATIM)
        list(APPEND LLMX_CUDA_OBJECTS ${inc})
    endforeach()
    add_custom_target(llmx-cuda-kernels DEPENDS ${LLMX_CUDA_OBJECTS})
    add_library(llmx-cuda STATIC src/backends/cuda/cuda_backend.cpp)
    add_dependencies(llmx-cuda llmx-cuda-kernels)
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
