#pragma once
// The kernel table: the embedded modules, kernel ids, each weight format's kernels, the row and decode builds, and the bindings each kernel reads and writes.
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "backends/backend.hpp"
#include "backends/device_profile.hpp"
#include "backends/vulkan/vulkan_device.hpp"
#include "quant/quant.hpp"

namespace backend {
namespace {

// The kernels, compiled by glslc at build time into the generated include directory as comma-separated words.
const uint32_t kSpvAdd[] = {
#include "vulkan/add.inc"
};
const uint32_t kSpvSumRows[] = {
#include "vulkan/sum_rows.inc"
};
const uint32_t kSpvSiluMul[] = {
#include "vulkan/silu_mul.inc"
};
const uint32_t kSpvGatherRows[] = {
#include "vulkan/gather_rows.inc"
};
const uint32_t kSpvRmsNormRows[] = {
#include "vulkan/rms_norm_rows.inc"
};
const uint32_t kSpvNormRopePartial[] = {
#include "vulkan/norm_rope_partial.inc"
};
const uint32_t kSpvEmbed[] = {
#include "vulkan/embed.inc"
};
const uint32_t kSpvMatmulRow[] = {
#include "vulkan/matmul_row.inc"
};
const uint32_t kSpvMatmulRowF32Only[] = {
#include "vulkan/matmul_row_f32_only.inc"
};
const uint32_t kSpvMatmulRowQ8W[] = {
#include "vulkan/matmul_row_q8w.inc"
};
const uint32_t kSpvMatmulRowQ4[] = {
#include "vulkan/matmul_row_q4.inc"
};
const uint32_t kSpvMatmulRowQ5[] = {
#include "vulkan/matmul_row_q5.inc"
};
const uint32_t kSpvMatmulRowQ50[] = {
#include "vulkan/matmul_row_q50.inc"
};
const uint32_t kSpvMatmulRowK4[] = {
#include "vulkan/matmul_row_k4.inc"
};
const uint32_t kSpvMatmulRowK5[] = {
#include "vulkan/matmul_row_k5.inc"
};
const uint32_t kSpvMatmulRowK[] = {
#include "vulkan/matmul_row_k.inc"
};
const uint32_t kSpvQuantizeX[] = {
#include "vulkan/quantize_x.inc"
};
const uint32_t kSpvKvWrite[] = {
#include "vulkan/kv_write.inc"
};
const uint32_t kSpvNormRopeKv[] = {
#include "vulkan/norm_rope_kv.inc"
};
const uint32_t kSpvAttention[] = {
#include "vulkan/attention.inc"
};
const uint32_t kSpvAttentionMerge[] = {
#include "vulkan/attention_merge.inc"
};
const uint32_t kSpvAttentionTile[] = {
#include "vulkan/attention_tile.inc"
};
// The cache-type variants of every kernel that touches K or V, in kv_variant()'s order: K f16, V f16, both.
const uint32_t kSpvKvWriteK16[] = {
#include "vulkan/kv_write_k16.inc"
};
const uint32_t kSpvKvWriteV16[] = {
#include "vulkan/kv_write_v16.inc"
};
const uint32_t kSpvKvWriteKV16[] = {
#include "vulkan/kv_write_kv16.inc"
};
const uint32_t kSpvAttentionG4[] = {
#include "vulkan/attention_g4.inc"
};
const uint32_t kSpvAttentionK16G4[] = {
#include "vulkan/attention_k16_g4.inc"
};
const uint32_t kSpvAttentionV16G4[] = {
#include "vulkan/attention_v16_g4.inc"
};
const uint32_t kSpvAttentionKV16G4[] = {
#include "vulkan/attention_kv16_g4.inc"
};
const uint32_t kSpvAttentionVec[] = {
#include "vulkan/attention_vec.inc"
};
const uint32_t kSpvAttentionVecK16[] = {
#include "vulkan/attention_vec_k16.inc"
};
const uint32_t kSpvAttentionVecV16[] = {
#include "vulkan/attention_vec_v16.inc"
};
const uint32_t kSpvAttentionVecKV16[] = {
#include "vulkan/attention_vec_kv16.inc"
};
const uint32_t kSpvAttentionVecG4[] = {
#include "vulkan/attention_vec_g4.inc"
};
const uint32_t kSpvAttentionVecK16G4[] = {
#include "vulkan/attention_vec_k16_g4.inc"
};
const uint32_t kSpvAttentionVecV16G4[] = {
#include "vulkan/attention_vec_v16_g4.inc"
};
const uint32_t kSpvAttentionVecKV16G4[] = {
#include "vulkan/attention_vec_kv16_g4.inc"
};
const uint32_t kSpvAttentionK16[] = {
#include "vulkan/attention_k16.inc"
};
const uint32_t kSpvAttentionV16[] = {
#include "vulkan/attention_v16.inc"
};
const uint32_t kSpvAttentionKV16[] = {
#include "vulkan/attention_kv16.inc"
};
const uint32_t kSpvAttentionTileK16[] = {
#include "vulkan/attention_tile_k16.inc"
};
const uint32_t kSpvAttentionTileV16[] = {
#include "vulkan/attention_tile_v16.inc"
};
const uint32_t kSpvAttentionTileKV16[] = {
#include "vulkan/attention_tile_kv16.inc"
};
const uint32_t kSpvNormRopeKvK16[] = {
#include "vulkan/norm_rope_kv_k16.inc"
};
const uint32_t kSpvNormRopeKvV16[] = {
#include "vulkan/norm_rope_kv_v16.inc"
};
const uint32_t kSpvNormRopeKvKV16[] = {
#include "vulkan/norm_rope_kv_kv16.inc"
};

const uint32_t kSpvMatmulRowQ4Dot[] = {
#include "vulkan/matmul_row_q4_dot.inc"
};
const uint32_t kSpvMatmulRowQ5Dot[] = {
#include "vulkan/matmul_row_q5_dot.inc"
};
const uint32_t kSpvMatmulRowQ50Dot[] = {
#include "vulkan/matmul_row_q50_dot.inc"
};
const uint32_t kSpvMatmulRowK4Dot[] = {
#include "vulkan/matmul_row_k4_dot.inc"
};
const uint32_t kSpvMatmulRowK5Dot[] = {
#include "vulkan/matmul_row_k5_dot.inc"
};
const uint32_t kSpvMatmulRowKDot[] = {
#include "vulkan/matmul_row_k_dot.inc"
};

const uint32_t kSpvMatmulTile[] = {
#include "vulkan/matmul_tile.inc"
};
const uint32_t kSpvQuantizeXW[] = {
#include "vulkan/quantize_xw.inc"
};
const uint32_t kSpvMatmulTileQ[] = {
#include "vulkan/matmul_tile_q.inc"
};
const uint32_t kSpvMatmulTileQ5[] = {
#include "vulkan/matmul_tile_q5.inc"
};
const uint32_t kSpvMatmulTileQ50[] = {
#include "vulkan/matmul_tile_q50.inc"
};
const uint32_t kSpvMatmulTileQ5i8[] = {
#include "vulkan/matmul_tile_q5i8.inc"
};
const uint32_t kSpvMatmulTileQ50i8[] = {
#include "vulkan/matmul_tile_q50i8.inc"
};
const uint32_t kSpvMatmulTileQ5i8Preserve[] = {
#include "vulkan/matmul_tile_q5i8_preserve.inc"
};
const uint32_t kSpvMatmulTileQ50i8Preserve[] = {
#include "vulkan/matmul_tile_q50i8_preserve.inc"
};
const uint32_t kSpvMatmulTileQ2[] = {
#include "vulkan/matmul_tile_q2.inc"
};
const uint32_t kSpvMatmulTileQ2Preserve[] = {
#include "vulkan/matmul_tile_q2_preserve.inc"
};
const uint32_t kSpvMatmulTileQ3[] = {
#include "vulkan/matmul_tile_q3.inc"
};
const uint32_t kSpvMatmulTileQ3Preserve[] = {
#include "vulkan/matmul_tile_q3_preserve.inc"
};
const uint32_t kSpvMatmulTileQ6[] = {
#include "vulkan/matmul_tile_q6.inc"
};
const uint32_t kSpvMatmulTileQ8[] = {
#include "vulkan/matmul_tile_q8.inc"
};
const uint32_t kSpvMatmulTileQi8[] = {
#include "vulkan/matmul_tile_qi8.inc"
};
const uint32_t kSpvMatmulTileQi8Preserve[] = {
#include "vulkan/matmul_tile_qi8_preserve.inc"
};
const uint32_t kSpvMatmulTileQ6i8[] = {
#include "vulkan/matmul_tile_q6i8.inc"
};
const uint32_t kSpvMatmulTileQ6i8Preserve[] = {
#include "vulkan/matmul_tile_q6i8_preserve.inc"
};
const uint32_t kSpvMatmulTileQ8i8[] = {
#include "vulkan/matmul_tile_q8i8.inc"
};
const uint32_t kSpvMatmulReduce[] = {
#include "vulkan/matmul_reduce.inc"
};
const uint32_t kSpvMatmulVecQ8[] = {
#include "vulkan/matmul_vec_q8.inc"
};
const uint32_t kSpvMatmulTileBf16[] = {
#include "vulkan/matmul_tile_bf16.inc"
};
const uint32_t kSpvMatmulTileBf16Preserve[] = {
#include "vulkan/matmul_tile_bf16_preserve.inc"
};
const uint32_t kSpvMatmulTileTq10Dot[] = {
#include "vulkan/matmul_tile_tq1_0_dot.inc"
};
const uint32_t kSpvMatmulTileTq20Dot[] = {
#include "vulkan/matmul_tile_tq2_0_dot.inc"
};
const uint32_t kSpvMatmulRowTq10Dot[] = {
#include "vulkan/matmul_row_tq1_0_dot.inc"
};
const uint32_t kSpvMatmulRowTq20Dot[] = {
#include "vulkan/matmul_row_tq2_0_dot.inc"
};
const uint32_t kSpvMatmulRowTq10[] = {
#include "vulkan/matmul_row_tq1_0.inc"
};
const uint32_t kSpvMatmulRowTq20[] = {
#include "vulkan/matmul_row_tq2_0.inc"
};
const uint32_t kSpvMatmulRowTq10Preserve[] = {
#include "vulkan/matmul_row_tq1_0_preserve.inc"
};
const uint32_t kSpvMatmulRowTq20Preserve[] = {
#include "vulkan/matmul_row_tq2_0_preserve.inc"
};
const uint32_t kSpvMatmulTileTq10[] = {
#include "vulkan/matmul_tile_tq1_0.inc"
};
const uint32_t kSpvMatmulTileTq20[] = {
#include "vulkan/matmul_tile_tq2_0.inc"
};
const uint32_t kSpvMatmulTileTq10Preserve[] = {
#include "vulkan/matmul_tile_tq1_0_preserve.inc"
};
const uint32_t kSpvMatmulTileTq20Preserve[] = {
#include "vulkan/matmul_tile_tq2_0_preserve.inc"
};
const uint32_t kSpvMatmulTileTq10Bf16[] = {
#include "vulkan/matmul_tile_tq1_0_bf16.inc"
};
const uint32_t kSpvMatmulTileTq20Bf16[] = {
#include "vulkan/matmul_tile_tq2_0_bf16.inc"
};
const uint32_t kSpvMatmulTileTq10Bf16Preserve[] = {
#include "vulkan/matmul_tile_tq1_0_bf16_preserve.inc"
};
const uint32_t kSpvMatmulTileTq20Bf16Preserve[] = {
#include "vulkan/matmul_tile_tq2_0_bf16_preserve.inc"
};
const uint32_t kSpvMatmulTileIq1SDot[] = {
#include "vulkan/matmul_tile_iq1_s_dot.inc"
};
const uint32_t kSpvMatmulTileIq1MDot[] = {
#include "vulkan/matmul_tile_iq1_m_dot.inc"
};
const uint32_t kSpvMatmulTileIq2SDot[] = {
#include "vulkan/matmul_tile_iq2_s_dot.inc"
};
const uint32_t kSpvMatmulTileIq3XxsDot[] = {
#include "vulkan/matmul_tile_iq3_xxs_dot.inc"
};
const uint32_t kSpvMatmulTileIq2XxsDot[] = {
#include "vulkan/matmul_tile_iq2_xxs_dot.inc"
};
const uint32_t kSpvMatmulTileIq2XsDot[] = {
#include "vulkan/matmul_tile_iq2_xs_dot.inc"
};
const uint32_t kSpvMatmulTileIq3SDot[] = {
#include "vulkan/matmul_tile_iq3_s_dot.inc"
};
const uint32_t kSpvMatmulRowIq1SDot[] = {
#include "vulkan/matmul_row_iq1_s_dot.inc"
};
const uint32_t kSpvMatmulRowIq1MDot[] = {
#include "vulkan/matmul_row_iq1_m_dot.inc"
};
const uint32_t kSpvMatmulRowIq2SDot[] = {
#include "vulkan/matmul_row_iq2_s_dot.inc"
};
const uint32_t kSpvMatmulRowIq3XxsDot[] = {
#include "vulkan/matmul_row_iq3_xxs_dot.inc"
};
const uint32_t kSpvMatmulRowIq2XxsDot[] = {
#include "vulkan/matmul_row_iq2_xxs_dot.inc"
};
const uint32_t kSpvMatmulRowIq2XsDot[] = {
#include "vulkan/matmul_row_iq2_xs_dot.inc"
};
const uint32_t kSpvMatmulRowIq3SDot[] = {
#include "vulkan/matmul_row_iq3_s_dot.inc"
};
const uint32_t kSpvMatmulRowIq1S[] = {
#include "vulkan/matmul_row_iq1_s.inc"
};
const uint32_t kSpvMatmulRowIq1M[] = {
#include "vulkan/matmul_row_iq1_m.inc"
};
const uint32_t kSpvMatmulRowIq2S[] = {
#include "vulkan/matmul_row_iq2_s.inc"
};
const uint32_t kSpvMatmulRowIq3Xxs[] = {
#include "vulkan/matmul_row_iq3_xxs.inc"
};
const uint32_t kSpvMatmulRowIq2Xxs[] = {
#include "vulkan/matmul_row_iq2_xxs.inc"
};
const uint32_t kSpvMatmulRowIq2Xs[] = {
#include "vulkan/matmul_row_iq2_xs.inc"
};
const uint32_t kSpvMatmulRowIq1SPreserve[] = {
#include "vulkan/matmul_row_iq1_s_preserve.inc"
};
const uint32_t kSpvMatmulRowIq1MPreserve[] = {
#include "vulkan/matmul_row_iq1_m_preserve.inc"
};
const uint32_t kSpvMatmulRowIq2SPreserve[] = {
#include "vulkan/matmul_row_iq2_s_preserve.inc"
};
const uint32_t kSpvMatmulRowIq3XxsPreserve[] = {
#include "vulkan/matmul_row_iq3_xxs_preserve.inc"
};
const uint32_t kSpvMatmulRowIq2XxsPreserve[] = {
#include "vulkan/matmul_row_iq2_xxs_preserve.inc"
};
const uint32_t kSpvMatmulRowIq2XsPreserve[] = {
#include "vulkan/matmul_row_iq2_xs_preserve.inc"
};
const uint32_t kSpvMatmulTileIq1S[] = {
#include "vulkan/matmul_tile_iq1_s.inc"
};
const uint32_t kSpvMatmulTileIq1M[] = {
#include "vulkan/matmul_tile_iq1_m.inc"
};
const uint32_t kSpvMatmulTileIq2S[] = {
#include "vulkan/matmul_tile_iq2_s.inc"
};
const uint32_t kSpvMatmulTileIq3Xxs[] = {
#include "vulkan/matmul_tile_iq3_xxs.inc"
};
const uint32_t kSpvMatmulTileIq2Xxs[] = {
#include "vulkan/matmul_tile_iq2_xxs.inc"
};
const uint32_t kSpvMatmulTileIq2Xs[] = {
#include "vulkan/matmul_tile_iq2_xs.inc"
};
const uint32_t kSpvMatmulTileIq1SPreserve[] = {
#include "vulkan/matmul_tile_iq1_s_preserve.inc"
};
const uint32_t kSpvMatmulTileIq1MPreserve[] = {
#include "vulkan/matmul_tile_iq1_m_preserve.inc"
};
const uint32_t kSpvMatmulTileIq2SPreserve[] = {
#include "vulkan/matmul_tile_iq2_s_preserve.inc"
};
const uint32_t kSpvMatmulTileIq3XxsPreserve[] = {
#include "vulkan/matmul_tile_iq3_xxs_preserve.inc"
};
const uint32_t kSpvMatmulTileIq2XxsPreserve[] = {
#include "vulkan/matmul_tile_iq2_xxs_preserve.inc"
};
const uint32_t kSpvMatmulTileIq2XsPreserve[] = {
#include "vulkan/matmul_tile_iq2_xs_preserve.inc"
};
const uint32_t kSpvMatmulTileIq1SBf16[] = {
#include "vulkan/matmul_tile_iq1_s_bf16.inc"
};
const uint32_t kSpvMatmulTileIq1MBf16[] = {
#include "vulkan/matmul_tile_iq1_m_bf16.inc"
};
const uint32_t kSpvMatmulTileIq2SBf16[] = {
#include "vulkan/matmul_tile_iq2_s_bf16.inc"
};
const uint32_t kSpvMatmulTileIq3XxsBf16[] = {
#include "vulkan/matmul_tile_iq3_xxs_bf16.inc"
};
const uint32_t kSpvMatmulTileIq2XxsBf16[] = {
#include "vulkan/matmul_tile_iq2_xxs_bf16.inc"
};
const uint32_t kSpvMatmulTileIq2XsBf16[] = {
#include "vulkan/matmul_tile_iq2_xs_bf16.inc"
};
const uint32_t kSpvMatmulTileIq1SBf16Preserve[] = {
#include "vulkan/matmul_tile_iq1_s_bf16_preserve.inc"
};
const uint32_t kSpvMatmulTileIq1MBf16Preserve[] = {
#include "vulkan/matmul_tile_iq1_m_bf16_preserve.inc"
};
const uint32_t kSpvMatmulTileIq2SBf16Preserve[] = {
#include "vulkan/matmul_tile_iq2_s_bf16_preserve.inc"
};
const uint32_t kSpvMatmulTileIq3XxsBf16Preserve[] = {
#include "vulkan/matmul_tile_iq3_xxs_bf16_preserve.inc"
};
const uint32_t kSpvMatmulTileIq2XxsBf16Preserve[] = {
#include "vulkan/matmul_tile_iq2_xxs_bf16_preserve.inc"
};
const uint32_t kSpvMatmulTileIq2XsBf16Preserve[] = {
#include "vulkan/matmul_tile_iq2_xs_bf16_preserve.inc"
};
const uint32_t kSpvMatmulRowIq3S[] = {
#include "vulkan/matmul_row_iq3_s.inc"
};
const uint32_t kSpvMatmulRowIq3SPreserve[] = {
#include "vulkan/matmul_row_iq3_s_preserve.inc"
};
const uint32_t kSpvMatmulTileIq3S[] = {
#include "vulkan/matmul_tile_iq3_s.inc"
};
const uint32_t kSpvMatmulTileIq3SPreserve[] = {
#include "vulkan/matmul_tile_iq3_s_preserve.inc"
};
const uint32_t kSpvMatmulTileIq3SBf16[] = {
#include "vulkan/matmul_tile_iq3_s_bf16.inc"
};
const uint32_t kSpvMatmulTileIq3SBf16Preserve[] = {
#include "vulkan/matmul_tile_iq3_s_bf16_preserve.inc"
};
const uint32_t kSpvMatmulRowFloatX[] = {
#include "vulkan/matmul_row_float_x.inc"
};
const uint32_t kSpvMatmulRowFloatXPreserve[] = {
#include "vulkan/matmul_row_float_x_preserve.inc"
};
const uint32_t kSpvMatmulRowFloatXQ2[] = {
#include "vulkan/matmul_row_float_x_q2.inc"
};
const uint32_t kSpvMatmulRowFloatXQ2Preserve[] = {
#include "vulkan/matmul_row_float_x_q2_preserve.inc"
};
const uint32_t kSpvMatmulRowFloatXQ3[] = {
#include "vulkan/matmul_row_float_x_q3.inc"
};
const uint32_t kSpvMatmulRowFloatXQ3Preserve[] = {
#include "vulkan/matmul_row_float_x_q3_preserve.inc"
};
const uint32_t kSpvMatmulRowQ2Dot[] = {
#include "vulkan/matmul_row_q2_dot.inc"
};
const uint32_t kSpvMatmulRowQ2DotPreserve[] = {
#include "vulkan/matmul_row_q2_dot_preserve.inc"
};
const uint32_t kSpvMatmulRowQ3Dot[] = {
#include "vulkan/matmul_row_q3_dot.inc"
};
const uint32_t kSpvMatmulRowQ3DotPreserve[] = {
#include "vulkan/matmul_row_q3_dot_preserve.inc"
};
const uint32_t kSpvMatmulRowPreserve[] = {
#include "vulkan/matmul_row_preserve.inc"
};
const uint32_t kSpvMatmulRowQ8wPreserve[] = {
#include "vulkan/matmul_row_q8w_preserve.inc"
};
const uint32_t kSpvMatmulVecQ8Preserve[] = {
#include "vulkan/matmul_vec_q8_preserve.inc"
};
const uint32_t kSpvSiluScaledPreserve[] = {
#include "vulkan/silu_scaled_preserve.inc"
};
const uint32_t kSpvMixStreamsPreserve[] = {
#include "vulkan/mix_streams_preserve.inc"
};
const uint32_t kSpvInjectStreamsPreserve[] = {
#include "vulkan/inject_streams_preserve.inc"
};
const uint32_t kSpvMatmulVecQ8I8[] = {
#include "vulkan/matmul_vec_q8_i8.inc"
};
const uint32_t kSpvMatmulVecQ8I8Preserve[] = {
#include "vulkan/matmul_vec_q8_i8_preserve.inc"
};
const uint32_t kSpvMatmulRowQ4I8[] = {
#include "vulkan/matmul_row_q4_i8.inc"
};
const uint32_t kSpvMatmulRowQ5I8[] = {
#include "vulkan/matmul_row_q5_i8.inc"
};
const uint32_t kSpvMatmulRowQ50I8[] = {
#include "vulkan/matmul_row_q50_i8.inc"
};
const uint32_t kSpvMatmulRowQ4I8Preserve[] = {
#include "vulkan/matmul_row_q4_i8_preserve.inc"
};
const uint32_t kSpvMatmulRowQ5I8Preserve[] = {
#include "vulkan/matmul_row_q5_i8_preserve.inc"
};
const uint32_t kSpvMatmulRowQ50I8Preserve[] = {
#include "vulkan/matmul_row_q50_i8_preserve.inc"
};
const uint32_t kSpvMatmulRowK4I8[] = {
#include "vulkan/matmul_row_k4_i8.inc"
};
const uint32_t kSpvMatmulRowK4I8Preserve[] = {
#include "vulkan/matmul_row_k4_i8_preserve.inc"
};
const uint32_t kSpvMatmulRowK5I8[] = {
#include "vulkan/matmul_row_k5_i8.inc"
};
const uint32_t kSpvMatmulRowK5I8Preserve[] = {
#include "vulkan/matmul_row_k5_i8_preserve.inc"
};
const uint32_t kSpvMatmulRowKI8[] = {
#include "vulkan/matmul_row_k_i8.inc"
};
const uint32_t kSpvMatmulRowKI8Preserve[] = {
#include "vulkan/matmul_row_k_i8_preserve.inc"
};
const uint32_t kSpvMatmulRowQ4Preserve[] = {
#include "vulkan/matmul_row_q4_preserve.inc"
};
const uint32_t kSpvMatmulRowQ5Preserve[] = {
#include "vulkan/matmul_row_q5_preserve.inc"
};
const uint32_t kSpvMatmulRowQ50Preserve[] = {
#include "vulkan/matmul_row_q50_preserve.inc"
};
const uint32_t kSpvMatmulRowQ4DotPreserve[] = {
#include "vulkan/matmul_row_q4_dot_preserve.inc"
};
const uint32_t kSpvMatmulRowQ5DotPreserve[] = {
#include "vulkan/matmul_row_q5_dot_preserve.inc"
};
const uint32_t kSpvMatmulRowQ50DotPreserve[] = {
#include "vulkan/matmul_row_q50_dot_preserve.inc"
};
const uint32_t kSpvMatmulRowK4DotPreserve[] = {
#include "vulkan/matmul_row_k4_dot_preserve.inc"
};
const uint32_t kSpvMatmulRowK5Preserve[] = {
#include "vulkan/matmul_row_k5_preserve.inc"
};
const uint32_t kSpvMatmulRowK5DotPreserve[] = {
#include "vulkan/matmul_row_k5_dot_preserve.inc"
};
const uint32_t kSpvMatmulRowKPreserve[] = {
#include "vulkan/matmul_row_k_preserve.inc"
};
const uint32_t kSpvMatmulRowKDotPreserve[] = {
#include "vulkan/matmul_row_k_dot_preserve.inc"
};
const uint32_t kSpvMatmulTilePreserve[] = {
#include "vulkan/matmul_tile_preserve.inc"
};
const uint32_t kSpvMatmulTileQPreserve[] = {
#include "vulkan/matmul_tile_q_preserve.inc"
};
const uint32_t kSpvMatmulTileQ5Preserve[] = {
#include "vulkan/matmul_tile_q5_preserve.inc"
};
const uint32_t kSpvMatmulTileQ50Preserve[] = {
#include "vulkan/matmul_tile_q50_preserve.inc"
};
const uint32_t kSpvMatmulTileQ6Preserve[] = {
#include "vulkan/matmul_tile_q6_preserve.inc"
};
const uint32_t kSpvMatmulReducePreserve[] = {
#include "vulkan/matmul_reduce_preserve.inc"
};
const uint32_t kSpvMoeRoute[] = {
#include "vulkan/moe_route.inc"
};
const uint32_t kSpvMoeRouteS1[] = {
#include "vulkan/moe_route_s1.inc"
};
const uint32_t kSpvMoeRouteS2[] = {
#include "vulkan/moe_route_s2.inc"
};
const uint32_t kSpvMoeRouteS4[] = {
#include "vulkan/moe_route_s4.inc"
};
const uint32_t kSpvMoeRouteS8[] = {
#include "vulkan/moe_route_s8.inc"
};
const uint32_t kSpvMoeCombine[] = {
#include "vulkan/moe_combine.inc"
};
const uint32_t kSpvMoeCombinePreserve[] = {
#include "vulkan/moe_combine_preserve.inc"
};
const uint32_t kSpvMoeGroup[] = {
#include "vulkan/moe_group.inc"
};
// The fused routed step's kernels (vulkan_moe.hpp).
const uint32_t kSpvMoeStepRouteS1[] = {
#include "vulkan/moe_step_route_s1.inc"
};
const uint32_t kSpvMoeStepRouteS2[] = {
#include "vulkan/moe_step_route_s2.inc"
};
const uint32_t kSpvMoeStepRouteS4[] = {
#include "vulkan/moe_step_route_s4.inc"
};
const uint32_t kSpvMoeStepRouteS8[] = {
#include "vulkan/moe_step_route_s8.inc"
};
const uint32_t kSpvMoeStepRoute[] = {
#include "vulkan/moe_step_route.inc"
};
const uint32_t kSpvMoeStepAct[] = {
#include "vulkan/moe_step_act.inc"
};
const uint32_t kSpvMoeStepUpK4q8[] = {
#include "vulkan/moe_step_up_k4q8.inc"
};
const uint32_t kSpvMoeStepUpK4q8Preserve[] = {
#include "vulkan/moe_step_up_k4q8_preserve.inc"
};
const uint32_t kSpvMoeStepUpK4q8I8[] = {
#include "vulkan/moe_step_up_k4q8_i8.inc"
};
const uint32_t kSpvMoeStepUpK4q8I8Preserve[] = {
#include "vulkan/moe_step_up_k4q8_i8_preserve.inc"
};
const uint32_t kSpvMoeStepUpK4[] = {
#include "vulkan/moe_step_up_k4.inc"
};
const uint32_t kSpvMoeStepUpK4Preserve[] = {
#include "vulkan/moe_step_up_k4_preserve.inc"
};
const uint32_t kSpvMoeStepUpK4I8[] = {
#include "vulkan/moe_step_up_k4_i8.inc"
};
const uint32_t kSpvMoeStepUpK4I8Preserve[] = {
#include "vulkan/moe_step_up_k4_i8_preserve.inc"
};
const uint32_t kSpvMoeStepDownQ8[] = {
#include "vulkan/moe_step_down_q8.inc"
};
const uint32_t kSpvMoeStepDownQ8Preserve[] = {
#include "vulkan/moe_step_down_q8_preserve.inc"
};
const uint32_t kSpvMoeStepDownQ8I8[] = {
#include "vulkan/moe_step_down_q8_i8.inc"
};
const uint32_t kSpvMoeStepDownQ8I8Preserve[] = {
#include "vulkan/moe_step_down_q8_i8_preserve.inc"
};
const uint32_t kSpvMoeStepDownQ5q8[] = {
#include "vulkan/moe_step_down_q5q8.inc"
};
const uint32_t kSpvMoeStepDownQ5q8Preserve[] = {
#include "vulkan/moe_step_down_q5q8_preserve.inc"
};
const uint32_t kSpvMoeStepDownQ5q8I8[] = {
#include "vulkan/moe_step_down_q5q8_i8.inc"
};
const uint32_t kSpvMoeStepDownQ5q8I8Preserve[] = {
#include "vulkan/moe_step_down_q5q8_i8_preserve.inc"
};
const uint32_t kSpvMoeStepDownK[] = {
#include "vulkan/moe_step_down_k.inc"
};
const uint32_t kSpvMoeStepDownKPreserve[] = {
#include "vulkan/moe_step_down_k_preserve.inc"
};
const uint32_t kSpvMoeStepDownKI8[] = {
#include "vulkan/moe_step_down_k_i8.inc"
};
const uint32_t kSpvMoeStepDownKI8Preserve[] = {
#include "vulkan/moe_step_down_k_i8_preserve.inc"
};
// The attention kernels at head width 256, qwen35's, in the same cache-type order.
const uint32_t kSpvAttentionTileD256[] = {
#include "vulkan/attention_tile_d256.inc"
};
const uint32_t kSpvAttentionTileD256K16[] = {
#include "vulkan/attention_tile_d256_k16.inc"
};
const uint32_t kSpvAttentionTileD256V16[] = {
#include "vulkan/attention_tile_d256_v16.inc"
};
const uint32_t kSpvAttentionTileD256KV16[] = {
#include "vulkan/attention_tile_d256_kv16.inc"
};
const uint32_t kSpvAttentionVecD256[] = {
#include "vulkan/attention_vec_d256.inc"
};
const uint32_t kSpvAttentionVecD256K16[] = {
#include "vulkan/attention_vec_d256_k16.inc"
};
const uint32_t kSpvAttentionVecD256V16[] = {
#include "vulkan/attention_vec_d256_v16.inc"
};
const uint32_t kSpvAttentionVecD256KV16[] = {
#include "vulkan/attention_vec_d256_kv16.inc"
};
const uint32_t kSpvAttentionVecD256G4[] = {
#include "vulkan/attention_vec_d256_g4.inc"
};
const uint32_t kSpvAttentionVecD256K16G4[] = {
#include "vulkan/attention_vec_d256_k16_g4.inc"
};
const uint32_t kSpvAttentionVecD256V16G4[] = {
#include "vulkan/attention_vec_d256_v16_g4.inc"
};
const uint32_t kSpvAttentionVecD256KV16G4[] = {
#include "vulkan/attention_vec_d256_kv16_g4.inc"
};
// The qwen35 layers' ops (docs/QWEN35.md).
const uint32_t kSpvSigmoidMul[] = {
#include "vulkan/sigmoid_mul.inc"
};
const uint32_t kSpvGatedRmsNorm[] = {
#include "vulkan/gated_rms_norm.inc"
};
const uint32_t kSpvCausalConvSilu[] = {
#include "vulkan/causal_conv_silu.inc"
};
const uint32_t kSpvDilatedConvSiluAdd[] = {
#include "vulkan/dilated_conv_silu_add.inc"
};
const uint32_t kSpvDeltaRule[] = {
#include "vulkan/delta_rule.inc"
};
// The embedded drafter's ops (docs/SPECULATIVE.md, section 7).
const uint32_t kSpvArgmaxRows[] = {
#include "vulkan/argmax_rows.inc"
};
const uint32_t kSpvEmbedIds[] = {
#include "vulkan/embed_ids.inc"
};
const uint32_t kSpvEmbedIdsMxfp4[] = {
#include "vulkan/embed_ids_mxfp4.inc"
};
const uint32_t kSpvDeltaRuleShort[] = {
#include "vulkan/delta_rule_short.inc"
};
const uint32_t kSpvGroupRmsNorm[] = {
#include "vulkan/group_rms_norm.inc"
};
const uint32_t kSpvSiluScaled[] = {
#include "vulkan/silu_scaled.inc"
};
const uint32_t kSpvMixStreams[] = {
#include "vulkan/mix_streams.inc"
};
const uint32_t kSpvInjectStreams[] = {
#include "vulkan/inject_streams.inc"
};
const uint32_t kSpvSignedSqrtGate[] = {
#include "vulkan/signed_sqrt_gate.inc"
};

const uint32_t kSpvEmbedMxfp4[] = {
#include "vulkan/embed_mxfp4.inc"
};
const uint32_t kSpvMatmulTileMxfp4[] = {
#include "vulkan/matmul_tile_mxfp4.inc"
};
const uint32_t kSpvMatmulTileMxfp4Bf16[] = {
#include "vulkan/matmul_tile_mxfp4_bf16.inc"
};
const uint32_t kSpvMatmulReduceMxfp4[] = {
#include "vulkan/matmul_reduce_mxfp4.inc"
};

const uint32_t kSpvCopyMxfp4[] = {
#include "vulkan/copy_mxfp4.inc"
};
const uint32_t kSpvMatmulRowMxfp4[] = {
#include "vulkan/matmul_row_mxfp4.inc"
};
const uint32_t kSpvMatmulRowMxfp4Dot[] = {
#include "vulkan/matmul_row_mxfp4_dot.inc"
};
const uint32_t kSpvMatmulRowMxfp4FloatX[] = {
#include "vulkan/matmul_row_mxfp4_float_x.inc"
};
const uint32_t kSpvMatmulTileQ8mx[] = {
#include "vulkan/matmul_tile_q8mx.inc"
};
const uint32_t kSpvKvIndexPool[] = {
#include "vulkan/kv_index_pool.inc"
};
const uint32_t kSpvKvIndexScoresTile[] = {
#include "vulkan/kv_index_scores_tile.inc"
};
const uint32_t kSpvKvIndexSelect[] = {
#include "vulkan/kv_index_select.inc"
};
const uint32_t kSpvKvIndexSelectWide[] = {
#include "vulkan/kv_index_select_wide.inc"
};

const uint32_t kSpvSparseAttention[] = {
#include "vulkan/sparse_attention.inc"
};
const uint32_t kSpvSparseAttentionK16[] = {
#include "vulkan/sparse_attention_k16.inc"
};
const uint32_t kSpvSparseAttentionV16[] = {
#include "vulkan/sparse_attention_v16.inc"
};
const uint32_t kSpvSparseAttentionKv16[] = {
#include "vulkan/sparse_attention_kv16.inc"
};
const uint32_t kSpvSparseAttentionVec[] = {
#include "vulkan/sparse_attention_vec.inc"
};
const uint32_t kSpvSparseAttentionVecK16[] = {
#include "vulkan/sparse_attention_vec_k16.inc"
};
const uint32_t kSpvSparseAttentionVecV16[] = {
#include "vulkan/sparse_attention_vec_v16.inc"
};
const uint32_t kSpvSparseAttentionVecKv16[] = {
#include "vulkan/sparse_attention_vec_kv16.inc"
};
const uint32_t kSpvSparseAttentionVecD256[] = {
#include "vulkan/sparse_attention_vec_d256.inc"
};
const uint32_t kSpvSparseAttentionVecD256K16[] = {
#include "vulkan/sparse_attention_vec_d256_k16.inc"
};
const uint32_t kSpvSparseAttentionVecD256V16[] = {
#include "vulkan/sparse_attention_vec_d256_v16.inc"
};
const uint32_t kSpvSparseAttentionVecD256Kv16[] = {
#include "vulkan/sparse_attention_vec_d256_kv16.inc"
};

const uint32_t kSpvEmbedWf16[] = {
#include "vulkan/embed_wf16.inc"
};
const uint32_t kSpvEmbedIdsWf16[] = {
#include "vulkan/embed_ids_wf16.inc"
};
const uint32_t kSpvMatmulRowWf16[] = {
#include "vulkan/matmul_row_wf16.inc"
};
const uint32_t kSpvMatmulTileWf16[] = {
#include "vulkan/matmul_tile_wf16.inc"
};
const uint32_t kSpvMatmulTileWf16Preserve[] = {
#include "vulkan/matmul_tile_wf16_preserve.inc"
};
const uint32_t kSpvMatmulTileWf16Bf16[] = {
#include "vulkan/matmul_tile_wf16_bf16.inc"
};
const uint32_t kSpvMatmulTileWf16Bf16Preserve[] = {
#include "vulkan/matmul_tile_wf16_bf16_preserve.inc"
};
const uint32_t kSpvEmbedWbf16[] = {
#include "vulkan/embed_wbf16.inc"
};
const uint32_t kSpvEmbedIdsWbf16[] = {
#include "vulkan/embed_ids_wbf16.inc"
};
const uint32_t kSpvMatmulRowWbf16[] = {
#include "vulkan/matmul_row_wbf16.inc"
};
const uint32_t kSpvMatmulTileWbf16[] = {
#include "vulkan/matmul_tile_wbf16.inc"
};
const uint32_t kSpvMatmulTileWbf16Preserve[] = {
#include "vulkan/matmul_tile_wbf16_preserve.inc"
};
const uint32_t kSpvMatmulTileWbf16Bf16[] = {
#include "vulkan/matmul_tile_wbf16_bf16.inc"
};
const uint32_t kSpvMatmulTileWbf16Bf16Preserve[] = {
#include "vulkan/matmul_tile_wbf16_bf16_preserve.inc"
};

const uint32_t kSpvMatmulRowIq4Dot[] = {
#include "vulkan/matmul_row_iq4_dot.inc"
};
const uint32_t kSpvMatmulRowIq4XsDot[] = {
#include "vulkan/matmul_row_iq4_xs_dot.inc"
};

const uint32_t kSpvMatmulTileIq4[] = {
#include "vulkan/matmul_tile_iq4.inc"
};
const uint32_t kSpvMatmulTileIq4Xs[] = {
#include "vulkan/matmul_tile_iq4_xs.inc"
};
// The vision encoder's ops (docs/VISION.md, section 3).
const uint32_t kSpvAddBias[] = {
#include "vulkan/add_bias.inc"
};
const uint32_t kSpvGelu[] = {
#include "vulkan/gelu.inc"
};
const uint32_t kSpvLayerNormRows[] = {
#include "vulkan/layer_norm_rows.inc"
};
const uint32_t kSpvMixRows[] = {
#include "vulkan/mix_rows.inc"
};
const uint32_t kSpvNormRopePartialStreams[] = {
#include "vulkan/norm_rope_partial_streams.inc"
};
const uint32_t kSpvAttentionFull[] = {
#include "vulkan/attention_full.inc"
};
const uint32_t kSpvAttentionFullTileD64[] = {
#include "vulkan/attention_full_tile_d64.inc"
};
const uint32_t kSpvAttentionFullTileD72[] = {
#include "vulkan/attention_full_tile_d72.inc"
};

// IDs, diagnostic names and module bindings share this order; cache variants rely on adjacent IDs.
#define LLMX_VULKAN_KERNELS(X) \
    X(K_ADD, "add", kSpvAdd, sizeof(kSpvAdd), 2, nullptr) \
    X(K_SUM_ROWS, "sum_rows", kSpvSumRows, sizeof(kSpvSumRows), 6, nullptr) \
    X(K_SILU_MUL, "silu_mul", kSpvSiluMul, sizeof(kSpvSiluMul), 4, nullptr) \
    X(K_GATHER_ROWS, "gather_rows", kSpvGatherRows, sizeof(kSpvGatherRows), 3, nullptr) \
    X(K_RMS_NORM_ROWS, "rms_norm_rows", kSpvRmsNormRows, sizeof(kSpvRmsNormRows), 4, nullptr) \
    X(K_NORM_ROPE_PARTIAL, "norm_rope_partial", kSpvNormRopePartial, sizeof(kSpvNormRopePartial), 6, nullptr) \
    X(K_EMBED, "embed", kSpvEmbed, sizeof(kSpvEmbed), 4, nullptr) \
    X(K_MATMUL_ROW, "matmul_row", kSpvMatmulRow, sizeof(kSpvMatmulRow), 12, kMatmulRowCounts, kSpvMatmulRowPreserve, sizeof(kSpvMatmulRowPreserve), 0, kPreserve32) \
    X(K_KV_WRITE, "kv_write", kSpvKvWrite, sizeof(kSpvKvWrite), 5, nullptr) \
    X(K_ATTENTION, "attention", kSpvAttention, sizeof(kSpvAttention), 7, nullptr) \
    X(K_SPARSE_ATTENTION, "sparse_attention", kSpvSparseAttention, sizeof(kSpvSparseAttention), 8, nullptr) \
    X(K_SPARSE_ATTENTION_K16, "sparse_attention_k16", kSpvSparseAttentionK16, sizeof(kSpvSparseAttentionK16), 8, nullptr) \
    X(K_SPARSE_ATTENTION_V16, "sparse_attention_v16", kSpvSparseAttentionV16, sizeof(kSpvSparseAttentionV16), 8, nullptr) \
    X(K_SPARSE_ATTENTION_KV16, "sparse_attention_kv16", kSpvSparseAttentionKv16, sizeof(kSpvSparseAttentionKv16), 8, nullptr) \
    X(K_SPARSE_ATTENTION_VEC, "sparse_attention_vec", kSpvSparseAttentionVec, sizeof(kSpvSparseAttentionVec), 8, nullptr) \
    X(K_SPARSE_ATTENTION_VEC_K16, "sparse_attention_vec_k16", kSpvSparseAttentionVecK16, sizeof(kSpvSparseAttentionVecK16), 8, nullptr) \
    X(K_SPARSE_ATTENTION_VEC_V16, "sparse_attention_vec_v16", kSpvSparseAttentionVecV16, sizeof(kSpvSparseAttentionVecV16), 8, nullptr) \
    X(K_SPARSE_ATTENTION_VEC_KV16, "sparse_attention_vec_kv16", kSpvSparseAttentionVecKv16, sizeof(kSpvSparseAttentionVecKv16), 8, nullptr) \
    X(K_SPARSE_ATTENTION_VEC_D256, "sparse_attention_vec_d256", kSpvSparseAttentionVecD256, sizeof(kSpvSparseAttentionVecD256), 8, nullptr) \
    X(K_SPARSE_ATTENTION_VEC_D256_K16, "sparse_attention_vec_d256_k16", kSpvSparseAttentionVecD256K16, sizeof(kSpvSparseAttentionVecD256K16), 8, nullptr) \
    X(K_SPARSE_ATTENTION_VEC_D256_V16, "sparse_attention_vec_d256_v16", kSpvSparseAttentionVecD256V16, sizeof(kSpvSparseAttentionVecD256V16), 8, nullptr) \
    X(K_SPARSE_ATTENTION_VEC_D256_KV16, "sparse_attention_vec_d256_kv16", kSpvSparseAttentionVecD256Kv16, sizeof(kSpvSparseAttentionVecD256Kv16), 8, nullptr) \
    X(K_ATTENTION_MERGE, "attention_merge", kSpvAttentionMerge, sizeof(kSpvAttentionMerge), 4, nullptr) \
    X(K_KV_INDEX_POOL, "kv_index_pool", kSpvKvIndexPool, sizeof(kSpvKvIndexPool), 7, nullptr) \
    X(K_KV_INDEX_SCORES_TILE, "kv_index_scores_tile", kSpvKvIndexScoresTile, sizeof(kSpvKvIndexScoresTile), 4, nullptr) \
    X(K_KV_INDEX_SELECT, "kv_index_select", kSpvKvIndexSelect, sizeof(kSpvKvIndexSelect), 3, nullptr) \
    X(K_KV_INDEX_SELECT_WIDE, "kv_index_select_wide", kSpvKvIndexSelectWide, sizeof(kSpvKvIndexSelectWide), 4, nullptr) \
    X(K_MATMUL_TILE, "matmul_tile", kSpvMatmulTile, sizeof(kSpvMatmulTile), 6, nullptr, kSpvMatmulTilePreserve, sizeof(kSpvMatmulTilePreserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_Q4, "matmul_row_q4", kSpvMatmulRowQ4, sizeof(kSpvMatmulRowQ4), 12, kMatmulRowCounts, kSpvMatmulRowQ4Preserve, sizeof(kSpvMatmulRowQ4Preserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_Q5, "matmul_row_q5", kSpvMatmulRowQ5, sizeof(kSpvMatmulRowQ5), 12, kMatmulRowCounts, kSpvMatmulRowQ5Preserve, sizeof(kSpvMatmulRowQ5Preserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_Q50, "matmul_row_q50", kSpvMatmulRowQ50, sizeof(kSpvMatmulRowQ50), 12, kMatmulRowCounts, kSpvMatmulRowQ50Preserve, sizeof(kSpvMatmulRowQ50Preserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_K4, "matmul_row_k4", kSpvMatmulRowK4, sizeof(kSpvMatmulRowK4), 12, kMatmulRowCounts) \
    X(K_MATMUL_ROW_K5, "matmul_row_k5", kSpvMatmulRowK5, sizeof(kSpvMatmulRowK5), 12, kMatmulRowCounts, kSpvMatmulRowK5Preserve, sizeof(kSpvMatmulRowK5Preserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_K, "matmul_row_k", kSpvMatmulRowK, sizeof(kSpvMatmulRowK), 12, kMatmulRowCounts, kSpvMatmulRowKPreserve, sizeof(kSpvMatmulRowKPreserve), 0, kPreserve32) \
    X(K_NORM_ROPE_KV, "norm_rope_kv", kSpvNormRopeKv, sizeof(kSpvNormRopeKv), 11, nullptr) \
    X(K_ATTENTION_TILE, "attention_tile", kSpvAttentionTile, sizeof(kSpvAttentionTile), 6, nullptr) \
    X(K_KV_WRITE_K16, "kv_write_k16", kSpvKvWriteK16, sizeof(kSpvKvWriteK16), 5, nullptr) \
    X(K_KV_WRITE_V16, "kv_write_v16", kSpvKvWriteV16, sizeof(kSpvKvWriteV16), 5, nullptr) \
    X(K_KV_WRITE_KV16, "kv_write_kv16", kSpvKvWriteKV16, sizeof(kSpvKvWriteKV16), 5, nullptr) \
    X(K_ATTENTION_K16, "attention_k16", kSpvAttentionK16, sizeof(kSpvAttentionK16), 7, nullptr) \
    X(K_ATTENTION_V16, "attention_v16", kSpvAttentionV16, sizeof(kSpvAttentionV16), 7, nullptr) \
    X(K_ATTENTION_KV16, "attention_kv16", kSpvAttentionKV16, sizeof(kSpvAttentionKV16), 7, nullptr) \
    X(K_ATTENTION_TILE_K16, "attention_tile_k16", kSpvAttentionTileK16, sizeof(kSpvAttentionTileK16), 6, nullptr) \
    X(K_ATTENTION_TILE_V16, "attention_tile_v16", kSpvAttentionTileV16, sizeof(kSpvAttentionTileV16), 6, nullptr) \
    X(K_ATTENTION_TILE_KV16, "attention_tile_kv16", kSpvAttentionTileKV16, sizeof(kSpvAttentionTileKV16), 6, nullptr) \
    X(K_NORM_ROPE_KV_K16, "norm_rope_kv_k16", kSpvNormRopeKvK16, sizeof(kSpvNormRopeKvK16), 11, nullptr) \
    X(K_NORM_ROPE_KV_V16, "norm_rope_kv_v16", kSpvNormRopeKvV16, sizeof(kSpvNormRopeKvV16), 11, nullptr) \
    X(K_NORM_ROPE_KV_KV16, "norm_rope_kv_kv16", kSpvNormRopeKvKV16, sizeof(kSpvNormRopeKvKV16), 11, nullptr) \
    X(K_QUANTIZE_X, "quantize_x", kSpvQuantizeX, sizeof(kSpvQuantizeX), 2, nullptr) \
    X(K_MATMUL_ROW_Q8W, "matmul_row_q8w", kSpvMatmulRowQ8W, sizeof(kSpvMatmulRowQ8W), 12, kMatmulRowCounts, kSpvMatmulRowQ8wPreserve, sizeof(kSpvMatmulRowQ8wPreserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_TALL, "matmul_tile_tall", kSpvMatmulTile, sizeof(kSpvMatmulTile), 6, nullptr, kSpvMatmulTilePreserve, sizeof(kSpvMatmulTilePreserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_Q4_DOT, "matmul_row_q4_dot", kSpvMatmulRowQ4Dot, sizeof(kSpvMatmulRowQ4Dot), 12, kMatmulRowCounts, kSpvMatmulRowQ4DotPreserve, sizeof(kSpvMatmulRowQ4DotPreserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_Q5_DOT, "matmul_row_q5_dot", kSpvMatmulRowQ5Dot, sizeof(kSpvMatmulRowQ5Dot), 12, kMatmulRowCounts, kSpvMatmulRowQ5DotPreserve, sizeof(kSpvMatmulRowQ5DotPreserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_Q50_DOT, "matmul_row_q50_dot", kSpvMatmulRowQ50Dot, sizeof(kSpvMatmulRowQ50Dot), 12, kMatmulRowCounts, kSpvMatmulRowQ50DotPreserve, sizeof(kSpvMatmulRowQ50DotPreserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_K4_DOT, "matmul_row_k4_dot", kSpvMatmulRowK4Dot, sizeof(kSpvMatmulRowK4Dot), 12, kMatmulRowCounts, kSpvMatmulRowK4DotPreserve, sizeof(kSpvMatmulRowK4DotPreserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_K5_DOT, "matmul_row_k5_dot", kSpvMatmulRowK5Dot, sizeof(kSpvMatmulRowK5Dot), 12, kMatmulRowCounts, kSpvMatmulRowK5DotPreserve, sizeof(kSpvMatmulRowK5DotPreserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_K_DOT, "matmul_row_k_dot", kSpvMatmulRowKDot, sizeof(kSpvMatmulRowKDot), 12, kMatmulRowCounts, kSpvMatmulRowKDotPreserve, sizeof(kSpvMatmulRowKDotPreserve), 0, kPreserve32) \
    X(K_QUANTIZE_XW, "quantize_xw", kSpvQuantizeXW, sizeof(kSpvQuantizeXW), 2, nullptr) \
    X(K_MATMUL_TILE_Q, "matmul_tile_q", kSpvMatmulTileQ, sizeof(kSpvMatmulTileQ), 5, kMatmulTileQCounts, kSpvMatmulTileQPreserve, sizeof(kSpvMatmulTileQPreserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_Q_TALL, "matmul_tile_q_tall", kSpvMatmulTileQ, sizeof(kSpvMatmulTileQ), 5, kMatmulTileQCounts, kSpvMatmulTileQPreserve, sizeof(kSpvMatmulTileQPreserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_Q5, "matmul_tile_q5", kSpvMatmulTileQ5, sizeof(kSpvMatmulTileQ5), 5, kMatmulTileQCounts, kSpvMatmulTileQ5Preserve, sizeof(kSpvMatmulTileQ5Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_Q50, "matmul_tile_q50", kSpvMatmulTileQ50, sizeof(kSpvMatmulTileQ50), 5, kMatmulTileQCounts, kSpvMatmulTileQ50Preserve, sizeof(kSpvMatmulTileQ50Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_Q5_TALL, "matmul_tile_q5_tall", kSpvMatmulTileQ5, sizeof(kSpvMatmulTileQ5), 5, kMatmulTileQCounts, kSpvMatmulTileQ5Preserve, sizeof(kSpvMatmulTileQ5Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_Q50_TALL, "matmul_tile_q50_tall", kSpvMatmulTileQ50, sizeof(kSpvMatmulTileQ50), 5, kMatmulTileQCounts, kSpvMatmulTileQ50Preserve, sizeof(kSpvMatmulTileQ50Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_Q5I8, "matmul_tile_q5i8", kSpvMatmulTileQ5i8, sizeof(kSpvMatmulTileQ5i8), 5, kMatmulTileQCounts, kSpvMatmulTileQ5i8Preserve, sizeof(kSpvMatmulTileQ5i8Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_Q50I8, "matmul_tile_q50i8", kSpvMatmulTileQ50i8, sizeof(kSpvMatmulTileQ50i8), 5, kMatmulTileQCounts, kSpvMatmulTileQ50i8Preserve, sizeof(kSpvMatmulTileQ50i8Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_Q5I8_TALL, "matmul_tile_q5i8_tall", kSpvMatmulTileQ5i8, sizeof(kSpvMatmulTileQ5i8), 5, kMatmulTileQCounts, kSpvMatmulTileQ5i8Preserve, sizeof(kSpvMatmulTileQ5i8Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_Q50I8_TALL, "matmul_tile_q50i8_tall", kSpvMatmulTileQ50i8, sizeof(kSpvMatmulTileQ50i8), 5, kMatmulTileQCounts, kSpvMatmulTileQ50i8Preserve, sizeof(kSpvMatmulTileQ50i8Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_Q2, "matmul_tile_q2", kSpvMatmulTileQ2, sizeof(kSpvMatmulTileQ2), 5, kMatmulTileQCounts, kSpvMatmulTileQ2Preserve, sizeof(kSpvMatmulTileQ2Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_Q3, "matmul_tile_q3", kSpvMatmulTileQ3, sizeof(kSpvMatmulTileQ3), 5, kMatmulTileQCounts, kSpvMatmulTileQ3Preserve, sizeof(kSpvMatmulTileQ3Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_Q6, "matmul_tile_q6", kSpvMatmulTileQ6, sizeof(kSpvMatmulTileQ6), 5, kMatmulTileQCounts, kSpvMatmulTileQ6Preserve, sizeof(kSpvMatmulTileQ6Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_Q8, "matmul_tile_q8", kSpvMatmulTileQ8, sizeof(kSpvMatmulTileQ8), 5, kMatmulTileQCounts) \
    X(K_MATMUL_TILE_Q8_TALL, "matmul_tile_q8_tall", kSpvMatmulTileQ8, sizeof(kSpvMatmulTileQ8), 5, kMatmulTileQCounts) \
    X(K_MATMUL_TILE_QI8, "matmul_tile_qi8", kSpvMatmulTileQi8, sizeof(kSpvMatmulTileQi8), 5, kMatmulTileQCounts, kSpvMatmulTileQi8Preserve, sizeof(kSpvMatmulTileQi8Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_QI8_TALL, "matmul_tile_qi8_tall", kSpvMatmulTileQi8, sizeof(kSpvMatmulTileQi8), 5, kMatmulTileQCounts, kSpvMatmulTileQi8Preserve, sizeof(kSpvMatmulTileQi8Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_Q6I8, "matmul_tile_q6i8", kSpvMatmulTileQ6i8, sizeof(kSpvMatmulTileQ6i8), 5, kMatmulTileQCounts, kSpvMatmulTileQ6i8Preserve, sizeof(kSpvMatmulTileQ6i8Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_Q8I8, "matmul_tile_q8i8", kSpvMatmulTileQ8i8, sizeof(kSpvMatmulTileQ8i8), 5, kMatmulTileQCounts) \
    X(K_MATMUL_TILE_Q8I8_TALL, "matmul_tile_q8i8_tall", kSpvMatmulTileQ8i8, sizeof(kSpvMatmulTileQ8i8), 5, kMatmulTileQCounts) \
    X(K_MATMUL_REDUCE, "matmul_reduce", kSpvMatmulReduce, sizeof(kSpvMatmulReduce), 2, kMatmulReduceCounts, kSpvMatmulReducePreserve, sizeof(kSpvMatmulReducePreserve), 0, kPreserve32) \
    X(K_MATMUL_VEC_Q8, "matmul_vec_q8", kSpvMatmulVecQ8, sizeof(kSpvMatmulVecQ8), 12, kMatmulRowCounts, kSpvMatmulVecQ8Preserve, sizeof(kSpvMatmulVecQ8Preserve), 0, kPreserve32) \
    X(K_MATMUL_VEC_Q8_I8, "matmul_vec_q8_i8", kSpvMatmulVecQ8I8, sizeof(kSpvMatmulVecQ8I8), 12, kMatmulRowCounts, kSpvMatmulVecQ8I8Preserve, sizeof(kSpvMatmulVecQ8I8Preserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_Q4_I8, "matmul_row_q4_i8", kSpvMatmulRowQ4I8, sizeof(kSpvMatmulRowQ4I8), 12, kMatmulRowCounts, kSpvMatmulRowQ4I8Preserve, sizeof(kSpvMatmulRowQ4I8Preserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_Q5_I8, "matmul_row_q5_i8", kSpvMatmulRowQ5I8, sizeof(kSpvMatmulRowQ5I8), 12, kMatmulRowCounts, kSpvMatmulRowQ5I8Preserve, sizeof(kSpvMatmulRowQ5I8Preserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_Q50_I8, "matmul_row_q50_i8", kSpvMatmulRowQ50I8, sizeof(kSpvMatmulRowQ50I8), 12, kMatmulRowCounts, kSpvMatmulRowQ50I8Preserve, sizeof(kSpvMatmulRowQ50I8Preserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_K4_I8, "matmul_row_k4_i8", kSpvMatmulRowK4I8, sizeof(kSpvMatmulRowK4I8), 12, kMatmulRowCounts, kSpvMatmulRowK4I8Preserve, sizeof(kSpvMatmulRowK4I8Preserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_K5_I8, "matmul_row_k5_i8", kSpvMatmulRowK5I8, sizeof(kSpvMatmulRowK5I8), 12, kMatmulRowCounts, kSpvMatmulRowK5I8Preserve, sizeof(kSpvMatmulRowK5I8Preserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_K_I8, "matmul_row_k_i8", kSpvMatmulRowKI8, sizeof(kSpvMatmulRowKI8), 12, kMatmulRowCounts, kSpvMatmulRowKI8Preserve, sizeof(kSpvMatmulRowKI8Preserve), 0, kPreserve32) \
    X(K_MOE_ROUTE_S1, "moe_route_s1", kSpvMoeRouteS1, sizeof(kSpvMoeRouteS1), 3, nullptr) \
    X(K_MOE_ROUTE_S2, "moe_route_s2", kSpvMoeRouteS2, sizeof(kSpvMoeRouteS2), 3, nullptr) \
    X(K_MOE_ROUTE_S4, "moe_route_s4", kSpvMoeRouteS4, sizeof(kSpvMoeRouteS4), 3, nullptr) \
    X(K_MOE_ROUTE_S8, "moe_route_s8", kSpvMoeRouteS8, sizeof(kSpvMoeRouteS8), 3, nullptr) \
    X(K_MOE_ROUTE, "moe_route", kSpvMoeRoute, sizeof(kSpvMoeRoute), 3, nullptr) \
    X(K_MOE_COMBINE, "moe_combine", kSpvMoeCombine, sizeof(kSpvMoeCombine), 4, nullptr, kSpvMoeCombinePreserve, sizeof(kSpvMoeCombinePreserve), 0, kPreserve32) \
    X(K_MOE_GROUP, "moe_group", kSpvMoeGroup, sizeof(kSpvMoeGroup), 2, nullptr) \
    X(K_MOE_STEP_ROUTE_S1, "moe_step_route_s1", kSpvMoeStepRouteS1, sizeof(kSpvMoeStepRouteS1), 16, kMoeStepRouteCounts) \
    X(K_MOE_STEP_ROUTE_S2, "moe_step_route_s2", kSpvMoeStepRouteS2, sizeof(kSpvMoeStepRouteS2), 16, kMoeStepRouteCounts) \
    X(K_MOE_STEP_ROUTE_S4, "moe_step_route_s4", kSpvMoeStepRouteS4, sizeof(kSpvMoeStepRouteS4), 16, kMoeStepRouteCounts) \
    X(K_MOE_STEP_ROUTE_S8, "moe_step_route_s8", kSpvMoeStepRouteS8, sizeof(kSpvMoeStepRouteS8), 16, kMoeStepRouteCounts) \
    X(K_MOE_STEP_ROUTE, "moe_step_route", kSpvMoeStepRoute, sizeof(kSpvMoeStepRoute), 16, kMoeStepRouteCounts) \
    X(K_MOE_STEP_UP_K4Q8, "moe_step_up_k4q8", kSpvMoeStepUpK4q8, sizeof(kSpvMoeStepUpK4q8), 12, kMoeStepUpCounts, kSpvMoeStepUpK4q8Preserve, sizeof(kSpvMoeStepUpK4q8Preserve), 0, kPreserve32) \
    X(K_MOE_STEP_UP_K4Q8_I8, "moe_step_up_k4q8_i8", kSpvMoeStepUpK4q8I8, sizeof(kSpvMoeStepUpK4q8I8), 12, kMoeStepUpCounts, kSpvMoeStepUpK4q8I8Preserve, sizeof(kSpvMoeStepUpK4q8I8Preserve), 0, kPreserve32) \
    X(K_MOE_STEP_UP_K4, "moe_step_up_k4", kSpvMoeStepUpK4, sizeof(kSpvMoeStepUpK4), 12, kMoeStepUpCounts, kSpvMoeStepUpK4Preserve, sizeof(kSpvMoeStepUpK4Preserve), 0, kPreserve32) \
    X(K_MOE_STEP_UP_K4_I8, "moe_step_up_k4_i8", kSpvMoeStepUpK4I8, sizeof(kSpvMoeStepUpK4I8), 12, kMoeStepUpCounts, kSpvMoeStepUpK4I8Preserve, sizeof(kSpvMoeStepUpK4I8Preserve), 0, kPreserve32) \
    X(K_MOE_STEP_DOWN_Q8, "moe_step_down_q8", kSpvMoeStepDownQ8, sizeof(kSpvMoeStepDownQ8), 14, kMoeStepDownCounts, kSpvMoeStepDownQ8Preserve, sizeof(kSpvMoeStepDownQ8Preserve), 0, kPreserve32) \
    X(K_MOE_STEP_DOWN_Q8_I8, "moe_step_down_q8_i8", kSpvMoeStepDownQ8I8, sizeof(kSpvMoeStepDownQ8I8), 14, kMoeStepDownCounts, kSpvMoeStepDownQ8I8Preserve, sizeof(kSpvMoeStepDownQ8I8Preserve), 0, kPreserve32) \
    X(K_MOE_STEP_DOWN_Q5Q8, "moe_step_down_q5q8", kSpvMoeStepDownQ5q8, sizeof(kSpvMoeStepDownQ5q8), 14, kMoeStepDownCounts, kSpvMoeStepDownQ5q8Preserve, sizeof(kSpvMoeStepDownQ5q8Preserve), 0, kPreserve32) \
    X(K_MOE_STEP_DOWN_Q5Q8_I8, "moe_step_down_q5q8_i8", kSpvMoeStepDownQ5q8I8, sizeof(kSpvMoeStepDownQ5q8I8), 14, kMoeStepDownCounts, kSpvMoeStepDownQ5q8I8Preserve, sizeof(kSpvMoeStepDownQ5q8I8Preserve), 0, kPreserve32) \
    X(K_MOE_STEP_DOWN_K, "moe_step_down_k", kSpvMoeStepDownK, sizeof(kSpvMoeStepDownK), 14, kMoeStepDownCounts, kSpvMoeStepDownKPreserve, sizeof(kSpvMoeStepDownKPreserve), 0, kPreserve32) \
    X(K_MOE_STEP_DOWN_K_I8, "moe_step_down_k_i8", kSpvMoeStepDownKI8, sizeof(kSpvMoeStepDownKI8), 14, kMoeStepDownCounts, kSpvMoeStepDownKI8Preserve, sizeof(kSpvMoeStepDownKI8Preserve), 0, kPreserve32) \
    X(K_MOE_STEP_ACT, "moe_step_act", kSpvMoeStepAct, sizeof(kSpvMoeStepAct), 5, nullptr) \
    X(K_ATTENTION_G4, "attention_g4", kSpvAttentionG4, sizeof(kSpvAttentionG4), 7, nullptr) \
    X(K_ATTENTION_K16_G4, "attention_k16_g4", kSpvAttentionK16G4, sizeof(kSpvAttentionK16G4), 7, nullptr) \
    X(K_ATTENTION_V16_G4, "attention_v16_g4", kSpvAttentionV16G4, sizeof(kSpvAttentionV16G4), 7, nullptr) \
    X(K_ATTENTION_KV16_G4, "attention_kv16_g4", kSpvAttentionKV16G4, sizeof(kSpvAttentionKV16G4), 7, nullptr) \
    X(K_ATTENTION_VEC, "attention_vec", kSpvAttentionVec, sizeof(kSpvAttentionVec), 7, nullptr) \
    X(K_ATTENTION_VEC_K16, "attention_vec_k16", kSpvAttentionVecK16, sizeof(kSpvAttentionVecK16), 7, nullptr) \
    X(K_ATTENTION_VEC_V16, "attention_vec_v16", kSpvAttentionVecV16, sizeof(kSpvAttentionVecV16), 7, nullptr) \
    X(K_ATTENTION_VEC_KV16, "attention_vec_kv16", kSpvAttentionVecKV16, sizeof(kSpvAttentionVecKV16), 7, nullptr) \
    X(K_ATTENTION_VEC_G4, "attention_vec_g4", kSpvAttentionVecG4, sizeof(kSpvAttentionVecG4), 7, nullptr) \
    X(K_ATTENTION_VEC_K16_G4, "attention_vec_k16_g4", kSpvAttentionVecK16G4, sizeof(kSpvAttentionVecK16G4), 7, nullptr) \
    X(K_ATTENTION_VEC_V16_G4, "attention_vec_v16_g4", kSpvAttentionVecV16G4, sizeof(kSpvAttentionVecV16G4), 7, nullptr) \
    X(K_ATTENTION_VEC_KV16_G4, "attention_vec_kv16_g4", kSpvAttentionVecKV16G4, sizeof(kSpvAttentionVecKV16G4), 7, nullptr) \
    X(K_MATMUL_ROW_F32, "matmul_row_f32", kSpvMatmulRow, sizeof(kSpvMatmulRow), 12, kMatmulRowCounts) \
    X(K_MATMUL_ROW_F32_ONLY, "matmul_row_f32_only", kSpvMatmulRowF32Only, sizeof(kSpvMatmulRowF32Only), 12, kMatmulRowCounts) \
    X(K_SIGMOID_MUL, "sigmoid_mul", kSpvSigmoidMul, sizeof(kSpvSigmoidMul), 4, nullptr) \
    X(K_GATED_RMS_NORM, "gated_rms_norm", kSpvGatedRmsNorm, sizeof(kSpvGatedRmsNorm), 5, nullptr) \
    X(K_CAUSAL_CONV_SILU, "causal_conv_silu", kSpvCausalConvSilu, sizeof(kSpvCausalConvSilu), 5, nullptr) \
    X(K_DELTA_RULE, "delta_rule", kSpvDeltaRule, sizeof(kSpvDeltaRule), 8, nullptr) \
    X(K_ATTENTION_TILE_D256, "attention_tile_d256", kSpvAttentionTileD256, sizeof(kSpvAttentionTileD256), 6, nullptr) \
    X(K_ATTENTION_TILE_D256_K16, "attention_tile_d256_k16", kSpvAttentionTileD256K16, sizeof(kSpvAttentionTileD256K16), 6, nullptr) \
    X(K_ATTENTION_TILE_D256_V16, "attention_tile_d256_v16", kSpvAttentionTileD256V16, sizeof(kSpvAttentionTileD256V16), 6, nullptr) \
    X(K_ATTENTION_TILE_D256_KV16, "attention_tile_d256_kv16", kSpvAttentionTileD256KV16, sizeof(kSpvAttentionTileD256KV16), 6, nullptr) \
    X(K_ATTENTION_VEC_D256, "attention_vec_d256", kSpvAttentionVecD256, sizeof(kSpvAttentionVecD256), 7, nullptr) \
    X(K_ATTENTION_VEC_D256_K16, "attention_vec_d256_k16", kSpvAttentionVecD256K16, sizeof(kSpvAttentionVecD256K16), 7, nullptr) \
    X(K_ATTENTION_VEC_D256_V16, "attention_vec_d256_v16", kSpvAttentionVecD256V16, sizeof(kSpvAttentionVecD256V16), 7, nullptr) \
    X(K_ATTENTION_VEC_D256_KV16, "attention_vec_d256_kv16", kSpvAttentionVecD256KV16, sizeof(kSpvAttentionVecD256KV16), 7, nullptr) \
    X(K_ATTENTION_VEC_D256_G4, "attention_vec_d256_g4", kSpvAttentionVecD256G4, sizeof(kSpvAttentionVecD256G4), 7, nullptr) \
    X(K_ATTENTION_VEC_D256_K16_G4, "attention_vec_d256_k16_g4", kSpvAttentionVecD256K16G4, sizeof(kSpvAttentionVecD256K16G4), 7, nullptr) \
    X(K_ATTENTION_VEC_D256_V16_G4, "attention_vec_d256_v16_g4", kSpvAttentionVecD256V16G4, sizeof(kSpvAttentionVecD256V16G4), 7, nullptr) \
    X(K_ATTENTION_VEC_D256_KV16_G4, "attention_vec_d256_kv16_g4", kSpvAttentionVecD256KV16G4, sizeof(kSpvAttentionVecD256KV16G4), 7, nullptr) \
    X(K_MATMUL_ROW_TQ1_0, "matmul_row_tq1_0", kSpvMatmulRowTq10, sizeof(kSpvMatmulRowTq10), 12, kMatmulRowCounts, kSpvMatmulRowTq10Preserve, sizeof(kSpvMatmulRowTq10Preserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_TQ2_0, "matmul_row_tq2_0", kSpvMatmulRowTq20, sizeof(kSpvMatmulRowTq20), 12, kMatmulRowCounts, kSpvMatmulRowTq20Preserve, sizeof(kSpvMatmulRowTq20Preserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_IQ1_S, "matmul_row_iq1_s", kSpvMatmulRowIq1S, sizeof(kSpvMatmulRowIq1S), 12, kMatmulRowCounts, kSpvMatmulRowIq1SPreserve, sizeof(kSpvMatmulRowIq1SPreserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_IQ1_M, "matmul_row_iq1_m", kSpvMatmulRowIq1M, sizeof(kSpvMatmulRowIq1M), 12, kMatmulRowCounts, kSpvMatmulRowIq1MPreserve, sizeof(kSpvMatmulRowIq1MPreserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_IQ2_S, "matmul_row_iq2_s", kSpvMatmulRowIq2S, sizeof(kSpvMatmulRowIq2S), 12, kMatmulRowCounts, kSpvMatmulRowIq2SPreserve, sizeof(kSpvMatmulRowIq2SPreserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_IQ3_XXS, "matmul_row_iq3_xxs", kSpvMatmulRowIq3Xxs, sizeof(kSpvMatmulRowIq3Xxs), 12, kMatmulRowCounts, kSpvMatmulRowIq3XxsPreserve, sizeof(kSpvMatmulRowIq3XxsPreserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_IQ2_XXS, "matmul_row_iq2_xxs", kSpvMatmulRowIq2Xxs, sizeof(kSpvMatmulRowIq2Xxs), 12, kMatmulRowCounts, kSpvMatmulRowIq2XxsPreserve, sizeof(kSpvMatmulRowIq2XxsPreserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_IQ2_XS, "matmul_row_iq2_xs", kSpvMatmulRowIq2Xs, sizeof(kSpvMatmulRowIq2Xs), 12, kMatmulRowCounts, kSpvMatmulRowIq2XsPreserve, sizeof(kSpvMatmulRowIq2XsPreserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_TQ1_0, "matmul_tile_tq1_0", kSpvMatmulTileTq10, sizeof(kSpvMatmulTileTq10), 6, nullptr, kSpvMatmulTileTq10Preserve, sizeof(kSpvMatmulTileTq10Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_TQ2_0, "matmul_tile_tq2_0", kSpvMatmulTileTq20, sizeof(kSpvMatmulTileTq20), 6, nullptr, kSpvMatmulTileTq20Preserve, sizeof(kSpvMatmulTileTq20Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ2_S, "matmul_tile_iq2_s", kSpvMatmulTileIq2S, sizeof(kSpvMatmulTileIq2S), 6, nullptr, kSpvMatmulTileIq2SPreserve, sizeof(kSpvMatmulTileIq2SPreserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ3_XXS, "matmul_tile_iq3_xxs", kSpvMatmulTileIq3Xxs, sizeof(kSpvMatmulTileIq3Xxs), 6, nullptr, kSpvMatmulTileIq3XxsPreserve, sizeof(kSpvMatmulTileIq3XxsPreserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ2_XXS, "matmul_tile_iq2_xxs", kSpvMatmulTileIq2Xxs, sizeof(kSpvMatmulTileIq2Xxs), 6, nullptr, kSpvMatmulTileIq2XxsPreserve, sizeof(kSpvMatmulTileIq2XxsPreserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ2_XS, "matmul_tile_iq2_xs", kSpvMatmulTileIq2Xs, sizeof(kSpvMatmulTileIq2Xs), 6, nullptr, kSpvMatmulTileIq2XsPreserve, sizeof(kSpvMatmulTileIq2XsPreserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_TQ1_0_TALL, "matmul_tile_tq1_0_tall", kSpvMatmulTileTq10, sizeof(kSpvMatmulTileTq10), 6, nullptr, kSpvMatmulTileTq10Preserve, sizeof(kSpvMatmulTileTq10Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_TQ2_0_TALL, "matmul_tile_tq2_0_tall", kSpvMatmulTileTq20, sizeof(kSpvMatmulTileTq20), 6, nullptr, kSpvMatmulTileTq20Preserve, sizeof(kSpvMatmulTileTq20Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ2_S_TALL, "matmul_tile_iq2_s_tall", kSpvMatmulTileIq2S, sizeof(kSpvMatmulTileIq2S), 6, nullptr, kSpvMatmulTileIq2SPreserve, sizeof(kSpvMatmulTileIq2SPreserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ3_XXS_TALL, "matmul_tile_iq3_xxs_tall", kSpvMatmulTileIq3Xxs, sizeof(kSpvMatmulTileIq3Xxs), 6, nullptr, kSpvMatmulTileIq3XxsPreserve, sizeof(kSpvMatmulTileIq3XxsPreserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ2_XXS_TALL, "matmul_tile_iq2_xxs_tall", kSpvMatmulTileIq2Xxs, sizeof(kSpvMatmulTileIq2Xxs), 6, nullptr, kSpvMatmulTileIq2XxsPreserve, sizeof(kSpvMatmulTileIq2XxsPreserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ2_XS_TALL, "matmul_tile_iq2_xs_tall", kSpvMatmulTileIq2Xs, sizeof(kSpvMatmulTileIq2Xs), 6, nullptr, kSpvMatmulTileIq2XsPreserve, sizeof(kSpvMatmulTileIq2XsPreserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_TQ1_0_BF16, "matmul_tile_tq1_0_bf16", kSpvMatmulTileTq10Bf16, sizeof(kSpvMatmulTileTq10Bf16), 6, nullptr, kSpvMatmulTileTq10Bf16Preserve, sizeof(kSpvMatmulTileTq10Bf16Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_TQ2_0_BF16, "matmul_tile_tq2_0_bf16", kSpvMatmulTileTq20Bf16, sizeof(kSpvMatmulTileTq20Bf16), 6, nullptr, kSpvMatmulTileTq20Bf16Preserve, sizeof(kSpvMatmulTileTq20Bf16Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ2_S_BF16, "matmul_tile_iq2_s_bf16", kSpvMatmulTileIq2SBf16, sizeof(kSpvMatmulTileIq2SBf16), 6, nullptr, kSpvMatmulTileIq2SBf16Preserve, sizeof(kSpvMatmulTileIq2SBf16Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ3_XXS_BF16, "matmul_tile_iq3_xxs_bf16", kSpvMatmulTileIq3XxsBf16, sizeof(kSpvMatmulTileIq3XxsBf16), 6, nullptr, kSpvMatmulTileIq3XxsBf16Preserve, sizeof(kSpvMatmulTileIq3XxsBf16Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ2_XXS_BF16, "matmul_tile_iq2_xxs_bf16", kSpvMatmulTileIq2XxsBf16, sizeof(kSpvMatmulTileIq2XxsBf16), 6, nullptr, kSpvMatmulTileIq2XxsBf16Preserve, sizeof(kSpvMatmulTileIq2XxsBf16Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ2_XS_BF16, "matmul_tile_iq2_xs_bf16", kSpvMatmulTileIq2XsBf16, sizeof(kSpvMatmulTileIq2XsBf16), 6, nullptr, kSpvMatmulTileIq2XsBf16Preserve, sizeof(kSpvMatmulTileIq2XsBf16Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_TQ1_0_BF16_TALL, "matmul_tile_tq1_0_bf16_tall", kSpvMatmulTileTq10Bf16, sizeof(kSpvMatmulTileTq10Bf16), 6, nullptr, kSpvMatmulTileTq10Bf16Preserve, sizeof(kSpvMatmulTileTq10Bf16Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_TQ2_0_BF16_TALL, "matmul_tile_tq2_0_bf16_tall", kSpvMatmulTileTq20Bf16, sizeof(kSpvMatmulTileTq20Bf16), 6, nullptr, kSpvMatmulTileTq20Bf16Preserve, sizeof(kSpvMatmulTileTq20Bf16Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ1_S, "matmul_tile_iq1_s", kSpvMatmulTileIq1S, sizeof(kSpvMatmulTileIq1S), 6, nullptr, kSpvMatmulTileIq1SPreserve, sizeof(kSpvMatmulTileIq1SPreserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ1_M, "matmul_tile_iq1_m", kSpvMatmulTileIq1M, sizeof(kSpvMatmulTileIq1M), 6, nullptr, kSpvMatmulTileIq1MPreserve, sizeof(kSpvMatmulTileIq1MPreserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ1_S_TALL, "matmul_tile_iq1_s_tall", kSpvMatmulTileIq1S, sizeof(kSpvMatmulTileIq1S), 6, nullptr, kSpvMatmulTileIq1SPreserve, sizeof(kSpvMatmulTileIq1SPreserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ1_M_TALL, "matmul_tile_iq1_m_tall", kSpvMatmulTileIq1M, sizeof(kSpvMatmulTileIq1M), 6, nullptr, kSpvMatmulTileIq1MPreserve, sizeof(kSpvMatmulTileIq1MPreserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ1_S_BF16, "matmul_tile_iq1_s_bf16", kSpvMatmulTileIq1SBf16, sizeof(kSpvMatmulTileIq1SBf16), 6, nullptr, kSpvMatmulTileIq1SBf16Preserve, sizeof(kSpvMatmulTileIq1SBf16Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ1_M_BF16, "matmul_tile_iq1_m_bf16", kSpvMatmulTileIq1MBf16, sizeof(kSpvMatmulTileIq1MBf16), 6, nullptr, kSpvMatmulTileIq1MBf16Preserve, sizeof(kSpvMatmulTileIq1MBf16Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ1_S_BF16_TALL, "matmul_tile_iq1_s_bf16_tall", kSpvMatmulTileIq1SBf16, sizeof(kSpvMatmulTileIq1SBf16), 6, nullptr, kSpvMatmulTileIq1SBf16Preserve, sizeof(kSpvMatmulTileIq1SBf16Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ1_M_BF16_TALL, "matmul_tile_iq1_m_bf16_tall", kSpvMatmulTileIq1MBf16, sizeof(kSpvMatmulTileIq1MBf16), 6, nullptr, kSpvMatmulTileIq1MBf16Preserve, sizeof(kSpvMatmulTileIq1MBf16Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ2_S_BF16_TALL, "matmul_tile_iq2_s_bf16_tall", kSpvMatmulTileIq2SBf16, sizeof(kSpvMatmulTileIq2SBf16), 6, nullptr, kSpvMatmulTileIq2SBf16Preserve, sizeof(kSpvMatmulTileIq2SBf16Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ3_XXS_BF16_TALL, "matmul_tile_iq3_xxs_bf16_tall", kSpvMatmulTileIq3XxsBf16, sizeof(kSpvMatmulTileIq3XxsBf16), 6, nullptr, kSpvMatmulTileIq3XxsBf16Preserve, sizeof(kSpvMatmulTileIq3XxsBf16Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ2_XXS_BF16_TALL, "matmul_tile_iq2_xxs_bf16_tall", kSpvMatmulTileIq2XxsBf16, sizeof(kSpvMatmulTileIq2XxsBf16), 6, nullptr, kSpvMatmulTileIq2XxsBf16Preserve, sizeof(kSpvMatmulTileIq2XxsBf16Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ2_XS_BF16_TALL, "matmul_tile_iq2_xs_bf16_tall", kSpvMatmulTileIq2XsBf16, sizeof(kSpvMatmulTileIq2XsBf16), 6, nullptr, kSpvMatmulTileIq2XsBf16Preserve, sizeof(kSpvMatmulTileIq2XsBf16Preserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_IQ3_S, "matmul_row_iq3_s", kSpvMatmulRowIq3S, sizeof(kSpvMatmulRowIq3S), 12, kMatmulRowCounts, kSpvMatmulRowIq3SPreserve, sizeof(kSpvMatmulRowIq3SPreserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ3_S, "matmul_tile_iq3_s", kSpvMatmulTileIq3S, sizeof(kSpvMatmulTileIq3S), 6, nullptr, kSpvMatmulTileIq3SPreserve, sizeof(kSpvMatmulTileIq3SPreserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ3_S_TALL, "matmul_tile_iq3_s_tall", kSpvMatmulTileIq3S, sizeof(kSpvMatmulTileIq3S), 6, nullptr, kSpvMatmulTileIq3SPreserve, sizeof(kSpvMatmulTileIq3SPreserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ3_S_BF16, "matmul_tile_iq3_s_bf16", kSpvMatmulTileIq3SBf16, sizeof(kSpvMatmulTileIq3SBf16), 6, nullptr, kSpvMatmulTileIq3SBf16Preserve, sizeof(kSpvMatmulTileIq3SBf16Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_IQ3_S_BF16_TALL, "matmul_tile_iq3_s_bf16_tall", kSpvMatmulTileIq3SBf16, sizeof(kSpvMatmulTileIq3SBf16), 6, nullptr, kSpvMatmulTileIq3SBf16Preserve, sizeof(kSpvMatmulTileIq3SBf16Preserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_FLOAT_X, "matmul_row_float_x", kSpvMatmulRowFloatX, sizeof(kSpvMatmulRowFloatX), 12, kMatmulRowCounts, kSpvMatmulRowFloatXPreserve, sizeof(kSpvMatmulRowFloatXPreserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_FLOAT_X_Q2, "matmul_row_float_x_q2", kSpvMatmulRowFloatXQ2, sizeof(kSpvMatmulRowFloatXQ2), 12, kMatmulRowCounts, kSpvMatmulRowFloatXQ2Preserve, sizeof(kSpvMatmulRowFloatXQ2Preserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_FLOAT_X_Q3, "matmul_row_float_x_q3", kSpvMatmulRowFloatXQ3, sizeof(kSpvMatmulRowFloatXQ3), 12, kMatmulRowCounts, kSpvMatmulRowFloatXQ3Preserve, sizeof(kSpvMatmulRowFloatXQ3Preserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_Q2_DOT, "matmul_row_q2_dot", kSpvMatmulRowQ2Dot, sizeof(kSpvMatmulRowQ2Dot), 12, kMatmulRowCounts, kSpvMatmulRowQ2DotPreserve, sizeof(kSpvMatmulRowQ2DotPreserve), 0, kPreserve32) \
    X(K_MATMUL_ROW_Q3_DOT, "matmul_row_q3_dot", kSpvMatmulRowQ3Dot, sizeof(kSpvMatmulRowQ3Dot), 12, kMatmulRowCounts, kSpvMatmulRowQ3DotPreserve, sizeof(kSpvMatmulRowQ3DotPreserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_BF16, "matmul_tile_bf16", kSpvMatmulTileBf16, sizeof(kSpvMatmulTileBf16), 6, nullptr, kSpvMatmulTileBf16Preserve, sizeof(kSpvMatmulTileBf16Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_BF16_TALL, "matmul_tile_bf16_tall", kSpvMatmulTileBf16, sizeof(kSpvMatmulTileBf16), 6, nullptr, kSpvMatmulTileBf16Preserve, sizeof(kSpvMatmulTileBf16Preserve), 0, kPreserve32) \
    X(K_EMBED_MXFP4, "embed_mxfp4", kSpvEmbedMxfp4, sizeof(kSpvEmbedMxfp4), 4, nullptr) \
    X(K_MATMUL_TILE_MXFP4, "matmul_tile_mxfp4", kSpvMatmulTileMxfp4, sizeof(kSpvMatmulTileMxfp4), 6, nullptr, nullptr, 0, kPreserve32 | kSpecial64, 0) \
    X(K_MATMUL_TILE_MXFP4_TALL, "matmul_tile_mxfp4_tall", kSpvMatmulTileMxfp4, sizeof(kSpvMatmulTileMxfp4), 6, nullptr, nullptr, 0, kPreserve32 | kSpecial64, 0) \
    X(K_MATMUL_TILE_MXFP4_BF16, "matmul_tile_mxfp4_bf16", kSpvMatmulTileMxfp4Bf16, sizeof(kSpvMatmulTileMxfp4Bf16), 6, nullptr, nullptr, 0, kPreserve32 | kSpecial64, 0) \
    X(K_MATMUL_TILE_MXFP4_BF16_TALL, "matmul_tile_mxfp4_bf16_tall", kSpvMatmulTileMxfp4Bf16, sizeof(kSpvMatmulTileMxfp4Bf16), 6, nullptr, nullptr, 0, kPreserve32 | kSpecial64, 0) \
    X(K_MATMUL_REDUCE_MXFP4, "matmul_reduce_mxfp4", kSpvMatmulReduceMxfp4, sizeof(kSpvMatmulReduceMxfp4), 2, kMatmulReduceCounts, nullptr, 0, kPreserve32 | kSpecial64, 0) \
    X(K_COPY_MXFP4, "copy_mxfp4", kSpvCopyMxfp4, sizeof(kSpvCopyMxfp4), 2, nullptr) \
    X(K_MATMUL_ROW_MXFP4, "matmul_row_mxfp4", kSpvMatmulRowMxfp4, sizeof(kSpvMatmulRowMxfp4), 12, kMatmulRowCounts, nullptr, 0, kPreserve32 | kSpecial64, 0) \
    X(K_MATMUL_ROW_MXFP4_DOT, "matmul_row_mxfp4_dot", kSpvMatmulRowMxfp4Dot, sizeof(kSpvMatmulRowMxfp4Dot), 12, kMatmulRowCounts, nullptr, 0, kPreserve32 | kSpecial64, 0) \
    X(K_MATMUL_ROW_MXFP4_FLOAT_X, "matmul_row_mxfp4_float_x", kSpvMatmulRowMxfp4FloatX, sizeof(kSpvMatmulRowMxfp4FloatX), 12, kMatmulRowCounts, nullptr, 0, kPreserve32 | kSpecial64, 0) \
    X(K_MATMUL_TILE_Q8MX, "matmul_tile_q8mx", kSpvMatmulTileQ8mx, sizeof(kSpvMatmulTileQ8mx), 6, kMatmulTileQ8mxCounts, nullptr, 0, kPreserve32 | kSpecial64, 0) \
    X(K_MATMUL_TILE_Q8MX_TALL, "matmul_tile_q8mx_tall", kSpvMatmulTileQ8mx, sizeof(kSpvMatmulTileQ8mx), 6, kMatmulTileQ8mxCounts, nullptr, 0, kPreserve32 | kSpecial64, 0) \
    X(K_ARGMAX_ROWS, "argmax_rows", kSpvArgmaxRows, sizeof(kSpvArgmaxRows), 4, nullptr) \
    X(K_EMBED_IDS, "embed_ids", kSpvEmbedIds, sizeof(kSpvEmbedIds), 4, nullptr) \
    X(K_DELTA_RULE_SHORT, "delta_rule_short", kSpvDeltaRuleShort, sizeof(kSpvDeltaRuleShort), 8, nullptr) \
    X(K_EMBED_IDS_MXFP4, "embed_ids_mxfp4", kSpvEmbedIdsMxfp4, sizeof(kSpvEmbedIdsMxfp4), 4, nullptr) \
    X(K_GROUP_RMS_NORM, "group_rms_norm", kSpvGroupRmsNorm, sizeof(kSpvGroupRmsNorm), 4, nullptr) \
    X(K_SILU_SCALED, "silu_scaled", kSpvSiluScaled, sizeof(kSpvSiluScaled), 3, nullptr, kSpvSiluScaledPreserve, sizeof(kSpvSiluScaledPreserve), 0, kPreserve32) \
    X(K_MIX_STREAMS, "mix_streams", kSpvMixStreams, sizeof(kSpvMixStreams), 4, nullptr, kSpvMixStreamsPreserve, sizeof(kSpvMixStreamsPreserve), 0, kPreserve32) \
    X(K_INJECT_STREAMS, "inject_streams", kSpvInjectStreams, sizeof(kSpvInjectStreams), 3, nullptr, kSpvInjectStreamsPreserve, sizeof(kSpvInjectStreamsPreserve), 0, kPreserve32) \
    X(K_SIGNED_SQRT_GATE, "signed_sqrt_gate", kSpvSignedSqrtGate, sizeof(kSpvSignedSqrtGate), 5, nullptr) \
    X(K_DILATED_CONV_SILU_ADD, "dilated_conv_silu_add", kSpvDilatedConvSiluAdd, sizeof(kSpvDilatedConvSiluAdd), 6, nullptr) \
    X(K_EMBED_WF16, "embed_wf16", kSpvEmbedWf16, sizeof(kSpvEmbedWf16), 4, nullptr) \
    X(K_MATMUL_ROW_WF16, "matmul_row_wf16", kSpvMatmulRowWf16, sizeof(kSpvMatmulRowWf16), 12, kMatmulRowCounts) \
    X(K_MATMUL_TILE_WF16, "matmul_tile_wf16", kSpvMatmulTileWf16, sizeof(kSpvMatmulTileWf16), 6, nullptr, kSpvMatmulTileWf16Preserve, sizeof(kSpvMatmulTileWf16Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_WF16_TALL, "matmul_tile_wf16_tall", kSpvMatmulTileWf16, sizeof(kSpvMatmulTileWf16), 6, nullptr, kSpvMatmulTileWf16Preserve, sizeof(kSpvMatmulTileWf16Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_WF16_BF16, "matmul_tile_wf16_bf16", kSpvMatmulTileWf16Bf16, sizeof(kSpvMatmulTileWf16Bf16), 6, nullptr, kSpvMatmulTileWf16Bf16Preserve, sizeof(kSpvMatmulTileWf16Bf16Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_WF16_BF16_TALL, "matmul_tile_wf16_bf16_tall", kSpvMatmulTileWf16Bf16, sizeof(kSpvMatmulTileWf16Bf16), 6, nullptr, kSpvMatmulTileWf16Bf16Preserve, sizeof(kSpvMatmulTileWf16Bf16Preserve), 0, kPreserve32) \
    X(K_EMBED_WBF16, "embed_wbf16", kSpvEmbedWbf16, sizeof(kSpvEmbedWbf16), 4, nullptr) \
    X(K_MATMUL_ROW_WBF16, "matmul_row_wbf16", kSpvMatmulRowWbf16, sizeof(kSpvMatmulRowWbf16), 12, kMatmulRowCounts) \
    X(K_MATMUL_TILE_WBF16, "matmul_tile_wbf16", kSpvMatmulTileWbf16, sizeof(kSpvMatmulTileWbf16), 6, nullptr, kSpvMatmulTileWbf16Preserve, sizeof(kSpvMatmulTileWbf16Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_WBF16_TALL, "matmul_tile_wbf16_tall", kSpvMatmulTileWbf16, sizeof(kSpvMatmulTileWbf16), 6, nullptr, kSpvMatmulTileWbf16Preserve, sizeof(kSpvMatmulTileWbf16Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_WBF16_BF16, "matmul_tile_wbf16_bf16", kSpvMatmulTileWbf16Bf16, sizeof(kSpvMatmulTileWbf16Bf16), 6, nullptr, kSpvMatmulTileWbf16Bf16Preserve, sizeof(kSpvMatmulTileWbf16Bf16Preserve), 0, kPreserve32) \
    X(K_MATMUL_TILE_WBF16_BF16_TALL, "matmul_tile_wbf16_bf16_tall", kSpvMatmulTileWbf16Bf16, sizeof(kSpvMatmulTileWbf16Bf16), 6, nullptr, kSpvMatmulTileWbf16Bf16Preserve, sizeof(kSpvMatmulTileWbf16Bf16Preserve), 0, kPreserve32) \
    X(K_EMBED_IDS_WF16, "embed_ids_wf16", kSpvEmbedIdsWf16, sizeof(kSpvEmbedIdsWf16), 4, nullptr) \
    X(K_EMBED_IDS_WBF16, "embed_ids_wbf16", kSpvEmbedIdsWbf16, sizeof(kSpvEmbedIdsWbf16), 4, nullptr) \
    X(K_MATMUL_TILE_TQ1_0_DOT, "matmul_tile_tq1_0_dot", kSpvMatmulTileTq10Dot, sizeof(kSpvMatmulTileTq10Dot), 5, kMatmulTileQCounts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_TILE_TQ2_0_DOT, "matmul_tile_tq2_0_dot", kSpvMatmulTileTq20Dot, sizeof(kSpvMatmulTileTq20Dot), 5, kMatmulTileQCounts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_TILE_IQ1_S_DOT, "matmul_tile_iq1_s_dot", kSpvMatmulTileIq1SDot, sizeof(kSpvMatmulTileIq1SDot), 6, kMatmulTileIq3Counts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_TILE_IQ1_M_DOT, "matmul_tile_iq1_m_dot", kSpvMatmulTileIq1MDot, sizeof(kSpvMatmulTileIq1MDot), 6, kMatmulTileIq3Counts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_TILE_IQ2_S_DOT, "matmul_tile_iq2_s_dot", kSpvMatmulTileIq2SDot, sizeof(kSpvMatmulTileIq2SDot), 6, kMatmulTileIq3Counts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_TILE_IQ3_XXS_DOT, "matmul_tile_iq3_xxs_dot", kSpvMatmulTileIq3XxsDot, sizeof(kSpvMatmulTileIq3XxsDot), 6, kMatmulTileIq3Counts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_TILE_IQ2_XXS_DOT, "matmul_tile_iq2_xxs_dot", kSpvMatmulTileIq2XxsDot, sizeof(kSpvMatmulTileIq2XxsDot), 6, kMatmulTileIq3Counts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_TILE_IQ2_XS_DOT, "matmul_tile_iq2_xs_dot", kSpvMatmulTileIq2XsDot, sizeof(kSpvMatmulTileIq2XsDot), 6, kMatmulTileIq3Counts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_TILE_IQ3_S_DOT, "matmul_tile_iq3_s_dot", kSpvMatmulTileIq3SDot, sizeof(kSpvMatmulTileIq3SDot), 6, kMatmulTileIq3Counts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_TILE_TQ1_0_DOT_TALL, "matmul_tile_tq1_0_dot_tall", kSpvMatmulTileTq10Dot, sizeof(kSpvMatmulTileTq10Dot), 5, kMatmulTileQCounts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_TILE_TQ2_0_DOT_TALL, "matmul_tile_tq2_0_dot_tall", kSpvMatmulTileTq20Dot, sizeof(kSpvMatmulTileTq20Dot), 5, kMatmulTileQCounts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_TILE_IQ1_S_DOT_TALL, "matmul_tile_iq1_s_dot_tall", kSpvMatmulTileIq1SDot, sizeof(kSpvMatmulTileIq1SDot), 6, kMatmulTileIq3Counts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_TILE_IQ1_M_DOT_TALL, "matmul_tile_iq1_m_dot_tall", kSpvMatmulTileIq1MDot, sizeof(kSpvMatmulTileIq1MDot), 6, kMatmulTileIq3Counts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_TILE_IQ2_S_DOT_TALL, "matmul_tile_iq2_s_dot_tall", kSpvMatmulTileIq2SDot, sizeof(kSpvMatmulTileIq2SDot), 6, kMatmulTileIq3Counts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_TILE_IQ3_XXS_DOT_TALL, "matmul_tile_iq3_xxs_dot_tall", kSpvMatmulTileIq3XxsDot, sizeof(kSpvMatmulTileIq3XxsDot), 6, kMatmulTileIq3Counts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_TILE_IQ2_XXS_DOT_TALL, "matmul_tile_iq2_xxs_dot_tall", kSpvMatmulTileIq2XxsDot, sizeof(kSpvMatmulTileIq2XxsDot), 6, kMatmulTileIq3Counts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_TILE_IQ2_XS_DOT_TALL, "matmul_tile_iq2_xs_dot_tall", kSpvMatmulTileIq2XsDot, sizeof(kSpvMatmulTileIq2XsDot), 6, kMatmulTileIq3Counts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_TILE_IQ3_S_DOT_TALL, "matmul_tile_iq3_s_dot_tall", kSpvMatmulTileIq3SDot, sizeof(kSpvMatmulTileIq3SDot), 6, kMatmulTileIq3Counts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_ROW_TQ1_0_DOT, "matmul_row_tq1_0_dot", kSpvMatmulRowTq10Dot, sizeof(kSpvMatmulRowTq10Dot), 12, kMatmulRowCounts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_ROW_TQ2_0_DOT, "matmul_row_tq2_0_dot", kSpvMatmulRowTq20Dot, sizeof(kSpvMatmulRowTq20Dot), 12, kMatmulRowCounts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_ROW_IQ1_S_DOT, "matmul_row_iq1_s_dot", kSpvMatmulRowIq1SDot, sizeof(kSpvMatmulRowIq1SDot), 12, kMatmulRowCounts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_ROW_IQ1_M_DOT, "matmul_row_iq1_m_dot", kSpvMatmulRowIq1MDot, sizeof(kSpvMatmulRowIq1MDot), 12, kMatmulRowCounts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_ROW_IQ2_S_DOT, "matmul_row_iq2_s_dot", kSpvMatmulRowIq2SDot, sizeof(kSpvMatmulRowIq2SDot), 12, kMatmulRowCounts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_ROW_IQ3_XXS_DOT, "matmul_row_iq3_xxs_dot", kSpvMatmulRowIq3XxsDot, sizeof(kSpvMatmulRowIq3XxsDot), 12, kMatmulRowCounts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_ROW_IQ2_XXS_DOT, "matmul_row_iq2_xxs_dot", kSpvMatmulRowIq2XxsDot, sizeof(kSpvMatmulRowIq2XxsDot), 12, kMatmulRowCounts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_ROW_IQ2_XS_DOT, "matmul_row_iq2_xs_dot", kSpvMatmulRowIq2XsDot, sizeof(kSpvMatmulRowIq2XsDot), 12, kMatmulRowCounts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_ROW_IQ3_S_DOT, "matmul_row_iq3_s_dot", kSpvMatmulRowIq3SDot, sizeof(kSpvMatmulRowIq3SDot), 12, kMatmulRowCounts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_ROW_IQ4_DOT, "matmul_row_iq4_dot", kSpvMatmulRowIq4Dot, sizeof(kSpvMatmulRowIq4Dot), 12, kMatmulRowCounts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_TILE_IQ4, "matmul_tile_iq4", kSpvMatmulTileIq4, sizeof(kSpvMatmulTileIq4), 5, kMatmulTileQCounts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_TILE_IQ4_TALL, "matmul_tile_iq4_tall", kSpvMatmulTileIq4, sizeof(kSpvMatmulTileIq4), 5, kMatmulTileQCounts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_ROW_IQ4_XS_DOT, "matmul_row_iq4_xs_dot", kSpvMatmulRowIq4XsDot, sizeof(kSpvMatmulRowIq4XsDot), 12, kMatmulRowCounts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_TILE_IQ4_XS, "matmul_tile_iq4_xs", kSpvMatmulTileIq4Xs, sizeof(kSpvMatmulTileIq4Xs), 5, kMatmulTileQCounts, nullptr, 0, kPreserve32, 0) \
    X(K_MATMUL_TILE_IQ4_XS_TALL, "matmul_tile_iq4_xs_tall", kSpvMatmulTileIq4Xs, sizeof(kSpvMatmulTileIq4Xs), 5, kMatmulTileQCounts, nullptr, 0, kPreserve32, 0) \
    X(K_ADD_BIAS, "add_bias", kSpvAddBias, sizeof(kSpvAddBias), 2, nullptr) \
    X(K_GELU, "gelu", kSpvGelu, sizeof(kSpvGelu), 2, nullptr) \
    X(K_LAYER_NORM_ROWS, "layer_norm_rows", kSpvLayerNormRows, sizeof(kSpvLayerNormRows), 4, nullptr) \
    X(K_MIX_ROWS, "mix_rows", kSpvMixRows, sizeof(kSpvMixRows), 4, nullptr) \
    X(K_NORM_ROPE_PARTIAL_STREAMS, "norm_rope_partial_streams", kSpvNormRopePartialStreams, sizeof(kSpvNormRopePartialStreams), 7, nullptr) \
    X(K_ATTENTION_FULL, "attention_full", kSpvAttentionFull, sizeof(kSpvAttentionFull), 7, nullptr) \
    X(K_ATTENTION_FULL_TILE_D64, "attention_full_tile_d64", kSpvAttentionFullTileD64, sizeof(kSpvAttentionFullTileD64), 6, nullptr) \
    X(K_ATTENTION_FULL_TILE_D72, "attention_full_tile_d72", kSpvAttentionFullTileD72, sizeof(kSpvAttentionFullTileD72), 6, nullptr)

#define LLMX_KERNEL_ID(id, name, ...) id,
enum KernelId { LLMX_VULKAN_KERNELS(LLMX_KERNEL_ID) K_COUNT };
#undef LLMX_KERNEL_ID

// A supported weight format's dispatch choices, independent of the requested activation dtype.
enum class RowLayout { values, groups32, q8_pairs, q4_pairs, k_blocks };
struct WeightKernels {
    uint32_t type;
    RowLayout layout;
    KernelId embed, embed_ids, row, dot_row, float_row;
    KernelId tile, tall_tile, bf16_tile, bf16_tall_tile;
    KernelId integer_tile, integer_tall_tile;
    KernelId int8_row, int8_tile, int8_tall_tile;   // the 8-bit builds (--dtype int8) on an integer-dot device; K_COUNT where a type has none
    bool fast_tile, mxfp4;
    bool DeviceProfile::*dot_from;
    size_t DeviceProfile::*moe_from;
    const void* dictionary = nullptr;
    size_t dictionary_bytes = 0;
};

const WeightKernels weight_formats[] = {
    {quant::GGML_TYPE_F32, RowLayout::values, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_F32, K_MATMUL_ROW_F32, K_MATMUL_ROW_F32,
     K_MATMUL_TILE, K_MATMUL_TILE_TALL, K_MATMUL_TILE_BF16, K_MATMUL_TILE_BF16_TALL,
     K_COUNT, K_COUNT, K_COUNT, K_COUNT, K_COUNT, true, false, &DeviceProfile::prefer_integer_dot, &DeviceProfile::moe_tile_from},
    {quant::GGML_TYPE_F16, RowLayout::values, K_EMBED_WF16, K_EMBED_IDS_WF16, K_MATMUL_ROW_WF16, K_MATMUL_ROW_WF16, K_MATMUL_ROW_WF16,
     K_MATMUL_TILE_WF16, K_MATMUL_TILE_WF16_TALL, K_MATMUL_TILE_WF16_BF16, K_MATMUL_TILE_WF16_BF16_TALL,
     K_COUNT, K_COUNT, K_COUNT, K_COUNT, K_COUNT, true, false, &DeviceProfile::prefer_integer_dot, &DeviceProfile::moe_tile_from},
    {quant::GGML_TYPE_BF16, RowLayout::values, K_EMBED_WBF16, K_EMBED_IDS_WBF16, K_MATMUL_ROW_WBF16, K_MATMUL_ROW_WBF16, K_MATMUL_ROW_WBF16,
     K_MATMUL_TILE_WBF16, K_MATMUL_TILE_WBF16_TALL, K_MATMUL_TILE_WBF16_BF16, K_MATMUL_TILE_WBF16_BF16_TALL,
     K_COUNT, K_COUNT, K_COUNT, K_COUNT, K_COUNT, true, false, &DeviceProfile::prefer_integer_dot, &DeviceProfile::moe_tile_from},
    {quant::GGML_TYPE_Q8_0, RowLayout::q8_pairs, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW, K_MATMUL_ROW, K_MATMUL_ROW_FLOAT_X,
     K_MATMUL_TILE, K_MATMUL_TILE_TALL, K_MATMUL_TILE_BF16, K_MATMUL_TILE_BF16_TALL,
     K_MATMUL_TILE_Q8, K_MATMUL_TILE_Q8_TALL, K_MATMUL_VEC_Q8_I8, K_MATMUL_TILE_Q8I8, K_MATMUL_TILE_Q8I8_TALL, true, false, &DeviceProfile::prefer_integer_dot, &DeviceProfile::moe_tile_from},
    {quant::GGML_TYPE_Q4_0, RowLayout::q4_pairs, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_Q4, K_MATMUL_ROW_Q4_DOT, K_MATMUL_ROW_FLOAT_X,
     K_MATMUL_TILE, K_MATMUL_TILE_TALL, K_MATMUL_TILE_BF16, K_MATMUL_TILE_BF16_TALL,
     K_MATMUL_TILE_Q, K_MATMUL_TILE_Q_TALL, K_MATMUL_ROW_Q4_I8, K_MATMUL_TILE_QI8, K_MATMUL_TILE_QI8_TALL, false, false, &DeviceProfile::prefer_integer_dot, &DeviceProfile::moe_tile_from_q4},
    {quant::GGML_TYPE_Q4_1, RowLayout::groups32, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_Q4, K_MATMUL_ROW_Q4_DOT, K_MATMUL_ROW_FLOAT_X,
     K_MATMUL_TILE, K_MATMUL_TILE_TALL, K_MATMUL_TILE_BF16, K_MATMUL_TILE_BF16_TALL,
     K_MATMUL_TILE_Q, K_MATMUL_TILE_Q_TALL, K_MATMUL_ROW_Q4_I8, K_MATMUL_TILE_QI8, K_MATMUL_TILE_QI8_TALL, false, false, &DeviceProfile::prefer_integer_dot, &DeviceProfile::moe_tile_from_q4},
    {quant::GGML_TYPE_Q5_0, RowLayout::groups32, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_Q50, K_MATMUL_ROW_Q50_DOT, K_MATMUL_ROW_FLOAT_X,
     K_MATMUL_TILE, K_MATMUL_TILE_TALL, K_MATMUL_TILE_BF16, K_MATMUL_TILE_BF16_TALL,
     K_MATMUL_TILE_Q50, K_MATMUL_TILE_Q50_TALL, K_MATMUL_ROW_Q50_I8, K_MATMUL_TILE_Q50I8, K_MATMUL_TILE_Q50I8_TALL, false, false, &DeviceProfile::prefer_integer_dot, &DeviceProfile::moe_tile_from_q4},
    {quant::GGML_TYPE_Q5_1, RowLayout::groups32, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_Q5, K_MATMUL_ROW_Q5_DOT, K_MATMUL_ROW_FLOAT_X,
     K_MATMUL_TILE, K_MATMUL_TILE_TALL, K_MATMUL_TILE_BF16, K_MATMUL_TILE_BF16_TALL,
     K_MATMUL_TILE_Q5, K_MATMUL_TILE_Q5_TALL, K_MATMUL_ROW_Q5_I8, K_MATMUL_TILE_Q5I8, K_MATMUL_TILE_Q5I8_TALL, false, false, &DeviceProfile::prefer_integer_dot, &DeviceProfile::moe_tile_from_q4},
    {quant::GGML_TYPE_Q2_K, RowLayout::k_blocks, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_FLOAT_X_Q2, K_MATMUL_ROW_Q2_DOT, K_MATMUL_ROW_FLOAT_X_Q2,
     K_MATMUL_TILE, K_MATMUL_TILE_TALL, K_MATMUL_TILE_BF16, K_MATMUL_TILE_BF16_TALL,
     K_MATMUL_TILE_Q2, K_MATMUL_TILE_Q2, K_COUNT, K_COUNT, K_COUNT, false, false, &DeviceProfile::prefer_integer_dot, &DeviceProfile::moe_tile_from_q4k},
    {quant::GGML_TYPE_Q3_K, RowLayout::k_blocks, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_FLOAT_X_Q3, K_MATMUL_ROW_Q3_DOT, K_MATMUL_ROW_FLOAT_X_Q3,
     K_MATMUL_TILE, K_MATMUL_TILE_TALL, K_MATMUL_TILE_BF16, K_MATMUL_TILE_BF16_TALL,
     K_MATMUL_TILE_Q3, K_MATMUL_TILE_Q3, K_COUNT, K_COUNT, K_COUNT, false, false, &DeviceProfile::prefer_integer_dot, &DeviceProfile::moe_tile_from_q4k},
    {quant::GGML_TYPE_Q4_K, RowLayout::k_blocks, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_K4, K_MATMUL_ROW_K4_DOT, K_MATMUL_ROW_FLOAT_X,
     K_MATMUL_TILE, K_MATMUL_TILE_TALL, K_MATMUL_TILE_BF16, K_MATMUL_TILE_BF16_TALL,
     K_MATMUL_TILE_Q, K_MATMUL_TILE_Q_TALL, K_MATMUL_ROW_K4_I8, K_MATMUL_TILE_QI8, K_MATMUL_TILE_QI8_TALL, false, false, &DeviceProfile::prefer_integer_dot, &DeviceProfile::moe_tile_from_q4k},
    {quant::GGML_TYPE_Q5_K, RowLayout::k_blocks, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_K5, K_MATMUL_ROW_K5_DOT, K_MATMUL_ROW_FLOAT_X,
     K_MATMUL_TILE, K_MATMUL_TILE_TALL, K_MATMUL_TILE_BF16, K_MATMUL_TILE_BF16_TALL,
     K_MATMUL_TILE_Q, K_MATMUL_TILE_Q_TALL, K_MATMUL_ROW_K5_I8, K_MATMUL_TILE_QI8, K_MATMUL_TILE_QI8_TALL, false, false, &DeviceProfile::prefer_integer_dot, &DeviceProfile::moe_tile_from_q5k},
    {quant::GGML_TYPE_Q6_K, RowLayout::k_blocks, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_K, K_MATMUL_ROW_K_DOT, K_MATMUL_ROW_FLOAT_X,
     K_MATMUL_TILE, K_MATMUL_TILE_TALL, K_MATMUL_TILE_BF16, K_MATMUL_TILE_BF16_TALL,
     K_MATMUL_TILE_Q6, K_MATMUL_TILE_Q6, K_MATMUL_ROW_K_I8, K_MATMUL_TILE_Q6I8, K_MATMUL_TILE_Q6I8, false, false, &DeviceProfile::prefer_integer_dot, &DeviceProfile::moe_tile_from},
    {quant::GGML_TYPE_TQ1_0, RowLayout::groups32, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_TQ1_0, K_MATMUL_ROW_TQ1_0_DOT, K_MATMUL_ROW_TQ1_0,
     K_MATMUL_TILE_TQ1_0, K_MATMUL_TILE_TQ1_0_TALL, K_MATMUL_TILE_TQ1_0_BF16, K_MATMUL_TILE_TQ1_0_BF16_TALL,
     K_MATMUL_TILE_TQ1_0_DOT, K_MATMUL_TILE_TQ1_0_DOT_TALL, K_COUNT, K_COUNT, K_COUNT, false, false, &DeviceProfile::lookup_integer_dot, &DeviceProfile::moe_tile_from_q4},
    {quant::GGML_TYPE_TQ2_0, RowLayout::groups32, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_TQ2_0, K_MATMUL_ROW_TQ2_0_DOT, K_MATMUL_ROW_TQ2_0,
     K_MATMUL_TILE_TQ2_0, K_MATMUL_TILE_TQ2_0_TALL, K_MATMUL_TILE_TQ2_0_BF16, K_MATMUL_TILE_TQ2_0_BF16_TALL,
     K_MATMUL_TILE_TQ2_0_DOT, K_MATMUL_TILE_TQ2_0_DOT_TALL, K_COUNT, K_COUNT, K_COUNT, false, false, &DeviceProfile::lookup_integer_dot, &DeviceProfile::moe_tile_from_q4},
    {quant::GGML_TYPE_IQ1_S, RowLayout::groups32, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_IQ1_S, K_MATMUL_ROW_IQ1_S_DOT, K_MATMUL_ROW_IQ1_S,
     K_MATMUL_TILE_IQ1_S, K_MATMUL_TILE_IQ1_S_TALL, K_MATMUL_TILE_IQ1_S_BF16, K_MATMUL_TILE_IQ1_S_BF16_TALL,
     K_MATMUL_TILE_IQ1_S_DOT, K_MATMUL_TILE_IQ1_S_DOT_TALL, K_COUNT, K_COUNT, K_COUNT, false, false, &DeviceProfile::lookup_integer_dot, &DeviceProfile::moe_tile_from_q4, quant::IQ1_GRID, sizeof(quant::IQ1_GRID)},
    {quant::GGML_TYPE_IQ1_M, RowLayout::groups32, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_IQ1_M, K_MATMUL_ROW_IQ1_M_DOT, K_MATMUL_ROW_IQ1_M,
     K_MATMUL_TILE_IQ1_M, K_MATMUL_TILE_IQ1_M_TALL, K_MATMUL_TILE_IQ1_M_BF16, K_MATMUL_TILE_IQ1_M_BF16_TALL,
     K_MATMUL_TILE_IQ1_M_DOT, K_MATMUL_TILE_IQ1_M_DOT_TALL, K_COUNT, K_COUNT, K_COUNT, false, false, &DeviceProfile::lookup_integer_dot, &DeviceProfile::moe_tile_from_q4, quant::IQ1_GRID, sizeof(quant::IQ1_GRID)},
    {quant::GGML_TYPE_IQ2_S, RowLayout::groups32, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_IQ2_S, K_MATMUL_ROW_IQ2_S_DOT, K_MATMUL_ROW_IQ2_S,
     K_MATMUL_TILE_IQ2_S, K_MATMUL_TILE_IQ2_S_TALL, K_MATMUL_TILE_IQ2_S_BF16, K_MATMUL_TILE_IQ2_S_BF16_TALL,
     K_MATMUL_TILE_IQ2_S_DOT, K_MATMUL_TILE_IQ2_S_DOT_TALL, K_COUNT, K_COUNT, K_COUNT, false, false, &DeviceProfile::lookup_integer_dot, &DeviceProfile::moe_tile_from_q4, quant::IQ2_S_GRID, sizeof(quant::IQ2_S_GRID)},
    {quant::GGML_TYPE_IQ3_XXS, RowLayout::groups32, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_IQ3_XXS, K_MATMUL_ROW_IQ3_XXS_DOT, K_MATMUL_ROW_IQ3_XXS,
     K_MATMUL_TILE_IQ3_XXS, K_MATMUL_TILE_IQ3_XXS_TALL, K_MATMUL_TILE_IQ3_XXS_BF16, K_MATMUL_TILE_IQ3_XXS_BF16_TALL,
     K_MATMUL_TILE_IQ3_XXS_DOT, K_MATMUL_TILE_IQ3_XXS_DOT_TALL, K_COUNT, K_COUNT, K_COUNT, false, false, &DeviceProfile::lookup_integer_dot, &DeviceProfile::moe_tile_from_q4, quant::IQ3_XXS_GRID, sizeof(quant::IQ3_XXS_GRID)},
    {quant::GGML_TYPE_IQ2_XXS, RowLayout::groups32, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_IQ2_XXS, K_MATMUL_ROW_IQ2_XXS_DOT, K_MATMUL_ROW_IQ2_XXS,
     K_MATMUL_TILE_IQ2_XXS, K_MATMUL_TILE_IQ2_XXS_TALL, K_MATMUL_TILE_IQ2_XXS_BF16, K_MATMUL_TILE_IQ2_XXS_BF16_TALL,
     K_MATMUL_TILE_IQ2_XXS_DOT, K_MATMUL_TILE_IQ2_XXS_DOT_TALL, K_COUNT, K_COUNT, K_COUNT, false, false, &DeviceProfile::lookup_integer_dot, &DeviceProfile::moe_tile_from_q4, quant::IQ2_XXS_GRID, sizeof(quant::IQ2_XXS_GRID)},
    {quant::GGML_TYPE_IQ2_XS, RowLayout::groups32, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_IQ2_XS, K_MATMUL_ROW_IQ2_XS_DOT, K_MATMUL_ROW_IQ2_XS,
     K_MATMUL_TILE_IQ2_XS, K_MATMUL_TILE_IQ2_XS_TALL, K_MATMUL_TILE_IQ2_XS_BF16, K_MATMUL_TILE_IQ2_XS_BF16_TALL,
     K_MATMUL_TILE_IQ2_XS_DOT, K_MATMUL_TILE_IQ2_XS_DOT_TALL, K_COUNT, K_COUNT, K_COUNT, false, false, &DeviceProfile::lookup_integer_dot, &DeviceProfile::moe_tile_from_q4, quant::IQ2_XS_GRID, sizeof(quant::IQ2_XS_GRID)},
    {quant::GGML_TYPE_IQ3_S, RowLayout::groups32, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_IQ3_S, K_MATMUL_ROW_IQ3_S_DOT, K_MATMUL_ROW_IQ3_S,
     K_MATMUL_TILE_IQ3_S, K_MATMUL_TILE_IQ3_S_TALL, K_MATMUL_TILE_IQ3_S_BF16, K_MATMUL_TILE_IQ3_S_BF16_TALL,
     K_MATMUL_TILE_IQ3_S_DOT, K_MATMUL_TILE_IQ3_S_DOT_TALL, K_COUNT, K_COUNT, K_COUNT, false, false, &DeviceProfile::lookup_integer_dot, &DeviceProfile::moe_tile_from_q4, quant::IQ3_S_GRID, sizeof(quant::IQ3_S_GRID)},
    {quant::GGML_TYPE_IQ4_NL, RowLayout::groups32, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_FLOAT_X, K_MATMUL_ROW_IQ4_DOT, K_MATMUL_ROW_FLOAT_X,
     K_MATMUL_TILE, K_MATMUL_TILE_TALL, K_MATMUL_TILE_BF16, K_MATMUL_TILE_BF16_TALL,
     K_MATMUL_TILE_IQ4, K_MATMUL_TILE_IQ4_TALL, K_COUNT, K_COUNT, K_COUNT, false, false, &DeviceProfile::lookup_integer_dot, &DeviceProfile::moe_tile_from_q4},
    {quant::GGML_TYPE_IQ4_XS, RowLayout::groups32, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_FLOAT_X, K_MATMUL_ROW_IQ4_XS_DOT, K_MATMUL_ROW_FLOAT_X,
     K_MATMUL_TILE, K_MATMUL_TILE_TALL, K_MATMUL_TILE_BF16, K_MATMUL_TILE_BF16_TALL,
     K_MATMUL_TILE_IQ4_XS, K_MATMUL_TILE_IQ4_XS_TALL, K_COUNT, K_COUNT, K_COUNT, false, false, &DeviceProfile::lookup_integer_dot, &DeviceProfile::moe_tile_from_q4},
    {quant::GGML_TYPE_MXFP4, RowLayout::groups32, K_EMBED_MXFP4, K_EMBED_IDS_MXFP4, K_MATMUL_ROW_MXFP4, K_MATMUL_ROW_MXFP4_DOT, K_MATMUL_ROW_MXFP4_FLOAT_X,
     K_MATMUL_TILE_MXFP4, K_MATMUL_TILE_MXFP4_TALL, K_MATMUL_TILE_MXFP4_BF16, K_MATMUL_TILE_MXFP4_BF16_TALL,
     K_MATMUL_TILE_Q8MX, K_MATMUL_TILE_Q8MX_TALL, K_COUNT, K_COUNT, K_COUNT, false, true, &DeviceProfile::lookup_integer_dot, &DeviceProfile::moe_tile_from},
};

const WeightKernels* weight_kernels(uint32_t type) {
    for (const WeightKernels& format : weight_formats) if (format.type == type) return &format;
    return nullptr;
}

// Dense and routed BF16 calls share the float tile's input-rounding variant.
inline KernelId float_tile_kernel(bool tall, Dtype dtype, uint32_t type) {
    const WeightKernels& format = *weight_kernels(type);
    if (dtype == Dtype::bf16) return tall ? format.bf16_tall_tile : format.bf16_tile;
    return tall ? format.tall_tile : format.tile;
}

// Row kernels share dispatch and take the column count as specialization constant 0; the policy selects floats or an activation twin.
inline bool is_row_kernel(KernelId id) {
    if (id == K_MATMUL_ROW_MXFP4 || id == K_MATMUL_ROW_MXFP4_DOT || id == K_MATMUL_ROW_MXFP4_FLOAT_X) return true;
    switch (id) {
    case K_MATMUL_ROW_WF16: case K_MATMUL_ROW_WBF16:
    case K_MATMUL_ROW_FLOAT_X_Q2: case K_MATMUL_ROW_FLOAT_X_Q3:
    case K_MATMUL_ROW_Q2_DOT: case K_MATMUL_ROW_Q3_DOT:
    case K_MATMUL_ROW_IQ4_DOT: case K_MATMUL_ROW_IQ4_XS_DOT:
    case K_MATMUL_ROW_IQ1_S: case K_MATMUL_ROW_IQ1_M:
    case K_MATMUL_ROW_IQ1_S_DOT: case K_MATMUL_ROW_IQ1_M_DOT:
    case K_MATMUL_ROW_IQ3_XXS: case K_MATMUL_ROW_IQ2_XXS:
    case K_MATMUL_ROW_IQ2_XS: case K_MATMUL_ROW_IQ3_XXS_DOT:
    case K_MATMUL_ROW_IQ2_XXS_DOT: case K_MATMUL_ROW_IQ2_XS_DOT:
    case K_MATMUL_ROW_TQ1_0: case K_MATMUL_ROW_TQ2_0: case K_MATMUL_ROW_IQ2_S: case K_MATMUL_ROW_TQ1_0_DOT: case K_MATMUL_ROW_TQ2_0_DOT: case K_MATMUL_ROW_IQ2_S_DOT:
    case K_MATMUL_ROW_IQ3_S: case K_MATMUL_ROW_IQ3_S_DOT:
    case K_MATMUL_ROW: case K_MATMUL_ROW_FLOAT_X: case K_MATMUL_ROW_F32: case K_MATMUL_ROW_F32_ONLY: case K_MATMUL_ROW_Q8W: case K_MATMUL_ROW_Q4: case K_MATMUL_ROW_Q5: case K_MATMUL_ROW_Q50:
    case K_MATMUL_ROW_K4: case K_MATMUL_ROW_K5: case K_MATMUL_ROW_K:
    case K_MATMUL_ROW_Q4_DOT: case K_MATMUL_ROW_Q5_DOT: case K_MATMUL_ROW_Q50_DOT:
    case K_MATMUL_ROW_K4_DOT: case K_MATMUL_ROW_K5_DOT: case K_MATMUL_ROW_K_DOT: case K_MATMUL_VEC_Q8:
    case K_MATMUL_ROW_Q4_I8: case K_MATMUL_ROW_Q5_I8: case K_MATMUL_ROW_Q50_I8: case K_MATMUL_ROW_K4_I8: case K_MATMUL_ROW_K5_I8: case K_MATMUL_ROW_K_I8: case K_MATMUL_VEC_Q8_I8:
        return true;
    default: return false;
    }
}

// Whether a kernel id is a tile kernel: they take their row count as specialization constant 0 and write only their outputs.
inline bool is_tile_kernel(KernelId id) {
    if (id == K_MATMUL_TILE_Q8MX || id == K_MATMUL_TILE_Q8MX_TALL) return true;
    if (id == K_MATMUL_TILE_MXFP4 || id == K_MATMUL_TILE_MXFP4_TALL ||
        id == K_MATMUL_TILE_MXFP4_BF16 || id == K_MATMUL_TILE_MXFP4_BF16_TALL) return true;
    switch (id) {
    case K_MATMUL_TILE_WF16: case K_MATMUL_TILE_WF16_TALL: case K_MATMUL_TILE_WF16_BF16: case K_MATMUL_TILE_WF16_BF16_TALL:
    case K_MATMUL_TILE_WBF16: case K_MATMUL_TILE_WBF16_TALL: case K_MATMUL_TILE_WBF16_BF16: case K_MATMUL_TILE_WBF16_BF16_TALL:
    case K_MATMUL_TILE: case K_MATMUL_TILE_TALL: case K_MATMUL_TILE_Q: case K_MATMUL_TILE_Q_TALL:
    case K_MATMUL_TILE_Q5: case K_MATMUL_TILE_Q50: case K_MATMUL_TILE_Q5_TALL: case K_MATMUL_TILE_Q50_TALL: case K_MATMUL_TILE_Q5I8: case K_MATMUL_TILE_Q50I8: case K_MATMUL_TILE_Q5I8_TALL: case K_MATMUL_TILE_Q50I8_TALL:
    case K_MATMUL_TILE_IQ4: case K_MATMUL_TILE_IQ4_TALL: case K_MATMUL_TILE_IQ4_XS: case K_MATMUL_TILE_IQ4_XS_TALL:
    case K_MATMUL_TILE_Q2: case K_MATMUL_TILE_Q3: case K_MATMUL_TILE_Q6: case K_MATMUL_TILE_Q8: case K_MATMUL_TILE_Q8_TALL:
    case K_MATMUL_TILE_QI8: case K_MATMUL_TILE_QI8_TALL: case K_MATMUL_TILE_Q6I8: case K_MATMUL_TILE_Q8I8: case K_MATMUL_TILE_Q8I8_TALL:
    case K_MATMUL_TILE_IQ1_S_DOT: case K_MATMUL_TILE_IQ1_M_DOT:
    case K_MATMUL_TILE_IQ1_S_DOT_TALL: case K_MATMUL_TILE_IQ1_M_DOT_TALL:
    case K_MATMUL_TILE_IQ1_S: case K_MATMUL_TILE_IQ1_M:
    case K_MATMUL_TILE_IQ1_S_TALL: case K_MATMUL_TILE_IQ1_M_TALL:
    case K_MATMUL_TILE_IQ1_S_BF16: case K_MATMUL_TILE_IQ1_M_BF16:
    case K_MATMUL_TILE_IQ1_S_BF16_TALL: case K_MATMUL_TILE_IQ1_M_BF16_TALL:
    case K_MATMUL_TILE_IQ3_XXS_DOT: case K_MATMUL_TILE_IQ2_XXS_DOT:
    case K_MATMUL_TILE_IQ2_XS_DOT: case K_MATMUL_TILE_TQ1_0_DOT: case K_MATMUL_TILE_TQ2_0_DOT: case K_MATMUL_TILE_IQ2_S_DOT:
    case K_MATMUL_TILE_IQ3_XXS_DOT_TALL: case K_MATMUL_TILE_IQ2_XXS_DOT_TALL:
    case K_MATMUL_TILE_IQ2_XS_DOT_TALL: case K_MATMUL_TILE_TQ1_0_DOT_TALL: case K_MATMUL_TILE_TQ2_0_DOT_TALL: case K_MATMUL_TILE_IQ2_S_DOT_TALL:
    case K_MATMUL_TILE_IQ3_S_DOT: case K_MATMUL_TILE_IQ3_S_DOT_TALL:
    case K_MATMUL_TILE_IQ3_XXS: case K_MATMUL_TILE_IQ2_XXS:
    case K_MATMUL_TILE_IQ2_XS: case K_MATMUL_TILE_TQ1_0: case K_MATMUL_TILE_TQ2_0: case K_MATMUL_TILE_IQ2_S:
    case K_MATMUL_TILE_IQ3_XXS_TALL: case K_MATMUL_TILE_IQ2_XXS_TALL:
    case K_MATMUL_TILE_IQ2_XS_TALL: case K_MATMUL_TILE_TQ1_0_TALL: case K_MATMUL_TILE_TQ2_0_TALL: case K_MATMUL_TILE_IQ2_S_TALL:
    case K_MATMUL_TILE_IQ3_XXS_BF16: case K_MATMUL_TILE_IQ2_XXS_BF16:
    case K_MATMUL_TILE_IQ2_XS_BF16: case K_MATMUL_TILE_TQ1_0_BF16: case K_MATMUL_TILE_TQ2_0_BF16: case K_MATMUL_TILE_IQ2_S_BF16:
    case K_MATMUL_TILE_IQ3_XXS_BF16_TALL: case K_MATMUL_TILE_IQ2_XXS_BF16_TALL:
    case K_MATMUL_TILE_IQ2_XS_BF16_TALL: case K_MATMUL_TILE_TQ1_0_BF16_TALL: case K_MATMUL_TILE_TQ2_0_BF16_TALL: case K_MATMUL_TILE_IQ2_S_BF16_TALL:
    case K_MATMUL_TILE_IQ3_S: case K_MATMUL_TILE_IQ3_S_TALL: case K_MATMUL_TILE_IQ3_S_BF16: case K_MATMUL_TILE_IQ3_S_BF16_TALL:
    case K_MATMUL_TILE_BF16: case K_MATMUL_TILE_BF16_TALL:
        return true;
    default: return false;
    }
}

// The Q8_0 decode kernel's 16-bit and 8-bit builds, which share their builds and dispatch (kVecBuilds).
inline bool is_vec_kernel(KernelId id) { return id == K_MATMUL_VEC_Q8 || id == K_MATMUL_VEC_Q8_I8; }

// The fused routed step's kernels that hold a Q8_0 decode body, whose builds are its one-column ones (kVecBuilds), and all the step's kernels, which name their builds by those.
inline bool moe_step_q8(KernelId id) {
    return id == K_MOE_STEP_UP_K4Q8 || id == K_MOE_STEP_UP_K4Q8_I8 || id == K_MOE_STEP_DOWN_Q8 || id == K_MOE_STEP_DOWN_Q8_I8 ||
           id == K_MOE_STEP_DOWN_Q5Q8 || id == K_MOE_STEP_DOWN_Q5Q8_I8;
}
inline bool moe_step_kernel(KernelId id) { return id >= K_MOE_STEP_ROUTE_S1 && id <= K_MOE_STEP_DOWN_K_I8; }
inline bool moe_step_route_kernel(KernelId id) { return id >= K_MOE_STEP_ROUTE_S1 && id <= K_MOE_STEP_ROUTE; }

// The row kernels that read the 8-bit twin.
inline bool reads_x8(KernelId id) {
    return id == K_MATMUL_ROW_Q4_I8 || id == K_MATMUL_ROW_Q5_I8 || id == K_MATMUL_ROW_Q50_I8 || id == K_MATMUL_ROW_K4_I8 || id == K_MATMUL_ROW_K5_I8 || id == K_MATMUL_ROW_K_I8 || id == K_MATMUL_VEC_Q8_I8;
}

// Whether a row kernel has a one-column build for one-column chunks, which frees the registers of seven unused accumulators.
// The wide Q8_0 row kernel keeps its wide build (docs/VULKAN.md, Kernel notes).
inline bool row_kernel_builds_one_column(KernelId id) {
    return is_row_kernel(id) && id != K_MATMUL_ROW_Q8W;
}

const uint32_t kRowColsWide = 8, kRowColsOne = 1;
// A kernel's pipelines: the wide build, then the one-column, then for the row kernels the wide build grouped by expert, then the Q8_0 decode kernel's further builds.
const int kVariants = 9;

// The Q8_0 reduction forms are selected by the device profile so every build gives a column the same bits.
// The subgroup tree depends on the device's reduction pairs; half-block order is fixed for all builds (docs/src/backends-vulkan.md, Decode builds).
const uint32_t kQ8Tree = 1, kQ8Half = 2;

// Q8_0 pipeline variants name columns, rows, load steps, reduction forms and adjacent column groups.
// Their numbering and specialization constants are in docs/src/backends-vulkan.md, Decode builds.
struct VecBuild {
    uint32_t cols, rows, steps, forms, span;
    uint32_t capacity() const { return cols * span; }
};
const VecBuild kVecBuilds[kVariants] = {{kRowColsWide, 4, 1, kQ8Tree, 1}, {kRowColsOne, 2, 1, 0, 1}, {kRowColsWide, 4, 1, 0, 1},
                                        {2, 2, 1, kQ8Tree, 1}, {4, 4, 1, kQ8Tree, 1}, {16, 4, 2, kQ8Tree, 1}, {16, 4, 2, kQ8Tree, 2},
                                        {kRowColsOne, 1, 4, 0, 1}, {kRowColsOne, 2, 2, 0, 1}};
// Its other one-column builds, one row and four steps and two rows and two steps, which one_column_build picks in place of variant 1 by the call's shape.
const int kVecOneRow = 7, kVecTwoSteps = 8;
// Its builds for plain columns, narrowest first.
const int kVecByWidth[] = {1, 3, 4, 0, 5, 6};

// The forms a Q8_0 decode build takes on a device: those it asks for that the device's profile allows, and the half-block order in every build where the profile allows it.
inline uint32_t vec_forms(int variant, const DeviceProfile& profile) {
    return (kVecBuilds[variant].forms & profile.q8_decode_forms) | (profile.q8_decode_forms & kQ8Half);
}

// Whether a row kernel is one of the Q4 and K-quant families, which have two-row decode builds (shaders/matmul_row.comp, ROWS).
inline bool row_kernel_builds_two_rows(KernelId id) {
    switch (id) {
    case K_MATMUL_ROW_Q4: case K_MATMUL_ROW_Q5: case K_MATMUL_ROW_Q50: case K_MATMUL_ROW_K4: case K_MATMUL_ROW_K5: case K_MATMUL_ROW_K:
    case K_MATMUL_ROW_Q4_DOT: case K_MATMUL_ROW_Q5_DOT: case K_MATMUL_ROW_Q50_DOT: case K_MATMUL_ROW_K4_DOT: case K_MATMUL_ROW_K5_DOT: case K_MATMUL_ROW_K_DOT:
    case K_MATMUL_ROW_Q4_I8: case K_MATMUL_ROW_Q5_I8: case K_MATMUL_ROW_Q50_I8: case K_MATMUL_ROW_K4_I8: case K_MATMUL_ROW_K5_I8: case K_MATMUL_ROW_K_I8:
        return true;
    default: return false;
    }
}

// Pipeline variants for the Q4 and K-quant row families: columns per lane and rows per cluster.
// Every build gives a column the same bits (docs/src/backends-vulkan.md, Decode builds).
struct RowBuild {
    uint32_t cols, rows;
};
const RowBuild kRowBuilds[] = {{kRowColsWide, 1}, {kRowColsOne, 1}, {kRowColsWide, 1}, {2, 2}, {4, 2}, {8, 2}, {16, 2}};
// The two-row builds, narrowest first.
const int kRowByWidth[] = {3, 4, 5, 6};

// The rows a cluster takes in a row kernel's build, by pipeline variant: one but in the two-row builds.
inline uint32_t build_rows(KernelId id, int variant) { return row_kernel_builds_two_rows(id) ? kRowBuilds[variant].rows : 1; }

// The columns a row kernel's build holds, by pipeline variant.
inline uint32_t build_cols(KernelId id, int variant) {
    if (is_vec_kernel(id)) return kVecBuilds[variant].capacity();
    if (row_kernel_builds_two_rows(id)) return kRowBuilds[variant].cols;
    return variant == 1 ? kRowColsOne : kRowColsWide;
}

// The tile kernel's row count, specialization constant 0: the shorter heights fill a device a taller tile would leave idle, the taller reads less shared memory per product.
const uint32_t kTileRowsSmall = 32, kTileRowsShort = 64, kTileRowsTall = 128;   // the small height is variant 1 of the short kernels
// The columns of an integer tile's narrow build, variant 2 (shaders/matmul_tile_q.comp, NARROW).
const size_t kNarrowCols = 16;

// Layouts the shaders fix, which the host must match.
const uint32_t kQ8LanesPerPair = 4;       // matmul_row.comp, wide Q8_0: `group = l / 4u` ("four lanes per pair")
const uint32_t kKQuantLanes = 8;          // matmul_row.comp, Q4_K, Q5_K and Q6_K: `group = l / 8u` ("eight lanes per block")
const size_t kAttentionTileRows = 32;     // attention_tile.comp: `TQ = 32u * RPG`, the query rows of one tile
const size_t kFullTileRows = 64;          // and of one tile of its full builds, whose `RPG` is 2

// The multiply-adds one dispatch of non-causal attention takes at most, about 60 ms on an MI50, well inside the GPU's job timeout.
// The per-row build is about 16 times slower a product, so it takes 16 times fewer.
const size_t kFullTileProducts = size_t(1) << 37;
const size_t kFullRowProducts = size_t(1) << 33;

// A kernel's bindings; `counts` gives each one's array length, one for a plain buffer. A dispatch lists its buffers binding by binding, array elements consecutively.
struct KernelSource {
    const uint32_t* words;
    size_t bytes;
    uint32_t bindings;
    const uint32_t* counts;
    const uint32_t* preserved_words = nullptr;
    size_t preserved_bytes = 0;
    uint32_t controls = 0;
    uint32_t preserved_controls = 0;
};

const uint32_t kMatmulRowCounts[12] = {3, 3, 3, 3, 1, 3, 1, 1, 1, 1, 1, 1};
// The fused routed step's kernels: the row kernel's bindings with their projections, then the routing's scores, ids, weights and counter, or the down kernel's weights and zero row.
const uint32_t kMoeStepRouteCounts[16] = {3, 3, 3, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
const uint32_t kMoeStepUpCounts[12] = {4, 4, 4, 4, 1, 4, 1, 1, 1, 1, 1, 1};
const uint32_t kMoeStepDownCounts[14] = {2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1};
// The integer-dot tile's outputs and weights, three of each, and the reduce's outputs.
const uint32_t kMatmulTileQCounts[5] = {3, 3, 1, 1, 1};
const uint32_t kMatmulTileIq3Counts[6] = {3, 3, 1, 1, 1, 1};
const uint32_t kMatmulTileQ8mxCounts[6] = {3, 3, 1, 1, 1, 3};
const uint32_t kMatmulReduceCounts[2] = {3, 1};

#define LLMX_KERNEL_NAME(id, name, ...) name,
const char* const kKernelNames[K_COUNT] = { LLMX_VULKAN_KERNELS(LLMX_KERNEL_NAME) };
#undef LLMX_KERNEL_NAME

#define LLMX_KERNEL_SOURCE(id, name, ...) {__VA_ARGS__},
const KernelSource kKernels[K_COUNT] = { LLMX_VULKAN_KERNELS(LLMX_KERNEL_SOURCE) };
#undef LLMX_KERNEL_SOURCE
#undef LLMX_VULKAN_KERNELS

// glslc may retain only one width for a repeated execution-mode id; the table declares the full requirement.
// Complete modules use their embedded words, while missing controls are added to a temporary module copy.
VkShaderModuleCreateInfo kernel_module(KernelId id, bool preserve, uint32_t supported, std::vector<uint32_t>& storage) {
    const KernelSource& src = kKernels[id];
    preserve = preserve && src.preserved_words;
    const uint32_t wanted = preserve ? src.preserved_controls : src.controls;
    if (wanted & ~supported) throw std::runtime_error(std::string("vulkan: device lacks float controls for ") + kKernelNames[id]);
    const uint32_t* words = preserve ? src.preserved_words : src.words;
    const size_t count = (preserve ? src.preserved_bytes : src.bytes) / sizeof(uint32_t);
    if (count < 5 || words[0] != 0x07230203u) throw std::logic_error("vulkan: a kernel module without a SPIR-V header");
    struct Mode { uint32_t bit, mode, width, capability; };
    const Mode modes[] = {{kDenorm32, 4459, 32, 4464}, {kSpecial32, 4461, 32, 4466}, {kSpecial64, 4461, 64, 4466}};
    uint32_t entry = 0, present = 0, capabilities = 0;
    size_t mode_end = 0, capability_end = 5;
    for (size_t at = 5; at < count;) {
        const uint32_t op = words[at] & 0xffffu, len = words[at] >> 16;
        if (!len || len > count - at) throw std::logic_error("vulkan: a malformed kernel module");
        if (op == 17 && len == 2) {
            capability_end = at + len;
            for (const auto& m : modes) if (words[at + 1] == m.capability) capabilities |= m.bit;
        } else if (op == 15 && len >= 4) {
            if (entry) throw std::logic_error("vulkan: a kernel module has multiple entry points");
            entry = words[at + 2];
            mode_end = at + len;
        } else if (op == 16 && len >= 3) {
            mode_end = at + len;
            const uint32_t mode = words[at + 2];
            if (mode >= 4459 && mode <= 4463) {
                uint32_t bit = 0;
                for (const auto& m : modes) if (len == 4 && mode == m.mode && words[at + 3] == m.width) bit = m.bit;
                if (!bit || !(wanted & bit) || (present & bit) || words[at + 1] != entry)
                    throw std::logic_error("vulkan: kernel float controls disagree with their declaration");
                present |= bit;
            }
        }
        at += len;
    }
    if (!entry || !mode_end) throw std::logic_error("vulkan: a kernel module without an entry point");
    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = count * sizeof(uint32_t);
    info.pCode = words;
    storage.clear();
    if ((present & wanted) == wanted && (capabilities & wanted) == wanted) return info;
    storage.assign(words, words + count);
    std::vector<uint32_t> extra;
    for (const auto& m : modes) if ((wanted & m.bit) && !(present & m.bit))
        extra.insert(extra.end(), {uint32_t(4u << 16) | 16u, entry, m.mode, m.width});
    storage.insert(storage.begin() + mode_end, extra.begin(), extra.end());
    extra.clear();
    for (const auto& m : modes) if ((wanted & m.bit) && !(capabilities & m.bit)) {
        extra.insert(extra.end(), {uint32_t(2u << 16) | 17u, m.capability});
        for (const auto& other : modes) if (other.capability == m.capability) capabilities |= other.bit;
    }
    storage.insert(storage.begin() + capability_end, extra.begin(), extra.end());
    info.codeSize = storage.size() * sizeof(uint32_t);
    info.pCode = storage.data();
    return info;
}

// The bindings a SPIR-V module writes, a bit each: a storage buffer is read only where its variable, or every member of its block, is NonWritable (GLSL readonly).
uint32_t spirv_writes(const uint32_t* words, size_t count) {
    if (count < 5) throw std::logic_error("vulkan: a kernel module without a header");
    const uint32_t bound = words[3];
    enum : uint32_t { kTypeArray = 28, kTypeRuntimeArray = 29, kTypeStruct = 30, kTypePointer = 32, kVariable = 59, kDecorate = 71, kMemberDecorate = 72 };
    enum : uint32_t { kNonWritable = 24, kBinding = 33, kUniform = 2, kStorageBuffer = 12 };
    // Per id: its binding, the type an array or pointer holds, a struct's members, and which are NonWritable.
    std::vector<uint32_t> binding(bound, UINT32_MAX), inner(bound, 0), members(bound, 0);
    std::vector<uint64_t> fixed_members(bound, 0);
    std::vector<uint8_t> fixed(bound, 0);
    std::vector<std::pair<uint32_t, uint32_t>> vars;   // (pointer type, variable)
    auto id = [&](uint32_t v) {
        if (v >= bound) throw std::logic_error("vulkan: a kernel module id out of bounds");
        return v;
    };
    for (size_t at = 5; at < count;) {
        const uint32_t op = words[at] & 0xffff, len = words[at] >> 16;
        if (!len || at + len > count) throw std::logic_error("vulkan: a malformed kernel module");
        const uint32_t* w = words + at;
        if (op == kDecorate && len >= 4 && w[2] == kBinding) binding[id(w[1])] = w[3];
        else if (op == kDecorate && len >= 3 && w[2] == kNonWritable) fixed[id(w[1])] = 1;
        else if (op == kMemberDecorate && len >= 4 && w[3] == kNonWritable && w[2] < 64) fixed_members[id(w[1])] |= uint64_t(1) << w[2];
        else if ((op == kTypeArray || op == kTypeRuntimeArray) && len >= 3) inner[id(w[1])] = id(w[2]);
        else if (op == kTypePointer && len >= 4) inner[id(w[1])] = id(w[3]);
        else if (op == kTypeStruct && len >= 2) members[id(w[1])] = len - 2;
        else if (op == kVariable && len >= 4 && (w[3] == kStorageBuffer || w[3] == kUniform)) vars.emplace_back(id(w[1]), id(w[2]));
        at += len;
    }
    uint32_t writes = 0;
    for (const auto& v : vars) {
        const uint32_t b = binding[v.second];
        if (b == UINT32_MAX) continue;
        if (b >= 32) throw std::logic_error("vulkan: a kernel binding past 31");
        uint32_t t = inner[v.first];
        while (!members[t] && inner[t]) t = inner[t];
        const bool block_fixed = members[t] && members[t] <= 64 && fixed_members[t] == (members[t] == 64 ? ~uint64_t(0) : (uint64_t(1) << members[t]) - 1);
        if (!fixed[v.second] && !block_fixed) writes |= 1u << b;
    }
    return writes;
}

// The bindings each kernel writes (spirv_writes), the union of its plain and its float-preserving module's.
const std::array<uint32_t, K_COUNT>& kernel_writes() {
    static const std::array<uint32_t, K_COUNT> table = [] {
        std::array<uint32_t, K_COUNT> t{};
        for (int i = 0; i < K_COUNT; ++i) {
            const KernelSource& s = kKernels[i];
            t[i] = spirv_writes(s.words, s.bytes / 4) | (s.preserved_words ? spirv_writes(s.preserved_words, s.preserved_bytes / 4) : 0u);
        }
        return t;
    }();
    return table;
}

// A compiled kernel: module, a layout of `bindings` pushed storage buffers and 128 bytes of push constants, and the pipeline.
struct Kernel {
    VkShaderModule module = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    uint32_t bindings = 0;
    uint32_t buffers = 0;   // sum of the bindings' array lengths
};

} // namespace
} // namespace backend
