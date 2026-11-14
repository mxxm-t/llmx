// The launch shapes the host and the kernels both decide by, so a grid and the kernel it launches cannot disagree (docs/src/backends-gpu.md).
#pragma once

namespace gpu_shape {

// MXFP4's quad rows (kernels/int16.gpu, mxfp4_quad_rows): kQuadLanes lanes a row, the column's twin staged in shared memory up to kQuadValues values.
constexpr unsigned kQuadLanes = 4;
constexpr unsigned long long kQuadValues = 8192;
// A one-column 8-bit MXFP4 row takes the quad rows at k_rows' 32-lane cluster where it has at least 16 blocks, a multiple of four, and at most kQuadValues values.
constexpr bool quad_rows(unsigned cluster, unsigned long long blocks) { return cluster == 32 && blocks >= 16 && blocks % 4 == 0 && blocks * 32 <= kQuadValues; }

}  // namespace gpu_shape
