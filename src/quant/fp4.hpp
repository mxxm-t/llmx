#pragma once
#include <cstdint>

namespace quant {
// Shared E2M1 codebook, doubled to signed integers. Both zero codes decode as +0.
inline constexpr int8_t FP4_VALUES[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
} // namespace quant
