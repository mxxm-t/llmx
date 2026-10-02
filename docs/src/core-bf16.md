# `src/core/bf16.hpp` - BF16 conversion

`f32_to_bf16` rounds binary32 to BF16, nearest with ties to even. It preserves the sign of zero, subnormals and infinity; a NaN remains a NaN even when its original payload occupies only discarded bits. `bf16_to_f32` widens the stored bits exactly. The functions use integer arithmetic and bit copies, with no external library or hardware BF16 requirement.

The CPU backend uses these conversions when a matrix call explicitly requests BF16 activations. `tests/dtype.cpp` checks every finite BF16 encoding, both signs, values below, at and above every rounding midpoint, overflow, infinities and NaNs against a mathematical nearest-neighbour reference.
