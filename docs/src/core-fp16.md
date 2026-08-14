# `src/core/fp16.hpp` — IEEE 754 binary16 conversion

From-scratch half <-> float conversion, no libraries.

- `f32_to_f16(float) -> uint16_t`: rebias exponent to half range, handle
  inf/nan, overflow->inf, subnormal underflow/normalization, round-to-nearest.
- `f16_to_f32(uint16_t) -> float`: sign/exponent/mantissa reconstruction,
  including subnormal normalization and inf/nan.

Used by the Q8_0 kernels to store/read the per-block f16 scale.
