# `src/core/fp16.hpp` - IEEE 754 binary16 conversion

From-scratch half <-> float conversion, no libraries.

- `f32_to_f16(float) -> uint16_t`: rebias exponent to half range, handle
  inf/nan, overflow->inf, subnormal underflow/normalization, round-to-nearest.
- `f16_to_f32(uint16_t) -> float`: sign/exponent/mantissa reconstruction,
  including subnormal normalization and inf/nan.

Both subnormal paths were wrong until 2026-09-19 and the bugs were invisible
in the original synthetic tests, which did not exercise subnormal block scales.
Decode incremented the exponent per normalizing shift instead of
decrementing, making every subnormal 2^(2k) too large (16x for Q6_K super-block
scales). Encode applied the normal path's 0xfff rounding bias after the shift
had already happened, emitting a normal half about 200x too large. The decode
bug alone made real published Q4_0 files produce pure garbage, since their
`token_embd` is Q6_K. `tests/roundtrip.py` now carries a tensor whose block
scale lands in the subnormal band, with a RELATIVE bound.

Used by the quant codecs to read f16 block scales (`quant/quant.hpp`, `quant/k_quants.hpp`) and to write them when quantizing to Q8_0 or Q4_0, the types `quantize` writes, or to Q4_1, whose quantizer only the tests call.
The F16 weight decoder also uses exact widening. The CPU backend uses it for scalar F16 weight tails and KV cache values past a row's last group of eight; its vector weight lanes, dot scales and KV groups of eight convert through F16C. `tests/half_weights.cpp` checks every finite stored encoding against a mathematical widening oracle, including signed zeros and subnormals.
