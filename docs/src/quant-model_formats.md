# `src/quant/model_formats.hpp` - DeepSeek-V4.1's own storage formats

`quant::model_formats` quantizes and dequantizes a block in place in F32, as DeepSeek-V4.1's kernels store its caches (docs/DEEPSEEK4.md, section 4.5): `qdq_fp8_power`, the window's FP8 E4M3 under a power-of-two scale, `qdq_fp4_power`, the index keys' and queries' FP4 E2M1 under a power-of-two scale, and `qdq_fp4_e4m3`, the entries' FP4 E2M1 under an E4M3 scale.
A power-of-two scale is `power_scale`: `ceil_log2`'s bit rule (the exponent field less 127, plus one where any mantissa bit is set) of the block's floored largest magnitude times the format's inverse maximum, one F32 multiply, then `pow2`; the floors are 1e-4 for FP8 and 6 x 2^-126 for FP4.
The E4M3 scale is the code nearest amax / 6, amax floored at 6 x 2^-9 so it is never zero, saturating at 448 as the kernels' conversion does.
`e4m3_code` and `e2m1_code` choose codes by exact comparisons, ties to even, each zero keeping its sign; `e4m3_value` and `e2m1_value` decode them, and a stored value is one F32 multiply of a code's value and the scale, so every backend writes the same bytes for the same inputs.
The packed layouts (docs/DEEPSEEK4.md, step 9c) are not here yet; CTest `deepseek4-ops` holds these functions to E4M3 and E2M1 written from their text.
