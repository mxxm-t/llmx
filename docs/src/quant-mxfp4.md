# `src/quant/mxfp4.hpp` - MXFP4 block decoding

The read-only MXFP4 block decoder in namespace `quant`.
Each 17-byte block holds an E8M0 exponent and 16 bytes of E2M1 nibbles for 32 values.
Low nibbles precede high nibbles, as in Q4_0.

`MXFP4_VALUES` holds the doubled E2M1 values as signed integers.
`MXFP4_FIRST_OVERFLOW_EXPONENT` names the first exponent at which a code can decode beyond finite f32: 253, where the largest doubled code gives `12 * 2^125 = 3 * 2^127`.
`mxfp4_scale` reads their scale, 2^(e - 128), from a 256-entry table of f32 bits computed at compile time.
Exponents 0 and 1 retain their subnormal scales; no half conversion occurs.
`dequantize_row_mxfp4` widens whole blocks into f32 rows.

The GGUF interpretation follows the approved quantization plan in STATUS:
both zero codes give +0; exponent 255 uses the same power-of-two construction,
so its nonzero outputs are signed 2^127 or infinity.
This differs from OCP E8M0, which reserves exponent 255 for NaN.
The native `mxfp4` test checks every exponent and code in every position against
an independent sign/exponent/mantissa oracle, including output guards and unaligned input.
It also checks every scale against a mathematical power of two and the decode scale's acceptance and exact product over all weight exponents and representative F32 exponent, mantissa and sign boundaries (524,288 combinations).

The registry offers this decoder to GGUF reads, raw conversion, embedding and CPU prefill.
CPU decode uses the packed dots described below; dense and routed prompt rows retain original F32 activations. Vulkan refuses this type until its separate implementation lands.
CPU MXFP4 support is merged; speed work remains open in STATUS. The separate Vulkan branch has an unresolved real-MoE ranking criterion.
The CPU implementation in `backends/cpu/q8_dots.hpp` uses an AVX2
nibble lookup and 16-bit activations. Scales whose product is outside the normal
f32 range, potentially infinite weights, or a nonfinite fast sum use decoded
f32 weights and double products over the same quantized activations.
The native test checks one-hot inputs at all finite-weight exponents, odd block
counts and cancellation under extreme scales. High-exponent dots retain decoded infinities even when the activation scale is small, and grouped normal and underflow columns are checked together. Removing the high-exponent guard in an isolated test build fails the new case. No performance result is claimed.
