# `src/quant/iq2_s.hpp` - IQ2_S block decoding

IQ2_S stores 256 values in 82 bytes, GGML type 22. Bytes 0..1 hold the
little-endian binary16 scale. Bytes 2..33 hold low dictionary-index bytes,
34..65 hold independent sign bits, 66..73 hold two high index bits per vector,
and 74..81 hold two scale nibbles per byte.

For eight-value vector g, its index is the low byte plus 256 times the
two-bit field at offset 2*(g%4) in high byte g/4. Vector lane i selects byte i
of the dictionary word. Each 16-value sub-block has a scale nibble, low
nibble first. Its multiplier is (1 + 2*nibble)/8. The half scale times
this multiplier times the dictionary magnitude is negated when the lane's
sign bit is set. Byte loads permit unaligned input and preserve both zero
signs. The decoded value is exactly representable in F32 for finite half scales.

The dictionary is fixed format data, not a generated quantizer or an inference
algorithm copied from another runtime. IQ2_S differs from the IQ2_XXS and
IQ2_XS layouts and tables; those remain unsupported. CPU/Vulkan read-only inference and the
pinned released-file qualification are recorded in [STATUS](../STATUS.md).
Storage decoding alone does not qualify another file or architecture.

Format data source: `ggml/src/ggml-common.h` at
`750127a5173ac6c29c2c159a10404bf184c421ab`, SHA-256
`3ac6eed12695ceea1acd18f556845023a45f92f6ce0530c18c70bb10450207ea`.
The 8192 little-endian dictionary bytes have SHA-256
`e1aa1473412b0552c2174c30ef22ab4073f6a181b85a17056e8249bd2932fd88`.
