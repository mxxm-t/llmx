# `src/quant/low_bits.hpp` - Q1_0 block decoding

Q1_0 is a read-only registry decoder. Each 18-byte block starts with a little-endian binary16 scale and then 16 packed bytes holding 128 sign bits, least-significant bit first. Codes 0/1 map to `-d/+d`, retaining negative scales and signed zero. The CPU uses the generic products after decoding; no quantizer or specialized CPU dot is registered.

The `low-bits` native check compares packed positions and every finite binary16 scale against an independent field oracle, including unaligned block starts, multiple blocks, zero-block calls and output guards. The [Vulkan owner](backends-vulkan.md) describes the shader decoder and its embedding, tile and row consumers. [Q1](../Q1.md) and [STATUS](../STATUS.md) record model and performance qualification.
