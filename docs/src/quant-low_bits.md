# `src/quant/low_bits.hpp` - Q1_0 and Q2_0 block decoding

Q1_0 and Q2_0 are read-only registry decoders. Each block starts with a
little-endian binary16 scale and then 16 packed bytes. Q1_0 holds 128 sign
bits, least-significant bit first, with 0/1 mapping to `-d/+d`. Q2_0 holds
64 two-bit fields, least-significant field first, mapping codes 0 through 3
to `-d`, zero, `+d` and `+2d`. Both use the generic CPU products after row
decoding; no quantizer or specialized dot is registered.

The `low-bits` native test builds blocks directly and compares every code
position against an independent field oracle. It checks every finite
binary16 scale, including negative and signed-zero values, unaligned block
starts, multiple blocks, zero-block calls and output guards. Code 3 in Q2_0
is checked even though ternary producers do not emit it.

The [Vulkan owner](backends-vulkan.md) describes the separate shader decoder
and its embedding, tile and row consumers. Focused operator checks do not
establish model, depth or performance qualification.
