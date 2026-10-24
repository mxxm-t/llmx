# `src/quant/iq1.hpp` - IQ1_S and IQ1_M block decoding

This family owns read-only CPU decoding of two 256-value layouts and the shared
2048-entry ternary dictionary. Registry consumers reuse the existing CPU
embedding and dequantized matrix owners; no CPU dot family or writer is added.
Vulkan wiring reuses the existing lookup matrix owners. Read-only CPU/Vulkan qualification includes independent format/operator checks, tiny and released-file HF, raw/chat 16K, split, integration, timing and hosted checks recorded in [STATUS](../STATUS.md).

IQ1_S (ID 19) uses 50 bytes: a little-endian binary16 scale at 0..1,
32 low index bytes at 2..33, and eight little-endian words at 34..49.
Each word holds four high-index triples at bits 0..11, an odd scale selector
at bits 12..14 and a shift sign at bit 15, shared by 32 values.

IQ1_M (ID 29) uses 56 bytes: low index bytes at 0..31, high-index/shift
nibbles at 32..47 and four little-endian scale words at 48..55.
The high nibble of scale word k supplies bits 4*k through 4*k+3 of the
common binary16 encoding. Each word's low twelve bits hold four three-bit
odd-scale selectors, each for sixteen values. Each high-index/shift nibble
holds three index bits and one shift sign for eight values.

Both decode eight-value dictionary vectors, with index low byte plus three
high bits. For a three-bit scale selector s, the value is
`(d * (1 + 2*s)) * (ternary + shift)`, with shift +1/8 or -1/8.
Byte loads handle arbitrary alignment and IQ1_S's alternating block alignment.
Finite binary16 scales produce exact F32 weights, including signed zero.

The `iq1` CTest checks every dictionary index and lane against independently
pinned base-three format data, logical-field packing, all finite half scales,
all odd scale selectors, both shift signs, consecutive unaligned blocks,
input immutability and output guards. It also checks exact scalar versus
prospective packed reconstruction over every finite half, odd multiplier,
ternary value and shift sign; this establishes arithmetic identity without
claiming Vulkan execution or model qualification.

See [dictionary provenance](quant-iq1-grid.md) for the fixed format data.
