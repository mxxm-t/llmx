# `src/quant/iq2_xs.hpp` - IQ2_XS block decoding

IQ2_XS stores 256 values in 74 bytes (GGML type 17): a little-endian binary16 scale, 32 little-endian 16-bit words, and eight packed scale bytes. Each word selects one of 512 eight-value dictionary vectors with its low nine bits; the upper seven bits hold the sign code. Each byte supplies two scale nibbles, each covering 16 values. The sign code expands to eight signs by setting bit 7 to make the byte's parity even.

A vector's scale is `d * (0.5 + s) / 4`, and its magnitudes come from the selected dictionary row. Byte reads handle unaligned blocks and preserve signed zero. The registry entry is read-only and uses the existing generic CPU product path; no quantizer or specialized CPU dot is added.

The independent dictionary bytes are pinned from `ggml/src/ggml-common.h` at revision `2fe50c3b322c54cfb984d68a6bdb64b8abf9e3ea`. The 512 little-endian 64-bit entries serialize to SHA-256 `06e47aaca60b4dc1d9b5a3f34540437058a6b142b4d7a59d5ded769b4d1bf1de`. The implementation uses only these fixed format values, not reference decoder code. Python raw-block tests independently pack all indexes, sign codes and scale nibbles and compare pure and NumPy decoders. Backend qualification is tracked in [STATUS](../STATUS.md).
