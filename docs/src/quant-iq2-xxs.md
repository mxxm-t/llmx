# `src/quant/iq2_xxs.hpp` - IQ2_XXS block decoding

IQ2_XXS stores 256 values in 66 bytes (GGML type 16): a little-endian binary16 scale followed by eight groups of 32 values. Each group has four one-byte dictionary indexes and one little-endian word. The word contains four seven-bit sign codes in its low 28 bits and a scale nibble in its high four bits. Each sign code expands to eight signs by setting bit 7 to make the full byte have even parity.

A vector selects eight byte magnitudes from the 256-entry dictionary. Its scale is `d * (0.5 + s) / 4`; multiplication and sign application preserve the format's signed zeros. The decoder reads bytes explicitly, so packed blocks may be unaligned. The registry entry is read-only and uses the existing generic CPU product path; this change adds no quantizer or specialized CPU dot.

The independent dictionary bytes are pinned from `ggml/src/ggml-common.h` at revision `2fe50c3b322c54cfb984d68a6bdb64b8abf9e3ea`. The 256 little-endian 64-bit entries serialize to SHA-256 `05826b5d3e472a3a78f196be62ac78acf81df0f909626e12ab9fa2a5d490dd54`. The implementation uses only these fixed format values, not reference decoder code. Python raw-block tests independently pack all indexes, sign codes and scale nibbles and compare pure and NumPy decoders. Backend qualification is tracked in [STATUS](../STATUS.md).
