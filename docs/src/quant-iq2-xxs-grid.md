# `src/quant/iq2_xxs_grid.inc` - IQ2_XXS format dictionary

The canonical 256 eight-value magnitude vectors, packed as little-endian uint64 values. The decoder in `iq2_xxs.hpp` and the Vulkan weight descriptor use this format data. The separate test dictionaries in `tests/spec_decode.py` and `tests/iq2_small.cpp` supply independent expected values.

The format source is ggml-common.h at mx revision `2fe50c3b322c54cfb984d68a6bdb64b8abf9e3ea`. The SHA-256 of the dictionary's 2048 decoded bytes is `05826b5d3e472a3a78f196be62ac78acf81df0f909626e12ab9fa2a5d490dd54`. This is required file-format data, not copied implementation logic.
