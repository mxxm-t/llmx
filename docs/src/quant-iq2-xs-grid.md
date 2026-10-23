# `src/quant/iq2_xs_grid.inc` - IQ2_XS format dictionary

The canonical 512 eight-value magnitude vectors, packed as little-endian uint64 values. The decoder in `iq2_xs.hpp` and the Vulkan weight descriptor use this format data. The separate test dictionaries in `tests/spec_decode.py` and `tests/iq2_small.cpp` supply independent expected values.

The format source is ggml-common.h at mx revision `2fe50c3b322c54cfb984d68a6bdb64b8abf9e3ea`. The SHA-256 of the dictionary's 4096 decoded bytes is `06e47aaca60b4dc1d9b5a3f34540437058a6b142b4d7a59d5ded769b4d1bf1de`. This is required file-format data, not copied implementation logic.
