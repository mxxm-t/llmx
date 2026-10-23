# `src/quant/iq3_xxs_grid.inc` - IQ3_XXS format dictionary

The fixed format data contains 256 four-byte dictionary vectors packed
least-significant byte first into 32-bit integers. Lane magnitudes range
from 4 to 62. The format decoder owns the array and selects lanes with shifts,
without depending on host byte order or input alignment.
See [IQ3_XXS decoding](quant-iq3-xxs.md) for field ordering and provenance.
