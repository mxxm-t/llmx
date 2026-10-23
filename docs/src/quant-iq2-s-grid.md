# `src/quant/iq2_s_grid.inc` - IQ2_S format dictionary

The fixed format data contains 1024 eight-byte dictionary vectors, each packed
least-significant byte first into a 64-bit integer. The format decoder owns
the array and selects lanes with shifts; it does not depend on host byte order.
See [IQ2_S decoding](quant-iq2-s.md) for the block fields and provenance.
