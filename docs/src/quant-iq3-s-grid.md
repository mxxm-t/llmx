# `src/quant/iq3_s_grid.inc` - fixed IQ3_S dictionary

The 512 packed four-byte words are the IQ3_S format's positive dictionary vectors, included by the CPU decoder. Vulkan uploads that same array once per backend. The IQ3-specific row and tile shader builds cache the words in workgroup shared memory; embedding reads the buffer directly. `iq3_s.glsl` owns decoding through a caller-supplied word accessor, and `lookup_cache.glsl` supplies the matrix-build cache shared with IQ2_S. The plain initializer list keeps one runtime definition of the constants.

[The decoder page](quant-iq3-s.md) records the lane order, format provenance, decoded-byte hash and independent test dictionary. Changing these entries changes the storage format, not a runtime tuning policy.
