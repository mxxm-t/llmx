# `src/quant/iq1_grid.inc` - shared IQ1 format dictionary

The 2048 eight-value ternary vectors are fixed format data. Each lane stores
ternary + 1 in two bits. Two vectors occupy one uint32, even index in the low
sixteen bits, lane zero in the low two bits. `iq1.hpp` owns the 1024-word array
and shift-based accessor, independent of host byte order.

The published data is pinned to [ggml-common.h](https://github.com/mxxm-t/mx-llama.cpp/blob/407d0bb1f12eaa49882d2714a93fe774eb6806f2/ggml/src/ggml-common.h)
and [ggml-quants.c](https://github.com/mxxm-t/mx-llama.cpp/blob/407d0bb1f12eaa49882d2714a93fe774eb6806f2/ggml/src/ggml-quants.c)
at revision `407d0bb1f12eaa49882d2714a93fe774eb6806f2`.
Only fixed dictionary data is retained; the decoder is independently written
for llmx's byte-oriented registry interface. The signed-byte `iq1s_grid`
and two-bit `kgrid_1bit_2048` representations were independently extracted and
all 16384 lanes cross-checked before packing the runtime words.

SHA-256 pins:

- `ggml-common.h`: `9d0d28bf7bf11f85ab43af21d2c489e4ed1f3599528d2e45bb3bc533442f8d4b`.
- `ggml-quants.c`: `27d97b5853bfebba8fc6af575a98d4aa3e66a2109b77fdc25d7cdd61c67c31a5`.
- Published signed-byte dictionary, index order and lane zero first: `07540ffc1aeaf6ad4d97e96b0fcc765aae39671d4ae4a27bbd0e796fde167c6a`.
- Runtime 1024 little-endian uint32 words: `e9ffebfc997ad6023063a667c0bf4eedd8ebf7f56b3da399cbc7d1b1f7ea52c6`.
- Independent native fixture `tests/iq1_grid.inc`, base-three vector numbers serialized as little-endian uint16: `964d7b98109a8e48d64ff22ebf527705d9e47f02108b893311300e82b4cba376`.

The test fixture represents each vector as sum((ternary[lane]+1)*3^lane)
and decodes with division and remainder, rather than the runtime's bit fields.
Block fields and qualification scope are described in [IQ1 decoding](quant-iq1.md).
