# `src/quant/quant.hpp` — quantization kernels + registry

Q8_0 block quantization, in namespace `quant`.

- `quantize_row_q8_0(src, dst, nblocks)`: compress 32 floats into a 2-byte f16
  scale + 32 int8 values per block (clamped to [-127, 127]).
- `dequantize_row_q8_0(src, dst, nblocks)`: reverse.
- `QuantType`: description of a quant type (block size, bytes/block,
  block-wise (de)quantize routines).
- `Registry` / `register_builtins()`: lookup a quant type by GGML id. Registers
  `Q8_0` and `F32`.

Adding a new quant is one file + one registry entry.
