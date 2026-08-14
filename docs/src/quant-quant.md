# `src/quant/quant.hpp` — quantization kernels + registry

Block quantization kernels, in namespace `quant`.

- `quantize_row_q8_0(src, dst, nblocks)`: compress 32 floats into a 2-byte f16
  scale + 32 int8 values per block (clamped to [-127, 127]).
- `dequantize_row_q8_0(src, dst, nblocks)`: reverse.
- `quantize_row_q4_0` / `dequantize_row_q4_0`: Q4_0 block = 2-byte f16 scale
  (`d = amax/7`) + 16 bytes of nibbles; each byte holds value `j` in the low
  nibble and `j+16` in the high nibble, stored unsigned 0..15 where the true
  value is `nibble - 8`.
- `QuantType`: description of a quant type (block size, bytes/block,
  block-wise (de)quantize routines).
- `Registry` / `register_builtins()`: lookup a quant type by GGML id. Registers
  `Q8_0`, `Q4_0`, and `F32`.

Adding a new quant is one file + one registry entry.

> Q4_0 has block kernels only — no fused backend matmul. `infer::Model` uses a
> generic dequant-row-to-f32 + dot path for it (correct but slower than Q8_0).
> A fused AVX2 dequant+FMA Q4_0 matvec is a follow-up.

