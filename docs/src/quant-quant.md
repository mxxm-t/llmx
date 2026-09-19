# `src/quant/quant.hpp` — quantization kernels + registry

Block quantization kernels, in namespace `quant`.

- `quantize_row_q8_0(src, dst, nblocks)`: compress 32 floats into a 2-byte f16
  scale + 32 int8 values per block (clamped to [-127, 127]).
- `dequantize_row_q8_0(src, dst, nblocks)`: reverse.
- `quantize_row_q4_0` / `dequantize_row_q4_0`: Q4_0 block = 2-byte f16 scale
  (`d = amax/7`) + 16 bytes of nibbles; each byte holds value `j` in the low
  nibble and `j+16` in the high nibble, stored unsigned 0..15 where the true
  value is `nibble - 8`.
- `quantize_row_q4_1` / `dequantize_row_q4_1`: Q4_1 block = f16 scale + f16
  min + 16 bytes of nibbles (20 bytes). The nibble is unsigned and the block
  carries its own offset, so the value is `d*q + m`, not `d*(q-8)`.
- `dequantize_row_q6_K` (in `k_quants.hpp`): Q6_K super-block of 256 values in 210 bytes - 128 low
  nibbles, 64 bytes of high 2-bit pairs, 16 int8 group scales, f16 super-block
  scale. Registered READ-ONLY: llmx must load it because llama.cpp upgrades
  selected tensors to it inside an otherwise Q4_0 file, but nothing here
  produces it and a quantizer would be unused code.
- `QuantType`: description of a quant type (block size, bytes/block,
  block-wise (de)quantize routines).
- `Registry` / `register_builtins()`: lookup a quant type by GGML id. Registers
  `Q8_0`, `Q4_0`, `Q4_1`, `Q4_K`, `Q5_K`, `Q6_K` and `F32`.

K-quant layouts and shared sub-scale decoding live in `k_quants.hpp`. Adding
a quant also requires GGUF type/size entries in `format/gguf.hpp`.

> Q4_0 has block kernels only — no fused backend matmul. `CpuBackend::matmul` uses a
> generic dequant-row-to-f32 + dot path for it (correct but slower than Q8_0).
> A fused AVX2 dequant+FMA Q4_0 matvec is a follow-up.

