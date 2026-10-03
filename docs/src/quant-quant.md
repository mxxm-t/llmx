# `src/quant/quant.hpp` - quantization kernels + registry

Block quantization kernels, in namespace `quant`.

- `quantize_row_q8_0(src, dst, nblocks)`: compress 32 floats into a 2-byte f16
  scale + 32 int8 values per block (clamped to [-127, 127]).
- `dequantize_row_q8_0(src, dst, nblocks)`: reverse.
- `quantize_row_q4_0` / `dequantize_row_q4_0`: Q4_0 block = 2-byte f16 scale (`d = amax/7`) + 16 bytes of nibbles; each byte holds value `j` in the low nibble and `j+16` in the high nibble, stored unsigned 0..15 where the true value is `nibble - 8`.
  The decode computes `d*(nibble - 8)`, so under a negative scale, which `quantize` never writes but other files carry, nibble 8 gives -0 as the format does.
  Encoding normally multiplies by the reciprocal of the F32 scale, preserving its existing rounding. If that reciprocal is infinite for a tiny positive scale, it divides that block into a 32-float local array and packs it with multiplier 1, keeping the values bounded before the integer casts and leaving the ordinary packing loop unchanged. A scale rounded to zero in F32 keeps zero codes; a scale representable in F32 can still round to zero in the stored binary16 format. Rounding a tiny scale in F32 can reach the -8 clamp as well as the ordinary -7 to 7 codes. `quantize-range` checks exact codes across power-of-two scales, half-way cases and rounded subnormal scales under gradual underflow; nonfinite inputs are outside its scope.
- `dequantize_row_q4_1`: Q4_1 block = f16 scale + f16
  min + 16 bytes of nibbles (20 bytes). The nibble is unsigned and the block
  carries its own offset, so the value is `d*q + m`, not `d*(q-8)`.
  llmx reads Q4_1 and never writes it, so the registry gives it no quantizer, as it gives the K-quants none; the tests pack Q4_1 weights with their own (`tests/quantizers.hpp`).
- `dequantize_row_q6_K` (in `k_quants.hpp`): Q6_K super-block of 256 values in 210 bytes - 128 low
  nibbles, 64 bytes of high 2-bit pairs, 16 int8 group scales, f16 super-block
  scale. Registered READ-ONLY: llmx must load it because common GGUF converters
  upgrade selected tensors to it inside an otherwise Q4_0 file, but nothing here
  produces it and a quantizer would be unused code.
- `QuantType`: an implemented type's `StorageType` metadata and block-wise
  (de)quantize routines. The registry takes its name and sizes from
  `storage_type`, so it defines no second storage layout.
- `Registry::instance().get(id)`: the implemented type for a GGML id, or null when llmx has no decoder for it.
  The one registry fills itself with `Q8_0`, `Q4_0`, `Q4_1`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_NL`, `MXFP4` and `F32` on first use and never changes after, so no caller sets it up and any thread may read it.
  `F32` is registered as a block of one value in 4 bytes.
- `row_bytes(type, nin, rows = 1)` is owned by `types.hpp` and remains available
  through this header. It sizes any known storage layout, including types
  absent from the decoder registry; successful sizing does not establish
  that an inference or conversion operation supports the type.

K-quant layouts and shared sub-scale decoding live in `k_quants.hpp`. The
type ids and block sizes the registry names live in `types.hpp`, whose page
([quant-types](quant-types.md)) lists what a new type takes.

`dequantize_row_iq4_nl` and `IQ4_NL_VALUES` own IQ4_NL decoding here: each 18-byte block holds a binary16 scale and 32 indices into the signed, nonuniform 16-value table. Low nibbles address values 0..15 and high nibbles values 16..31. The decoded weight is the scale times the table value, with no offset correction. Negative and zero scales retain the format's zero signs. The registry is read-only; the CPU fused row dot uses the same table. `iq4-nl` checks all finite scale/code/position combinations against a separate mathematical oracle, and `roundtrip` checks the CLI decoder against `tests/spec_decode.py`. The Python `iq4-nl` component adds tiny dense/MoE HF checks, and the pinned 0.6B file passes its file-exact CPU check. STATUS records completed support checks and assessed performance tradeoffs, with original-weight quality approval and eventual landing-head checks still pending; Vulkan refuses this type at load.

CPU Q8_0 decode keeps the original F32 inputs through its float dot; Q4_0, Q4_1, Q4_K, Q5_K and Q6_K decode takes the integer dots in `backends/cpu/q8_dots.hpp`.
Batched prompt rows of Q8_0, Q4_0 and Q4_1 dequantize through these block routines (`docs/src/backends-cpu.md`).


MXFP4 is read-only, decoded by `mxfp4.hpp` ([quant-mxfp4](quant-mxfp4.md)). The generic CPU prompt path widens its blocks to F32, while decode uses 16-bit activation dots. Registration also enables raw conversion and embedding; it does not imply Vulkan support.
