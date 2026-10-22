# `src/quant/quant.hpp` - quantization kernels + registry

Weight decoders, block quantization kernels and their registry, in namespace `quant`.

- `dequantize_row_f16` / `dequantize_row_bf16`: read little-endian 16-bit words and widen exactly through `core/fp16.hpp` / `core/bf16.hpp`. Each block is one value in two bytes. Both formats are read-only; neither registry entry has a quantizer. These decoders serve raw conversion, CPU embedding and the CPU's batched float products. Registration does not add Vulkan weight kernels.

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
- `dequantize_row_q2_K` (in `k_quants.hpp`): read-only 256-value blocks with sixteen affine groups, decoded for conversion, CPU embedding and generic float products. No Q2_K quantizer is registered.
- `dequantize_row_q3_K` (in `k_quants.hpp`): read-only 256-value blocks with sixteen signed-scale groups and signed 3-bit values, decoded for conversion, CPU embedding and generic float products. No Q3_K quantizer is registered.
- `dequantize_row_q6_K` (in `k_quants.hpp`): Q6_K super-block of 256 values in 210 bytes - 128 low
  nibbles, 64 bytes of high 2-bit pairs, 16 int8 group scales, f16 super-block
  scale. Registered READ-ONLY: llmx must load it because common GGUF converters
  upgrade selected tensors to it inside an otherwise Q4_0 file, but nothing here
  produces it and a quantizer would be unused code.
- `QuantType`: an implemented type's `StorageType` metadata and block-wise
  (de)quantize routines. The registry takes its name and sizes from
  `storage_type`, so it defines no second storage layout.
- `Registry::instance().get(id)`: the implemented type for a GGML id, or null when llmx has no decoder for it.
  The one registry fills itself with `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ3_S`, `IQ4_NL`, `IQ4_XS` and `MXFP4` on first use and never changes after, so no caller sets it up and any thread may read it. The [README support matrix](../../README.md#quantization-types) distinguishes those decoders from backend support and quantize writers.
  `F32` is registered as a block of one value in 4 bytes.
- `row_bytes(type, nin, rows = 1)` is owned by `types.hpp` and remains available
  through this header. It sizes any known storage layout, including types
  absent from the decoder registry; successful sizing does not establish
  that an inference or conversion operation supports the type.

K-quant layouts and shared sub-scale decoding live in `k_quants.hpp`. The
type ids and block sizes the registry names live in `types.hpp`, whose page
([quant-types](quant-types.md)) lists what a new type takes.

IQ3_S is read-only, decoded by `iq3_s.hpp` ([quant-iq3-s](quant-iq3-s.md)), and supported for CPU/Vulkan inference. Its recorded scope and evidence are in STATUS.
The registry exposes its block decoder for raw conversion and callers that
already use registered decoders. Backend qualification is a separate gate.

`dequantize_row_iq4_nl` and `IQ4_NL_VALUES` own IQ4_NL decoding here: each 18-byte block holds a binary16 scale and 32 indices into the signed, nonuniform 16-value table. Low nibbles address values 0..15 and high nibbles values 16..31. The decoded weight is the scale times the table value, with no offset correction. Negative and zero scales retain the format's zero signs. The registry is read-only; the CPU fused float row dot uses the same table. `iq4-nl` checks all finite scale/code/position combinations against a separate mathematical oracle, and `roundtrip` checks the CLI decoder against `tests/spec_decode.py`. The Python `iq4-nl` component adds tiny dense/MoE HF checks, and the pinned 0.6B file passes its file-exact CPU, MI50 and Radeon checks. Original-weight quality approval remains separate. Published Vulkan support uses the same format through its shared shader lookup decoder, signed integer row dots and tiles with wider float products as fallback; functional/device qualification is recorded in STATUS, with measured speed limits retained as follow-up work ([backends-vulkan](backends-vulkan.md)).

CPU Q8_0 decode keeps the original F32 inputs through its float dot; Q4_0, Q4_1, Q4_K, Q5_K and Q6_K decode takes the integer dots in `backends/cpu/q8_dots.hpp`.
Batched prompt rows of Q8_0, Q4_0 and Q4_1 dequantize through these block routines (`docs/src/backends-cpu.md`).


MXFP4 is read-only, decoded by `mxfp4.hpp` ([quant-mxfp4](quant-mxfp4.md)). The generic CPU prompt path widens its blocks to F32, while decode uses 16-bit activation dots. Registration also enables raw conversion and embedding; it does not imply Vulkan support.

Q5_1 is read-only: `dequantize_row_q5_1` reads 32 affine values from a 24-byte block (half scale, half minimum, four fifth-bit bytes, then 16 nibble bytes). CPU dense and routed calls reuse the existing dequantized products, with no quantized activation dot added.

Q5_0 support is read-only: `dequantize_row_q5_0` reads a binary16 scale, four fifth-bit bytes and 16 nibble bytes from each 22-byte block. The decoded weight is `d * (code - 16)`; negative scales give negative zero at code 16. Byte loads handle consecutive blocks with alternating two-byte alignment. Registry-driven CPU embedding, dense/grouped and routed products reuse their existing arithmetic: F16/int8 requests use wider F32 products, and BF16 requests round inputs through the existing BF16 owner. Vulkan uses the shared decoder and compile-time Q5 row/tile variants with codes centered before the dot. Focused CPU/Radeon operators, five-policy tiny file-exact HF consumers and native MI50 integer-dot operator checks are recorded in STATUS. Tiny consumers explicitly select the independently calibrated Q5_0 family through the shared tolerance owner; its routing-turnover rows remain refused. The original-seed native MI50 int8 tiny consumer passes. MI50 split-int8 and converted-file consumers also pass, and the released mixed-sidecar MTP check matches target output identities; it does not establish all-expert execution, HF sidecar quality or performance. One matched P512/G128 MI50 model-benchmark screen is recorded in [STATUS](../STATUS.md) and its evidence record. It contains one pair per cell; the small differences are not attributed, and repetition/layout controls are appropriate if claiming a regression or speedup. This is not MTP or server-throughput qualification.

IQ4_XS has read-only CPU and Vulkan support. `iq4_xs_scale` reads each signed six-bit subscale from its four low bits and two high bits; `dequantize_row_iq4_xs` scales eight groups of 32 values in a 136-byte block. Both IQ4 formats use `dequantize_iq4_group` and the existing `IQ4_NL_VALUES` codebook. The binary16 scale is multiplied by the signed subscale before the lookup value, preserving both signs of zero. The native `iq4-nl` test covers both formats, while `raw-blocks` and `roundtrip` share independently packed XS fixtures and the pure/NumPy spec decoders. Vulkan reuses its existing lookup row and tile families; STATUS records local qualification and its exact scope.
