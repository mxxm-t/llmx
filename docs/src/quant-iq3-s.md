# `src/quant/iq3_s.hpp` - IQ3_S block decoding

`dequantize_row_iq3_s(src, dst, nblocks)` reads GGML type 21, with
256 values per 110-byte block. The block fields, in byte order, are a
little-endian binary16 scale `d`, 64 low dictionary-index bytes, eight
high-index bytes, 32 sign bytes and four packed scale bytes.

For value `i` within a block, let `g = i / 4`. Its dictionary index is
`qs[g] + 256 * ((qh[g / 8] >> (g % 8)) & 1)` and its vector component is
`i % 4`. Each 32-value sub-block has a scale nibble, low nibble first,
giving multiplier `1 + 2 * nibble`. The decoder multiplies `d` by that
multiplier and then by the positive dictionary component, then negates
the result if bit `i % 8` of sign byte `i / 8` is set. This order preserves
both zero signs. Byte loads accept unaligned blocks.

`IQ3_S_GRID` holds 512 packed four-byte vectors. The literal format data
lives once in `iq3_s_grid.inc`. Vulkan uploads the same array into one
immutable backend-owned buffer. `iq3_s.glsl` implements the same field
decoding through a caller-supplied word accessor; the specialized row and
tile builds use `lookup_cache.glsl` to stage the 512 words in workgroup
shared memory, while embedding reads them directly from the buffer. Byte
lane zero is the least significant byte; shifts select the lane. The
dictionary is distinct from IQ3_XXS and the IQ4 formats.

The format declaration and fixed dictionary are pinned to ggml revision
`eefc4e7321c869496146697d63362f073941aed6`, `ggml/src/ggml-common.h`,
whose SHA-256 is
`0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c`.
The 2048 decoded little-endian dictionary bytes have SHA-256
`bd1af4945a1717c65610b0284e4628b9a1ba3b306fae3a06f6e5f597356e349f`.
These are fixed format constants; their historical empirical selection
is not regenerated here. The decoding implementation is written from the
field layout and index formulas.

The native `iq3-s` test uses a separate decoded-byte dictionary and a
mathematical double-precision binary16 oracle. It checks every dictionary
index at every vector position and component lane, isolated high-index
and sign bits, every scale nibble at each sub-block, all finite binary16
scales including signed zeros, three unaligned consecutive blocks,
unchanged input and output guards. Registry metadata, read-only status,
zero-block calls and partial-width refusal are checked as well.
`tests/spec_decode.py` supplies independent pure and NumPy decoders with
a separately pinned decoded-byte dictionary. `tests/raw_blocks.py`
packs logical fields independently, checks exact values and zero signs,
fixed hand-computed bytes, the dictionary hash and pure/NumPy parity.
These format checks do not establish model quality or backend performance.

The CPU backend part of the native test checks 101,760 product/control values
across F32, F16, BF16 and int8 requests. F16 and int8 retain the F32 product
path, BF16 rounds inputs, and actual `MatrixPaths` are checked. Independent
BF16 tie cases and single-expert grouped, head, additive and routed calls are
included. Focused native operator checks pass on CPU, Radeon VII and MI50.
Candidate tiny-HF checks pass 150 logits and 12 NLL comparisons for each of
the four request dtypes; focused TP2 checks pass 12 logits and 6 NLL
comparisons per dtype. The exact released file passes 6 ranking checks and 8
NLL cases, with an 8-token CLI/server smoke. Integrated 16K raw/chat checks
pass for the recorded F16 and F32 paths. These checks support read-only IQ3_S CPU/Vulkan inference for the
recorded model and execution scope. See [STATUS](../STATUS.md) for the
qualification evidence and limits.
