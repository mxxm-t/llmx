# `src/quant/iq3_xxs.hpp` - IQ3_XXS block decoding

IQ3_XXS stores 256 values in 98 bytes, GGML type 18. Bytes 0..1 hold the
little-endian binary16 scale d, bytes 2..65 hold 64 dictionary indices,
and bytes 66..97 hold eight little-endian 32-bit words.

Each word describes 32 consecutive values. Its high four bits select scale
s; its low 28 bits hold four seven-bit sign codes. Each sign code describes
eight consecutive values, formed from two adjacent four-byte dictionary
vectors. The low seven sign bits are explicit; bit seven is their parity,
so every eight-lane sign mask has even parity. `iq3_xxs_signs` expands that
format field without a second table. All 128 codes agree with the canonical
format sign table.

A lane's magnitude is d * (1 + 2*s) / 4 times its dictionary byte; its sign
bit negates that result. Byte reads handle unaligned blocks and word fields;
shifts select dictionary lanes independently of host byte order. Multiplication
and final negation preserve both zero signs. Finite half scales have at most
11 significant bits, the odd scale uses at most five, and dictionary magnitudes
are at most 62, so finite decoded values are exactly representable in F32.
Their maximum magnitude is 31474672. Nonfinite scales are not a finite-range
guarantee.

The registry entry is read-only, with no quantize writer. CPU products use the
existing registry fallback; this initial owner adds no specialized matrix path.
Format, model, Vulkan and performance qualification are separate work recorded
in [STATUS](../STATUS.md).

The fixed dictionary is format data from `ggml/src/ggml-common.h` at revision
`750127a5173ac6c29c2c159a10404bf184c421ab`. Its raw Git LF blob has SHA-256
`0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c`;
the equivalent CRLF snapshot used by earlier format records hashes to
`3ac6eed12695ceea1acd18f556845023a45f92f6ce0530c18c70bb10450207ea`.
The 1024 little-endian dictionary bytes have SHA-256
`46e35f5a997efdee6c99ce57854c8a0d4f0ff8ca57e5e8a60c0793ea580acf5d`.
The decoder is independently implemented from these fields; no reference
decoding algorithm is copied.
