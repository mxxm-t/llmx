# `src/quant/k_quants.hpp` - K-quant decoding

Read-only block decoders for Q2_K, Q3_K, Q4_K, Q5_K and Q6_K. Each block describes 256
weights, occupying 84, 110, 144, 176 or 210 bytes respectively. Q4_K and Q5_K share the
6-bit scale/min decoder `get_scale_min_k4`; Q5_K additionally reconstructs high
weight bits. Q6_K reconstructs signed 6-bit values with per-group signed scales.

`quant.hpp` registers these decoders; `types.hpp` holds the type ids and
block sizes. For Q4_K, Q5_K and Q6_K the CPU backend uses integer dots for decode, the prompt dots for batched
rows at least 4096 wide, and the dequantized-row path for narrower ones
(`docs/src/backends-cpu.md`). No K-quant writer exists.

Q2_K and Q3_K use the CPU's generic dequantized float products; neither has a
quantizer. Q2_K's first sixteen bytes
hold a scale nibble and a minimum nibble for each 16-value group. The next
64 bytes interleave four two-bit planes within each 128-value half; the final
four bytes hold the two F16 super-block scales. Each decoded value is
`(d * scale) * code - dmin * minimum`. Vulkan float products and embeddings use
the shared decoder in `shaders/qdecode.glsl`; packed row and tile products share
code reconstruction and range repair in `shaders/k23_dot.glsl`. The CPU registry
is not the Vulkan capability table.

Q3_K stores 32 high-bit bytes, 64 low-code bytes, twelve packed scale bytes
and one F16 super-block scale. Each of its sixteen 16-value groups has a signed
6-bit scale; the low two code bits and the high-bit plane reconstruct signed
3-bit values in -4..3. The CPU registry and Vulkan weight descriptor use the
same decoded values. CPU F16/int8 requests use wider F32 products. Vulkan
integer-dot profiles use block-int16 rows and eligible prompt tiles for these
requests; other profiles retain F32. Explicit F32 retains original inputs, and
BF16 rounds inputs through the existing matrix precision owner. No Q3_K writer
is registered.

Q4_K and Q5_K pack the sub-scales and sub-mins of their eight groups in twelve bytes: groups 0 to 3 are the low six bits of bytes 0 to 3 (scales) and 4 to 7 (mins), and groups 4 to 7 the nibbles of bytes 8 to 11 with the top two bits of the earlier bytes above them.
