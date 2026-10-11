# `src/quant/k_quants.hpp` - K-quant decoding

Read-only block decoders for Q4_K, Q5_K and Q6_K. Each block describes 256
weights, occupying 144, 176 or 210 bytes respectively. Q4_K and Q5_K share the
6-bit scale/min decoder `get_scale_min_k4`; Q5_K additionally reconstructs high
weight bits. Q6_K reconstructs signed 6-bit values with per-group signed scales.

`quant.hpp` registers these decoders; `types.hpp` holds the type ids and
block sizes. The CPU backend uses integer dots for decode, the prompt dots for batched
rows at least 4096 wide, and the dequantized-row path for narrower ones
(`docs/src/backends-cpu.md`). No K-quant writer exists.

Q4_K and Q5_K pack the sub-scales and sub-mins of their eight groups in twelve bytes: groups 0 to 3 are the low six bits of bytes 0 to 3 (scales) and 4 to 7 (mins), and groups 4 to 7 the nibbles of bytes 8 to 11 with the top two bits of the earlier bytes above them.
