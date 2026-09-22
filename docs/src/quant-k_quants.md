# `src/quant/k_quants.hpp` - K-quant decoding

Read-only block decoders for Q4_K, Q5_K and Q6_K. Each block describes 256
weights, occupying 144, 176 or 210 bytes respectively. Q4_K and Q5_K share the
6-bit scale/min decoder `get_scale_min_k4`; Q5_K additionally reconstructs high
weight bits. Q6_K reconstructs signed 6-bit values with per-group signed scales.

`quant.hpp` registers these decoders; `format/gguf.hpp` holds file type IDs and
block sizes. The CPU backend uses fused Q4_K, Q5_K and Q6_K dots for decode, and
the generic dequantized-row path for batched prefill. No K-quant writer exists.
