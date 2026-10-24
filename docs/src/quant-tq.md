# `src/quant/tq.hpp` - paired ternary weight decoders

`src/quant/tq.hpp` owns the paired read-only block decoders. Both layouts
contain 256 values and end with a little-endian binary16 scale. `tq_value`
applies that scale to the centered code `code - 1`; no encoder is provided.

TQ1_0 is 54 bytes: 48 five-trit bytes, four four-trit bytes, then the scale.
The first 32 packed bytes expand as five planes of 32 values. The next 16
bytes expand as five planes of 16 values. The final four bytes expand as four
planes of four values. For each byte and plane, the format first multiplies
in an 8-bit value, then maps it to a trit with `(wrapped * 3) >> 8`. The
wraparound is part of decoding arbitrary packed bytes; it is not equivalent
to ordinary base-three digit extraction.

TQ2_0 is 66 bytes: 64 packed bytes followed by the scale. Each 32-byte half
contains four low-to-high two-bit planes of 32 values. Each two-bit code is
centered by subtracting one. Code 3 therefore decodes to `+2 * d`, even though
the usual ternary encoder only emits codes 0, 1 and 2.

The native `tq-format` test checks every packed byte pattern, every finite
binary16 scale, unaligned consecutive blocks, source/output guards and the
registry-driven CPU matrix product against its own field oracle. This records
CPU format support only. The shared Vulkan owners now implement both formats,
with focused product, embedding and range checks passing on Radeon VII.
Independent model-file and performance qualification remain pending in [STATUS](../STATUS.md).

The paired tiny HF preparation lives in `tests/tq.py`. It reuses
`tests/block_fixture.py` for dense tied/untied and MoE models, including the
existing whole-block split layout, and `tests/spec_decode.py` for exact decoded
weights. Seed 12345, raw packed bytes and signed finite half scales determine
all weights; periodic blocks include signed zero and subnormal scales. Norms
and routers remain F32. These are test packers, not runtime quantize writers.

`python tools/gen_baseline.py tq --output-dir DIR` prepares
`baseline_tq1_0.json`, `baseline_tq1_0_split.json`, `baseline_tq2_0.json` and
`baseline_tq2_0_split.json` with the existing pinned CPU HF environment and
routing-gap refusal. No HF reference generation or model comparison is claimed
by this preparation. The `tq1-0` and `tq2-0` components consume those references
through the shared 150-logit/12-NLL checker. Initial F32 execution keeps its
existing bounds. Narrowed execution requires prospectively generated and
reviewed TQ-family budgets; absent budgets are an error, not permission to use
another family's tolerance.
