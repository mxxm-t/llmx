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
Read-only CPU/Vulkan support is qualified with applicable model-file and performance evidence and the approved scoped Vulkan chat baseline exception in [PRECISION](../PRECISION.md). See [STATUS](../STATUS.md).

The paired tiny HF preparation lives in `tests/tq.py`. It reuses
`tests/block_fixture.py` for dense tied/untied and MoE models, including the
existing whole-block split layout, and `tests/spec_decode.py` for exact decoded
weights. Seeds 12345 for TQ1_0 and `TQ2_SEED = 12346` for TQ2_0, raw packed
bytes and signed finite half scales determine all weights; periodic blocks
include signed zero and subnormal scales. Norms
and routers remain F32. These are test packers, not runtime quantize writers.

`python tools/gen_baseline.py tq --output-dir DIR` prepares
`baseline_tq1_0.json`, `baseline_tq1_0_split.json`, `baseline_tq2_0.json` and
`baseline_tq2_0_split.json` with the existing pinned CPU HF environment and
routing-gap refusal. The pinned HF generation completed all twelve fixtures
across the four files; this is reference generation, not a runtime comparison.
The first TQ2_0 ordinary-MoE seed 12345 routing-gap refusal is preserved;
HF-only screening selected seed 12346 before any candidate comparison.
The `tq1-0` and `tq2-0` components consume those references
through the shared 150-logit/12-NLL checker. Initial F32 execution keeps its
existing bounds. The reviewed TQ F16/BF16 families are now frozen before
narrowed comparisons; absent family data remains an error, not permission to
use another family's tolerance. CPU tensor-split HF checks pass at widths 2
and 4; pinned Vulkan TP2 and layer-split coverage is recorded below.

Family calibration uses the same fixture constructors and existing
rounding owner. From the pinned HF environment, run each family separately
with a fresh output path:

```
python tools/calibrate_dtype.py --family tq1-0 --output tq1-0-budget.json
python tools/calibrate_dtype.py --family tq2-0 --output tq2-0-budget.json
```

Each family constructs 30 models: ordinary and split layouts, three variants,
and its own fixture seed (12345 for TQ1_0, `TQ2_SEED` for TQ2_0), plus
seeds 0, 1, 2 and 3. It records F16 and BF16 errors against exact-weight F32
HF, with F32 sums and routers, for 60 reference rows. Int8 requests have no
TQ block-int8 path and use the F16 budget through the completed-path owner;
F32-weight products and routers remain F32. The existing 2x margin and
routing-turnover refusals are unchanged. Each output
starts from the checked-in budget and adds or reproduces only its named family.
Both new sections are frozen from 120 accepted rows, with no routing refusals
and every prior family/class limit unchanged. These are reference errors,
not released-model bounds. [Calibration identities and exact budgets](../benchmarks/tq-calibration-20261024.json).

Ordinary CPU and Radeon VII auto/BF16 consumers now pass 150 full-vocabulary
comparisons and 12 NLL checks per format and policy. Int8 requests also pass
300 full-vocabulary comparisons and 24 NLL checks on each backend using the
existing fallback owner; this is not native TQ int8 coverage. The selected
bounds follow actual matrix-path witnesses, while the retained text logs do not
include raw per-capture witness metadata. CPU auto retains F32
arithmetic and strict bounds; Radeon auto uses block-int16 TQ products with
retained F32 operations. BF16 uses input rounding and F32 sums on both.
The earlier explicit-F32 checks also pass. CPU tensor-split HF checks pass
at widths 2 and 4. Pinned [ordinary MI50](../benchmarks/tq-e5-tiny-hf-20261024.json),
[Vulkan TP2](../benchmarks/tq-capture-launch-20261024.json) and
[layer-split](../benchmarks/tq-layer-final-20261024.json) HF coverage also passes.
Released-file depth and performance qualification is complete with the approved scoped Vulkan chat baseline exception. [Scoped dtype results and artifact pins](../benchmarks/tq-tiny-dtypes-20261024.json) and [int8-request fallback results](../benchmarks/tq-int8-fallback-20261024.json).
