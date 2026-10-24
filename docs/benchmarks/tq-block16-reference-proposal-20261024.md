# TQ block-int16 depth reference correction

Status: approved by BOSS on 2026-10-24: "Approve scoped reference correction". Approval fixes the reference method for the stated scope; it is not a qualification or merge pass.

Before this correction, the depth rule in [PRECISION](../PRECISION.md#independent-hf-depth-acceptance-agreed-2026-10-01) required literal F16 matrix-input rounding for the F16 class. Vulkan's qualified block-int16 implementation instead scales each group of 32 F32 inputs to signed 16-bit integers. Rule 4 permits that representation after its F16 calibration and range checks. The two input-rounding methods produce different mathematical models even when both use F16 KV storage.

## Evidence

The following TQ1_0 raw comparisons score the same frozen 512 positions after a 16K prefix. Each HF arm uses the file's independently decoded weights, the same pinned HF implementation and logical F16 KV writes. Counts use the unchanged exact/top-two rule and 0.1 margin.

| Chosen output | HF reference | Exact | Accepted tie | Miss |
| --- | --- | ---: | ---: | ---: |
| Vulkan F32 matrices | F32 matrices | 507 | 5 | 0 |
| Vulkan default block-int16 | F32 matrices | 501 | 10 | 1 |
| Vulkan default block-int16 | Literal-F16 matrices | 482 | 17 | 13 |
| Vulkan default block-int16 | Block-int16 reconstruction | 507 | 5 | 0 |
| HF literal-F16 matrices | Block-int16 reconstruction | 481 | 15 | 16 |
| HF block-int16 reconstruction | Literal-F16 matrices | 481 | 17 | 14 |

The first control supports the cache implementation in this workload. The HF-only differences establish sensitivity to matrix-input rounding without any llmx kernel involved. The block-int16 result supports the actual default path, but was diagnostic under the policy written before approval. It does not prove every operation exact or waive the other gates.

The independent block-int16 producer implements the ordinary mathematical quantizer: per group of 32, `d = amax / 32767`, `id = 32767 / amax`, round `x * id` halfway away from zero, clamp to signed codes from -32767 through 32767, then reconstruct `q * d`. Zero blocks produce zero. Nonfinite blocks and blocks requiring the shader's extreme-range branch are refused. HF then uses F32 matrix products and accumulation, rather than llmx's packed integer dot. Existing independent operator and range checks cover that separate implementation concern.

The producer checks all 197 Linear module names and records per-cell calls, rows and blocks. The audit confirms 33 calls and 16895 input rows for each internal Linear, and 543 head rows because unscored prompt chunks request only one discarded head row. All 28 cache layers record 33 writes and 16895 rows. The direct/chunk control is bit-identical over 64 tokens. The 512-by-151936 output is finite and independently checked against its stored top-50 values. No OOM occurred and the owned container was removed.

Source, receipt, reference and per-position evidence is retained in [the cache control](tq-f32-f16kv-control-20261024.json) and [the block-int16 diagnostic](tq-block16-hf-diagnostic-20261024.json). Producer SHA256: `ae7a0ae18960f81b7416ebf826092adb97c50d2aa75c315a83574e47b8508aec`. The diagnostic was written after the original failure and has already been compared with one candidate cell; this was explicitly a retrospective proposal, not a predeclared pass.

## Original proposal (historical)

For the released TQ1_0 and TQ2_0 files below, use one source-derived block-int16 reconstruction policy for every Linear in each complete Vulkan default raw/chat depth workload whose executed matrix paths are all block-int16. Keep F16 KV writes and all retained F32 non-matrix operations. Do not select a reference per position or take the better of two results.

- TQ1_0: `e505d615403d960c79f3a970912c61e41bf32f33e0ec29060fe265fbabdc9812`.
- TQ2_0: `047d7f218e106cbd421c21e6f2a81b188cf6f47bcc1044e5cc0be5d194ac3f0a`.

The exact/top-two rule, 0.1 margin and all 512 positions stay unchanged. Keep the original literal-F16 comparisons, all misses and the full-F32 quality companion beside the corrected comparison. Tiny-model calibration against the F16 class, short rankings/NLL, F32 and BF16 checks, CPU references, range checks, batch/split identity and performance requirements stay unchanged. Other files and arithmetic paths are outside this amendment.

The existing depth policy was jointly agreed by LDEV and XDEV under BOSS's delegation. Renew that agreement, or obtain BOSS's decision if the reviewer is unavailable, before treating this method as the gate. Do not describe the proposal as an approved tolerance increase: no numerical tolerance changes are proposed.

If accepted, freeze this producer and method, complete the three remaining raw/chat references for the two files, audit all four comparisons, and retain failures. Approval permits qualification; it does not grant a merge pass. A later attempt to generalize the method to other quant families needs its own explicit scope and review.

## Approval record

BOSS approved the scoped correction in the development conversation on 2026-10-24 after seeing the independent HF-only disagreement and the retained Vulkan comparison. The producer above is frozen unchanged. Complete the remaining TQ1_0 chat and TQ2_0 raw/chat references, then audit all four against the unchanged rule. The remaining three references are running on rig CPUs0-3; their completion and comparison review are not yet claimed. The historical proposal section preserves the request as submitted, including its pre-approval instructions.
