# Remaining Q1/Q2 split-fixture F32 proposal

The user approved these five exact fixtures on 2026-10-27, separately from the
earlier three-fixture amendment. They are installed for qualification. The user requires independent
correctness proof before merging; passing a changed threshold is not that proof.

| Exact split fixture | Original logit limit | Approved logit limit |
|---|---:|---:|
| Q1_0 dense-untied | 0.000020000 | 0.000102016 |
| Q1_0 dense-tied | 0.000020000 | 0.000102016 |
| Q1_0 MoE | 0.000020000 | 0.000102016 |
| Q2_0 dense-tied | 0.000020000 | 0.000098201 |
| Q2_0 MoE | 0.000020000 | 0.000098201 |

The rule is the same reference-only rule as the earlier amendment: the existing
margin plus the original fixed HF full/cached F32 maximum for the same family
and split shape. No candidate error selects the limits. The
[exact identities and unrounded values](q1-q2-split-f32-proposal-20261027.json)
pin the family, variant, configuration and decoded and packed weight hashes.
Only completed all-F32 matrix witnesses qualify, in CPU or Vulkan CLI or raw
captures. NLL remains 1e-5. Operator, decoder, other-policy, unlisted-fixture,
released-model and depth limits do not change.

Each fixture fails the original limit without relying on its new quant path:
main 9d00fb0a5 holding independently decoded F32 weights reproduces every one
of the 13 captured batched rows bit for bit. For the four newly surveyed
fixtures, all 64 greedy IDs also match, and both arms' NLL errors are below
1e-5. Decode bytes differ. See the [four-fixture controls](q1-q2-split-main-controls-20261027.json)
and the [Q2 tied control](q1-q2-approved-f32-progress-20261027.json).
CPU widths two and four pass all ten existing group-reference comparisons,
while their original strict HF failures remain recorded in the
[group report](q1-q2-cpu-groups-progress-20261027.json).

These results establish an existing arithmetic discrepancy at the captured
positions, not correctness of every operator or device execution. Independent
decoder and CPU/Radeon VII/MI50 operator checks are separate evidence; remaining
GPU model/split, released-file and depth checks are still required. A wider
margin could admit a small implementation error, so this remains an explicit
policy decision. Main's Q2 split-MoE decode error of 0.000128508 would still
exceed the proposed allowance: this is an empirical rule, not a universal
rounding bound or a blanket exception for main.

Original failed checks remain retained. Approval permits completing
qualification with these exact identities; it does not grant a pass or merge.
The separately discovered Q1 split-MoE NLL failure remains unresolved and
is not covered by this logit-only amendment.
