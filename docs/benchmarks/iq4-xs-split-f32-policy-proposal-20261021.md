# IQ4_XS split-MoE F32 reference-sensitivity allowance (approved; affected requalification complete)

Policy decision: the user approved this reference-sensitivity allowance on
2026-10-21, after the original generic-bound failure. It applies only to F32
logits for the exact existing synthetic IQ4_XS split-MoE fixture identity, and
only when all completed matrix-path witnesses are F32. This is a post-failure
policy amendment, not a passed gate or a mathematical bound on runtime error.
The shared test-bound owner matches the approved allowance to the exact fixture
identity and completed F32 path witnesses. The affected CPU, TP2 and TP4
requalification is complete under this criterion. This qualifies only those
retained calls and does not waive hosted CI or broader landing checks; current
status is recorded in STATUS.

| Quantity | Absolute logit error |
| --- | ---: |
| Current F32 limit | 0.00002000000000000000 |
| Maximum independent HF F32 full/cached discrepancy | 0.00006875395774841309 |
| Approved exact-fixture limit | 0.00008875395774841308 |

The reference maximum comes from split-MoE seed 1, not from a candidate runtime
result. The retained fixed sweep covers ordinary and split dense-untied,
dense-tied and MoE models at seeds 12345, 0, 1, 2 and 3. All 30 models retained
their selected expert sets. The original full-forward HF results reproduce the
committed goldens. These are calibration cases, not held-out validation. Raw
results and the original broader, unapproved proposal are retained in
[the reference measurement](iq4-xs-f32-proposal-20261021.json).

The approved scope is narrower than that historical proposal: only the existing
split-MoE fixture, selected by exact fixture identity (variant, configuration
and fixture/weight identity), and only when its completed path witnesses are all
F32. Ordinary XS, split dense, all other fixture families, native operator
checks, NLL (1e-5), released-file and long-context bounds remain unchanged.
F32 logits remain at 2e-5 for every other fixture and family. F16, BF16 and int8 keep
their existing frozen budgets. No runtime storage type alone selects this limit.

The retained strict failures remain failures under the original criterion.
Published main with independently widened weights also fails that criterion;
candidate widened and main widened outputs are bit-identical. CPU packed and
widened outputs are bit-identical in the exact control. GPU packed and widened
outputs differ, but captured first-projection inputs and decoded weights agree,
and both products fit independently computed F32 accumulation bounds. These
controls support reference/arithmetic sensitivity rather than an identified XS
decode defect; they do not prove every downstream operation correct. See
[the integration evidence](iq4-xs-integration-20261021.json).

The approved-rule requalification consumed 162 retained calls: CPU, TP2 and
TP4 each ran 50 full-vocabulary comparisons and 4 NLL checks. All three stages
have zero failures under the approved limit. The maximum logit errors were
3.341357421859925e-5 (CPU), 3.6863510131990296e-5 (TP2) and
3.586351013185052e-5 (TP4); maximum NLL errors were 1.7477574463242718e-6,
3.815242553883991e-6 and 4.302242553322344e-6, respectively. The original
strict-bound comparison had 20/30/30 failing logits for CPU/TP2/TP4; those
failures are preserved and are not described as strict passes. All NLL errors
remain within the unchanged 1e-5 limit.

Audit artifacts are under
`build-quant-resume-20261020/iq4-xs-integration/`: the requalification summary
has SHA-256 `1a45e40775514b1c4a1def673579ae146ebc5e5dc36d589ff4526516283fcfba`,
and the root audit has SHA-256
`e3f0598c9ae61bef8725c1f92c75770d2aa8222f16cfaaff69dc155affd95520b`.

The additive rule is an empirical acceptance criterion. HF disagreement is not
a rigorous error bound for a third implementation. Its risk is accepting a small
implementation error inside the larger logit margin; unchanged independent
operator, NLL, released-file and long-context checks remain separate constraints.
No tolerance is chosen from a candidate's maximum error.

A read-only developer review first proposed a uniform F64 HF oracle with the
strict limits retained. On comparison, the reviewer and root recommended this
narrow empirical allowance: replacing the oracle is also a policy amendment,
requires qualified new goldens, and the retained Windows long-case error against
diagnostic F64 is still 2.35609e-5. The F64 output remains diagnostic, not a
replacement golden. The user subsequently approved the narrower allowance above;
the earlier review history does not itself constitute that approval.

An excess over the approved rule still fails. Approval and this focused
requalification do not automatically qualify all IQ4_XS behavior or remove
hosted CI requirements. Exact-head hosted CI and landing remain pending.
