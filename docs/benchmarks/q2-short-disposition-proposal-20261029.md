# Q2 retained short-case disposition decision

**User approved, 2026-10-29.** The eight exact file/platform/cell exceptions
below are approved. Original failed comparisons remain failed and explicitly
labelled exceptions. No runtime, golden, reference selection, numerical limit
or test implementation changes in this record.
Complete Q2 quality disposition and merge approval remain open. The later
request for a separate quality fixture does not expand these historical
exact-file exceptions.
This updates the [earlier proposal](q2-short-exception-proposal-20261028.json)
with the completed exact-method short, depth and code6 evidence; that earlier
proposal and every original failure remain retained.

The approved decision permits the eight exact canonical failed cells
below to remain reported as named exceptions while Q2 qualification continues.
Every other existing applicable requirement remains required. These failed
comparisons retain their values and exception labels; they must never be
reported as passing their original criteria. This is neither merge approval
nor a claim that every end-to-end result is numerically equivalent to HF.

## Exact scope and remaining failures

The scope is only `Qwen3-0.6B-Q2_0.gguf`, SHA256
`8e15e65d208fcc4709ec949fd11dc4a8584b4569f53539e6c39cc4d00ecf6a8b`,
its retained MI50/RADV default block-int16 body/Q6_K head arithmetic, and the
exact token IDs, KV types, schedules, original references and output hashes
in the [pinned ledger](q2-code6-disposition-evidence-20261029.json).
The F32 KV rows below still use the default matrix policy; explicit F32
matrix execution is a separate control.

The columns distinguish the original canonical gate from the newly completed
exact selected-policy diagnostic. The latter does not replace the canonical
reference. Ranking overlap uses the existing shared ranking function.

| Canonical cell | Original F32 comparison, Q2 | Exact selected-policy diagnostic, Q2 | Exact-value main/Q4 control |
| --- | --- | --- | --- |
| Capital, F16 KV | Top1 agrees; top5 4/5, fail | Top1 agrees; top5 5/5 | Original and selected top5 4/5 |
| Code `def add(a, b):`, F16 KV | Top1 40604 vs HF 54200; top5 5/5, fail | Same failed top1; HF top-two gap 0.351575851 | Original and selected top1/top5 pass |
| Colors, F16 KV | Top1 102644 vs HF 102563; top5 3/5, fail | Top1 agrees; top5 4/5, fail | Original top1 also differs, top5 3/5; selected top1 differs, top5 4/5 |
| Colors, F32 KV | Top1 102644 vs HF 102563; top5 5/5, fail | Top1 agrees; top5 5/5 | Original passes; selected top1 differs |
| Rome17, F16 KV | Top1 agrees; top5 3/5, fail | Top1 agrees; top5 5/5 | Original top5 4/5; selected top5 5/5 |
| Boulter continuous NLL, F32 KV, batched | Delta -0.019794392, fail | Delta -0.003996583, diagnostic pass | Identical retained CLI stdout/mean across main/Q4, candidate/Q4 and Q2; per-target F32-KV arrays were not captured |
| Boulter continuous NLL, F16 KV, singleton | Delta +0.030597398, fail | Delta +0.031272777, diagnostic fail | Also fails: +0.021108794 original and +0.021784172 selected; Q2-minus-Q4 is +0.009488604 |
| Boulter continuous NLL, F32 KV, singleton | Delta -0.012433873, fail | Not measured; no new HF variant selected | Also fails original: -0.013269968; Q2-minus-Q4 is +0.000836095 |

NLL uses the fixed 247-token excerpt and 246 scored targets; the existing
continuous limit remains 0.01. The F16-KV batched NLL is **not** an exception:
it passes the original reference at +0.008521461 and the exact selected
diagnostic at -0.002550249. All 246 F64 loss words in this batched control are
byte-identical across main/Q4, candidate/Q4 and Q2. This shared existing-main
result must not be conflated with their differing singleton outputs.

The exact selected HF batch/singleton NLL difference is 0.011747089 without
runtime involvement. The older approximate-helper control separately gives
0.0237900000688. Older controls used one HF thread and division-based input
reconstruction; the exact producer uses four threads and the fixed shader
arithmetic. Their differences cannot be attributed solely to the helper.
Neither spread becomes an extra error allowance or selects another oracle.

The 13 previously missing existing NLL cells are now complete. All 12 windows
pass, covering context 64 with four or two chunks and context 123 with two
chunks, each batched/per-token and with both KV types. Maximum absolute
window delta is 0.012708657 against 0.02. F32-KV singleton gives Q2 mean
17.551664531 and main/Q4 17.550828436, both outside 0.01 from original HF
17.564098404. These are CLI means, not per-target identity claims. Across both
KV types, all 16 default NLL cells are measured: 13 pass and three continuous
cells fail. Explicit F32 matrices with F32 KV separately pass all eight NLL
checks. No additional HF generation, schedule, limit or exception was used
to finish this existing coverage.

## Evidence supporting this disposition

The already-required selected-policy 16K depth checks are complete with the
frozen exact input rule, F32 sums/nonmatrix operations, and F16 KV. Raw gives
504 exact / 8 accepted ties / 0 misses; chat gives 498 / 14 / 0 under the
unchanged 0.1 top-two rule. Both original-F32 quality companions are complete
and descriptive. Runtime mean NLL differences from original F32 are
+0.009785205 raw and -0.002189568 chat; these are not additional short gates.
Explicit F32 matrix arithmetic with F32 KV also passes the existing canonical
short ranking and NLL checks. [Depth and quality pins](q1-controls-q2-quality-20261029.json),
[explicit-F32 control](q1-chat-q2-f32-control-20261028.json).

The bounded six-token code observation preserves each arm's retained CLI
top-ten values and gives byte-identical baseline/observer full-vocabulary
outputs. It observes all 28 layers: 113 dispatches and 197 projections per arm.
All 75,328 captured activation blocks match independent rational F32
quantization for codes, scales and scaled sums. All 788 GPU weight-edge
samples match independent expected bytes; interior GPU weights were not
fully read back. Every measured matrix result is close to a double dot fed
its own actual reconstructed inputs and independently decoded GGUF weights.
Largest absolute errors are 0.000034351 Q2 and 0.000146874 Q4. Existing row
and K23 expressions are reported as diagnostics, without extending their
test scope or defining a new gate.

The first raw Q difference has byte-identical inputs and packed twins.
Actual SPIR-V, NIR, ACO and final ISA establish each path's arithmetic and
reduction order. Exact software replay at token 3, row 818 reproduces Q2 word
`c08c25c7` and Q4 word `c08c25cc`. Q2 is closer to the same-input independent
double value there, and has smaller first-layer Q/K/V RMS errors. Q4 matching
HF top1 therefore does not establish a Q2 matrix defect. Small valid
differences grow across the layers to residual difference 9.946579 and logit
difference 0.322684. Raw QKV is observed before norm/RoPE/KV; attention-output
and FFN-down observations include their residual additions.
[Observation, replay, source and failure pins](q2-code6-disposition-evidence-20261029.json).

No specific further defect in the new Q2 path has affirmative evidence from
these results. Nonmatrix intermediates were bracketed rather than exhaustively
replayed, so complete mathematical correctness is not proved. Their unchanged
shared implementation alone is not evidence for another Q2 defect or grounds
for an automatic new instrumentation gate. No further Q2 run is proposed
without a concrete contradictory finding.

The file itself has poor quality on this fixed excerpt even in independent
original-F32 HF (PPL about 42 million). That alone does not prove mathematical
ill-conditioning; the separate arithmetic experiments demonstrate sensitivity.
The published payload and 64-value type-42 layout are pinned and agree with
the retained mx decoder source. Matched mx benchmarks read the same file,
but do not supply an independent mx PPL oracle. Publisher training/quantizer
procedure provenance remains unknown. The trained Bonsai payload mentioned
in Q1-Q2 is a separate file refused for its actual unsupported YaRN metadata;
replacing the current fixture or stripping that metadata would not clear this
ledger. Poor model quality is context for the decision, not a gate waiver.

## Approved conditions

- Approval applies only to the eight named original-reference cells and their
  pinned retained outputs. The selected-policy code/colors/singleton failures
  remain explicitly reported diagnostics. No unlisted failure is covered.
- Original references, the shared ranking function, the 0.1 depth tie rule,
  continuous/window NLL limits and all tiny-family/operator/range budgets stay
  unchanged. No F64 replacement reference or general margin increase is authorized.
- All other existing applicable format, conversion, tiny-HF, operator/range,
  short/windowed, depth, batch/split identity and matched-performance checks
  remain required. Unmeasured cells do not become passes or exceptions. This
  adds no new nonmatrix or full-model diagnostic requirement.
- The final candidate must match the approved retained CLI output artifacts
  in the ledger, at their recorded precision, before inheriting any exception.
  Changed outputs require renewed review before this exception applies. This
  does not invent full-vector identity where only CLI values were retained.
- No automatic extension to another GGUF, backend, arithmetic policy, output
  set or changed source. Q1 and other precision policies inherit nothing.
- Retain the eight failures as explicit accepted exceptions when assessing
  qualification. Every other existing requirement remains required; this
  decision does not itself merge the feature.

This is the user's exact-file amendment to the real-model requirements in
[PRECISION, Test side](../PRECISION.md#4-the-test-side), approved on 2026-10-29.
It accepts a finite known product difference; it does not prove that the
failed comparisons passed. Generic dtype-family budget approval is separate
and does not grant additional released-file exceptions.
