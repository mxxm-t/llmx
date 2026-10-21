# Q3_K BF16 family-limit calibration (approved 2026-10-21)

This approved Q3_K-only BF16 amendment uses an independent reference measurement and the existing maximum-times-two rule. It departs from the normal pre-candidate freeze order: the original generic BF16 limit failed first, and this calibration followed that failure. The user explicitly approved this scoped amendment on 2026-10-21 after reviewing the evidence. The Q3_K BF16 family is installed; affected qualification remains separate and the original generic-limit failure is preserved.

| Error | Frozen generic limit | Independent-reference maximum | Approved limit (2x maximum) |
| --- | ---: | ---: | ---: |
| Full-vocabulary logit absolute error | 0.05409973859786987 | 0.1174459457397461 | 0.2348918914794922 |
| Mean NLL absolute error | 0.007924573802609913 | 0.010312975525200052 | 0.020625951050400104 |

The maxima are from the additional-seed split-MoE fixture, seed 1, not the original failing dense-untied seed. The measurement covers 30 tiny Q3_K BF16 models: all 6 original-seed models were routing-stable; 23 of 24 additional-seed models were stable, with one ordinary MoE seed-2 routing change retained as refused and excluded from maxima. It contains 150 paired full-vocabulary cases and 60 NLL cases. The original and additional partitions both enter calibration maxima; neither is held out.

The independent reference uses independently decoded F32 weights, literal BF16-rounded matrix inputs widened to F32, F32 product sums, F32 routers, and the unchanged HF NLL scorer. It is limited to tiny synthetic ordinary/split dense-untied, dense-tied and MoE fixtures, contexts 4 and 16, and the recorded offline CPU HF environment. It does not establish released-file or real-model quality, F32 acceptance, or server qualification. No F32 or released-file bound, other family limit, or consumer reference is proposed to change.

The retained failing split case remains a failure under the generic limit: its NLL error is 0.009645381058994928 against 0.007924573802609913. A separate control found identical printed aggregate NLL for main-widened, candidate-widened and packed weights; the independent BF16-rounded HF result also differs from original HF beyond the generic limit. That is evidence of reference sensitivity for this case, not per-token identity, an accepted budget, or a gate pass. A prior control setup refusal is retained alongside the successful name-joined control.

**Decision:** the user approved this Q3_K-only BF16 family addition after the original generic-limit failure, using the measured independent-reference maximum-times-two rule above. Only witnessed BF16 execution selects it. Q3_K F16/int8 use the generic budgets and all-F32 calls stay strict. All other frozen values, including the IQ4_XS family and fixture allowance, are unchanged. Candidate acceptance still requires affected qualification; the original failure is not rewritten as a pass.

Evidence is preserved under `build-quant-resume-20261020/q3-k-focused-rig/`:

- `bf16-family-proposal/reference-only-evidence.json` SHA-256 `b0999ab1ee0d5ffd95b25cf84616795ec7e9ebada694c124c83ee5f46afb08c1`; protocol `protocol.json` SHA-256 `1b9c0d3b8bbde5066fa7e009f3cbfc7fdc40a32c52c2cf973c1cc88ff444fcde`.
- `bf16-family-proposal/q3-k-family-proposal.json` SHA-256 `c60db18f8f9302b9577457a3e607cb861435abbb803b95061243e19d20e7fef3`.
- `bf16-family-proposal/selector-serialization-audit.json` SHA-256 `849d2751eaccd36106c54359352236574e703a6cb114d73646de42d79d429440`; it confirms proposal-only serialization, no HF or candidate calls, and unchanged other budget values in memory.
- `q3-bf16-split-failure.json` SHA-256 `2a0aef3d73d222bb9595b84286e22c542033a980c50cd8fcdf0abfbe72a2801a`; `q3-bf16-diagnostic-evidence.json` SHA-256 `0e15d5292695493340693a2bcdabd033266df0ce07f6bf57dbd12babad8f5577`.

The pre-amendment frozen budget input used by the protocol has SHA-256 `1bcd46bd403ccc3e8f67bcece2e768829e5eeaee19cdccda897fed2b1f7d5fbd`. The selector audit reports identical before/after frozen-budget hashes and did not write a product budget file.

## Affected qualification completed 2026-10-21

The frozen alignment runtime passes the approved tiny-fixture checks on one MI50 and on TP2: 150 full-vocabulary comparisons and 12 NLL checks per placement, dense untied/tied and MoE, with completed BF16 path witnesses. The maximum logit/NLL errors are 0.03530302/0.00314192 on one device and 0.07754329/0.00964538 on TP2. Parent replayed every capture through the shared acceptance owner and verified runtime identities and clean container removal. Under the old generic limits, the retained TP2 captures still contain 20 failing logit comparisons and two failing NLL comparisons; the approved amendment does not rewrite that history. Other policy values remain identical. [Qualification and artifact identities](q23-approved-bf16-validation-20261021.json).
