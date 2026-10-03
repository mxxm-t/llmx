# Precision

This page defines the activation precision system: how a run's dtype is chosen, how each device carries it out and how the tests hold each executed path to its own tolerance. Weight storage and KV cache storage are separate choices.
The user approved the policy on 2026-09-30, as agreed at 630644c0, with its seven questions decided as recommended (section 7). Sections 6 through 8 preserve that delivery plan and its gates; [STATUS](STATUS.md) records the implementation's qualification and landing evidence.
The parked prototypes below are research preceding the 2026-09-30 approval. The implementation includes the resolver, executed policies, flag and qualified native F16 preference, with its release gates and retained speed gaps recorded in [STATUS](STATUS.md). Calibration measurements retain their stated historical scope.
The approved delivery plan held dependent quantization merges until the unified system was complete. Vulkan MXFP4 is integrated with this release; subsequent quantization branches are checked against this completed policy and current main, with their own applicable gates. An earlier green run does not approve a later numerical policy.

## The rule (signed 2026-09-30)

The rule was agreed in the collaboration log on 2026-09-30 and follows vLLM's actual dtype behaviour with a small implementation.

1. **auto** (the default) takes a non-F32 declared dtype where every device can implement it, otherwise a common preferred supported dtype, falling back to F32 when no 16-bit choice is common. A declared F32 model takes the preferred choice under auto, as vLLM does; explicit `--dtype f32` stays F32.
   A GGUF file records no source dtype, so its architecture module declares a documented default: BF16 for qwen3, qwen3moe, qwen35 and qwen35moe.
   An explicit dtype overrides it.
   On the MI50, the Radeon VII and an AVX2 CPU, auto is F16; on a backend with qualified native BF16 kernels it is BF16.
2. **`--dtype auto|f16|bf16|f32`**: a valid value is never refused.
   A device without it natively takes a fast exact emulation, else F32, with a visible warning.
   F16 and BF16 never stand in for each other; F32 is the only wider fallback.
3. **Weights are read exactly and ordinary sums are F32.** Existing wider recomputation for unsafe ranges remains until an equally safe implementation replaces it.
   The F32 intermediates llmx keeps today (norms, softmax, rope, recurrent state, residual) stay until a narrower change is measured and passes the gates.
4. **Implementations.** A kernel may implement a dtype in another form, block-scaled 16-bit integers for F16 for instance, only when it passes that dtype's calibrated budget and exact range checks, with a witness that it ran.
5. **One resolution per run.** One owner resolves the dtype once; the requested, effective and native, emulated or fallback dtype of each device is shown once, on the CLI and in `/v1/health`.
6. **Gates.** One tolerance owner keyed by the dtype that ran; a per-dtype budget calibrated and frozen before any candidate is measured; F32 paths keep their bounds; exact conversion and range checks; the real-model gates and within-backend byte identity unchanged; speed at default clocks on one card, first support may leave speed parity open after this plan's release dependencies are satisfied. Gains and losses on existing paths are still assessed together under AGENTS.md.
7. **Owners.** XDEV owns the full dtype implementation, steps 2 through 6: resolver and execution, CPU and GPU kernels, tolerance owner, flag/fallback and docs. LDEV owns the step 1 documentation landing and reviews the development plan and each checkpoint; this replaces the earlier CPU/GPU ownership split by the user's instruction of 2026-09-30.

The ROADMAP entry that states the rule for the gates is `8. Correctness & perf gates`, Precision policy, on the branch `docs/gate-guidelines`.

## 1. The resolver

**Where it lives.** The resolution reads the model's declared dtype, which the model layer holds, and each device's capabilities, which the backends layer reports, so by the layer rule (AGENTS.md, Architecture) it belongs to the model layer, beside `place_model` in `src/model/place.hpp`, the one entry that already holds a run's weights and backends.
It returns a `DtypePlan` with the placed model and applies that policy to the model's execution context before inference. Every dense, grouped, routed and split matmul consumes that policy through the backend's existing dispatch owner. It must not change another loaded model's policy through shared mutable backend state. The inference layer keeps the resulting record, the CLI writes it and the server reports it. A report alone is not an implementation.
Nothing below the model layer chooses a dtype, and no kernel chooses one by weight type.

**What it reads.**
- The declared dtype: a column of the architecture registry (`src/model/arch/registry.hpp`), BF16 for the four Qwen entries, copied into `ModelWeights` when a file is read.
  A future safetensors loader reads the checkpoint's `torch_dtype` instead, as vLLM reads the HF config.
- Each backend's native dtypes, preferred 16-bit first (section 2).
- The request: `--dtype auto|f16|bf16|f32`, defaulting to `auto`.

**How it resolves**, as vLLM does at its pinned commit 1117140e (`vllm/config/model.py`, `_resolve_auto_dtype`, [source](https://github.com/vllm-project/vllm/blob/1117140edbaf3d1e1f73b01b09cde3cf98ec24ff/vllm/config/model.py)): the platform's supported dtypes, filtered by the model's validity, give a preferred dtype, the first; a float32 model is taken down to the preferred dtype; the declared dtype is kept where it is supported, and the preferred one taken otherwise.
llmx makes one plan for every device of a run, since a split's stages pass activations to one another:
1. For a non-F32 declaration, the declared dtype where every device lists a complete supported implementation.
2. Else the first device's preferred dtype that every device lists as supported. The record identifies the actual implementation and any fallback on each device.
3. Else F32.
A model that declares F32 takes step 2's dtype, as vLLM takes float32 models down.
vLLM's list of models that refuse F16 has no counterpart until an architecture needs one.

**The record.** Per run: the requested dtype, the declared dtype, the effective dtype, and per device its name, how it carries the effective dtype (native, emulated or fallback) and its paths, the activation form of each kernel family as groups `class (members)` (section 2).
The CLI writes it once to stderr as the model is loaded; on the MI50 it reads, with the retained F32 operations named beside the integer families:

    dtype: auto -> f16 (model declares bf16); vulkan:0 native: block-int16 (quantized rows and eligible tiles), f32 (other products, routers)

A fallback adds `warning:` and names the dtype it replaced.
`/v1/health` gives the same record as a `dtype` object: `requested`, `declared`, `effective` and `devices`, each with `device`, `how` and `paths`.
Keep the implementation in the placement owner and existing backend dispatch owners. It needs a small plan/record, capabilities and one explicit connection to execution, not a second precision framework. The pre-approval `wip/dtype-auto` prototype only built a report. The released implementation applies the resolved policy to execution and retains completed matrix-path witnesses; STATUS records its landing evidence and remaining measured speed gaps.

## 2. Capabilities and kernel classes

**A backend's capabilities.** `Backend::native_dtypes()` lists the dtypes whose kernels the backend has, preferred 16-bit first.
The auto-preference list advertises a dtype only after the backend implements that policy across its reached kernel families. An explicit dtype served only by emulation or F32 fallback is not thereby added to the native auto-preference list. Report native arithmetic, emulation and wider fallback accurately; F16C conversion alone is not native F16 arithmetic. The supported policies are listed below. Here native means the backend supplies the qualified policy directly; it does not assert literal half-arithmetic instructions on every path. The path record names the arithmetic used:
- CPU (AVX2, F16C, no AVX-512 BF16): F16, F32.
- Vulkan on the MI50 and the Radeon VII: F16 (qualified block-int16 and wider F32 paths), F32.
- A device with BF16 arithmetic lists BF16 first once its BF16 kernels exist.

**A kernel's class.** `Backend::dtype_path(effective)` names, per kernel family, the form its matmul activations take:
- `f32`, `f16`, `bf16`: activations in that type;
- `block-int16`: blocks of 32 scaled to 16-bit integers, which may implement the F16 class once it passes the F16 budget and its exact range checks (rule 4);
- `block-int8`: blocks scaled to 8-bit integers, which implements no class; under auto a path of this form moves to a 16-bit form (section 3), or is withdrawn.
The groups name weight types (`MXFP4`), `quantized` for packed quantized weight formats excluding F32, F16 and BF16 storage, or a phase (`prompts`, `decode`).

**Witnesses.** A static catalog describes possible paths; it does not prove which path a call executed. Tests use the actual dispatched path for the measured phase, shape and operation role, through a counter, kernel name or forced-path unit check. The selected plan alone never chooses a looser bound for a path that stayed F32. Check both sides of every threshold and grouped, routed and split callers. Existing Vulkan per-kernel timing names may provide this evidence without a new profiling system.

## 3. Backend paths before and after dtype integration

Historical activation forms before this policy, from `main` at 4970a071 and the then-private MXFP4 Vulkan integration branch (`integrate/mxfp4-vulkan`):

| Backend | Prompts | Generated rows | Output head | MXFP4 prompts | MXFP4 rows | F32 weights |
|---|---|---|---|---|---|---|
| CPU, AVX2 | f32 except K-quant rows at width >=4096: Q4_K/Q5_K block-int8, Q6_K block-int16 | Q8_0 f32; Q4_0, Q4_1, Q6_K block-int16; Q4_K, Q5_K block-int8 | as generated rows | f32 | block-int16 | f32 |
| Vulkan, MI50 (integer dot preferred) | block-int16 (integer tile) | Q8_0 block-int16; other quantized types block-int8 | Q4_0, Q4_1, Q6_K block-int16 | f32 (predecoded F32 tile, wide range recompute) | block-int16 | f32 |
| Vulkan, Radeon VII | f32 (float tile) | block-int16 | block-int16 | f32 | block-int16 | f32 |

The CPU prompt cell above describes dense products. Routed expert prompts use block-int8 for Q4_K/Q5_K and block-int16 for Q4_0/Q4_1/Q6_K at every valid supported width, without the dense 4096 cutoff; Q8_0, MXFP4 and F32 routed prompts retain F32 activations.

Under auto every Qwen model resolves to F16 on these three, and each path becomes:
- **CPU:** Q4_K/Q5_K generated rows, wide dense prompts and routed products use block-int16 activations, alongside the other qualified 16-bit integer paths. Existing F32 prompt paths remain documented wider implementations. AVX2/F16C does not provide a literal F16 dot kernel; the completed path record distinguishes block-int16 and F32 products.
- **MI50:** eligible prompts use the block-int16 tile, and quantized rows use block-int16. Eligible dense MXFP4 prompts use exact weight copies in the bounded scratch buffer; projections beyond its budget and routed MXFP4 prompts use the wider F32 tile. Unsafe ranges retain wider recomputation. The device profile, rather than the dtype resolver, chooses the eligible kernel.
- **Radeon VII:** the supported F16 policy uses block-int16 rows and the existing wider F32 prompt tile. A block-int16 prompt tile remains a possible measured optimization, not a separate precision policy.
- **BF16 hardware:** a future backend advertises native BF16 only after its kernels conform. Current CPU and eligible Vulkan backends emulate an explicit BF16 request by rounding inputs and widening for F32 arithmetic; auto selects F16.
- **Operation precision** is not inferred from weight storage. Explicitly retained F32 operations, such as the existing norms and router, stay F32; other matmuls follow the selected policy or report their actual wider fallback. A matrix stored as F32 is not by itself an activation-policy exception.

## 4. The test side

**One tolerance owner.** `tests/common.py` holds every tiny-fixture bound, keyed by the policy actually executed by the checked operations (`hf_bounds`). A static list of possible paths or the weight type does not witness execution. F32-only calls retain F32 bounds even if a lower-precision kernel could run at another shape. Mixed calls record the paths actually reached and the applicable calibrated policy.
F32 keeps its bounds, 2e-5 on logits and 1e-5 on NLL.
The per-type constant it replaces, MXFP4's 2e-4, goes; file-exact goldens keep only HF's outputs and their metadata, never a tolerance.

**Calibration.** `tools/calibrate_dtype.py` measures each 16-bit class against HF in float32 in the pinned HF environment: HF with the weights exact, the input of every matrix product but the routers' rounded to the dtype, and F32 sums, over the tiny fixtures (the dense F32 model tied and untied, the MoE model, the three MXFP4 fixtures) at their own weights and four further seeds.
A class's budget is twice the largest error, frozen in `tests/data/dtype_budget.json` before any candidate path is measured against it, and changed only with the user's approval.
The parked branch's first measurement: F16 at most 0.00165 logits and 0.000148 NLL, budgets 0.0033 and 0.00030; BF16 at most 0.0270 and 0.0040, budgets 0.054 and 0.0079.
Retain the calibration script, exact weights, seeds, environment and raw results so the budget can be reproduced independently of a candidate. The reference calculation must match the mixed-precision operation choices being validated.

**Kernel checks.** Against an independent reference fed the same rounded inputs, allow only the stated accumulation error. A separate comparison against the original inputs may additionally allow the sum of each weight's magnitude times its input-rounding error; do not count that error twice. Conversion and range checks remain exact: every declared finite scale, zero and sign, extreme exponents, and the fallback where an intermediate leaves the safe range while the correct result is representable.

**Unchanged gates.** The real-model gates keep their bounds: file-exact HF NLL and rankings, near-tie rules, raw and chat-template 16k sequences against file-exact HF, and chat-00. CPU and GPU numerical changes both run the depth checks. `tools/long_context_check.py` self/CPU consistency is additional evidence and cannot replace the independent HF comparison. Preserve within-backend byte identity, batch invariance and split identity.
A path whose numerics change reports its HF error and headroom; a path that does not stays byte-identical.

### Independent HF depth acceptance (agreed 2026-10-01)

BOSS delegated the depth decision to LDEV and XDEV; LDEV confirmed their joint agreement in the TUI at 13:56. This changes only the top-one rule of the independent raw and chat-template 16k checks. The named owner is `tests/common.py`'s `HF_DEPTH_TOP1_MARGIN`, 0.1, through `hf_depth_top1`. A row is exact when llmx chooses HF's first token, an accepted tie only when it chooses HF's second token and HF's top-two gap is at most that margin, and a miss otherwise. Report separate exact, accepted-tie and miss counts, preserving every position, both reference IDs and its gap. This is not a new maximum-logit bound.

Predeclare one reference policy per workload from the arithmetic it reaches, including its retained F32 operations. Its weights are the file's exact decoded weights; F16/BF16 matrix inputs round literally to that type and sums remain F32. Do not choose a reference per position or after comparing which one agrees. Keep the original-F32 HF quality comparison beside the selected-policy comparison. Short-model rankings, NLL, calibration and range bounds remain unchanged. Freeze this rule before reevaluating retained captures; it cannot accept the already recorded Radeon BF16 raw mismatch whose HF gap is 0.69439793.

## 5. The dtype flag

`--dtype auto|f16|bf16|f32` on every model command and `serve`, `auto` by default. Synthetic `bench` without a model refuses the flag.
- A misspelt value is a usage error (status 2); a valid value is never refused.
- Where a device lacks the value natively, it takes a fast exact emulation where one exists (BF16 or F16 activations rounded to that type and widened to F32 in the kernel), else F32, and the CLI writes one warning line and the record marks the device `fallback` or `emulated`.
- `f32` means F32 activations on every path. Vulkan quantized weights use the F32-activation row kernel for narrow batches and the float tile for wider batches, bypassing the integer activation twins.

## 6. Approved delivery order (2026-09-30)

Each step lands on its own branch off `main`, with its gate:

1. **Guidelines.** AGENTS.md's gates and tests section and ROADMAP's rule (branch `docs/gate-guidelines`) and this plan. Gate: the suite's `docs` and `dead-code`, both developers' review, the user's approval.
2. **Resolver, application contract and record.** The registry default, capabilities, resolver, connection to the model's execution context, CLI line and `/v1/health` field. Test resolution and application separately. Intermediate code may describe existing paths, including block-int8, but cannot claim an effective F16 policy until the conformance steps implement it. Keep incomplete policy behavior private and integrate the dependent commits before release. Gate: CTest with a resolver test (declared, preferred, mixed devices, F32 fallback), the CPU suite, Qwen3-0.6B byte identity against main, the hosted run.
3. **Tolerance owner and budgets.** `hf_bounds`, the record parsers, `tools/calibrate_dtype.py` and the frozen budget, every tiny-fixture check witnessing its class. Gate: the suite on the CPU and a device, `reference-consumer` covering the owner, the budget file reviewed before any candidate uses it.
4. **F16 conformance, GPU.** MI50 rows from block-int8 to block-int16 and the Radeon VII unchanged; the MXFP4 Vulkan integration lands with its prompts on the integer tile (`wip/mxfp4-int16tile-v2`: its boundary regressions for scale bytes 0, 1, 103, 104, 143 to 146, 252 and 255, `mxfp4-vulkan` against a reference on the rounded inputs with a witness, and the tile cost of its range guard). Gate: the device tier on both cards, HF error and headroom, speed against main and the reference at default clocks.
5. **F16 conformance, CPU** (XDEV). Q4_K/Q5_K prompt and generated rows from block-int8 to a qualifying implementation, plus the CPU's capabilities, execution-policy application and actual-path witnesses. Gate: the model/kernel tier, independent HF depth checks, batch/split invariance and matched performance against main and mx-llama.cpp.
6. **The flag.** `--dtype`, emulation and F32 fallback on each backend, warnings. Gate: the `cli` component, the suite at each value on the CPU and a device.

What waits:
- Existing CPU MXFP4 support remains on main; its prior evidence is preserved.
- The pending CPU AVX2 decoder optimization, Vulkan MXFP4 integration, int16 tile and new quantization types remain held until the approved unified system is complete, including CPU/GPU conformance and explicit dtype/fallback behavior.
- Steps may be developed in separate dependent branches. Their landing order must leave main truthful about the arithmetic it runs. Any earlier staged quant merge requires an explicit user-approved amendment to this hold.
- Re-run each pending branch's applicable gate against the completed policy and current main; an earlier green run does not approve a later numerical policy.

## 7. Decisions (approved 2026-09-30)

The user took each question as recommended.

1. **MI50 rows at 16 bits.** Moving the MI50's block-int8 rows to block-int16 is what F16 requires; the 16-bit prompt fix cost decode little on these types, but it is to be measured per type. Decision: move them in step 4 and record any decode cost as the fix's.
2. **`--dtype f32` on Vulkan.** Quantized rows have no F32-activation kernel. Decision: write them in step 6, since a fallback that cannot run F32 would break rule 2.
3. **BF16 emulation.** A device without BF16 emulates it by rounding activations to BF16 and computing in F32 on the float tile. Decision: accept that as exact emulation, with its speed recorded.
4. **Existing F32 CPU prompt paths.** Decision: keep a documented F32 implementation where it passes and is faster. Wide K-quant prompts are a separate existing integer path and must also conform; no speed claim for a proposed conversion is made before measurement.
5. **The budget's margin.** Twice the largest error over five weight sets per fixture. Decision: keep twice, and recalibrate only with the user's approval, never to admit a failing path.
6. **The record on every run.** One stderr line for every model command. Decision: always, since the rule makes precision visible and the tests read it.
7. **GGUF dtype metadata.** A GGUF file carries a quantization type but no source dtype. Decision: the architecture's documented default, overridable by the flag; a safetensors path reads `torch_dtype`.

## 8. Development plan (2026-09-30, confirmed by LDEV)

The user assigns implementation to XDEV and asks for LDEV confirmation of this development plan before coding. LDEV explicitly confirmed section 8 at 9fda628d in the collaboration log at 09:39 on 2026-09-30 and instructed XDEV to start. The policy and seven decisions above are already user-approved; this section makes the implementation order, ownership and evidence explicit. No second user approval is requested for unchanged policy.

### Baseline and handoff

- Start from main after LDEV lands step 1. Until then, a private dependent worktree may use the agreed docs branch; no feature is merged around the dependency.
- Treat wip/dtype-auto at 521393c1 as reference material, not a passing implementation. Salvage reviewed pieces individually: fix the invalid test string, duplicated include, F32-auto rule, premature capability claims and static-catalog witness. Never cherry-pick its report-only behavior and call the policy implemented.
- Preserve the calibration rows and provenance. The old run used torch 2.5.1+cpu and transformers 4.55.2, one thread and the recorded seeds. Verify that its reference operation choices match the selected mixed-precision policy before applying its frozen budgets; do not increase them to pass a candidate.
- Keep wip/mxfp4-int16tile-v2 at 5afd95e and the pending CPU AVX2 decoder separate. Their old checks remain evidence, not approval under the new policy. Pending quant branches land only after the complete system and their renewed gates pass.
- XDEV uses its own worktrees and containers. LDEV's existing work/evidence is read-only unless explicitly handed over; coordinate GPU ownership before any run, retaining default clocks.

### A. Make the policy reach execution (step 2)

The placement owner resolves the request once and stores an immutable policy with the model. Carry the selected policy through the existing execution context and matrix-operation call boundary; do not use a process-global or mutable backend-wide dtype setting that changes another model. Dense, grouped, routed, additive and split calls reach the same backend dispatch owner. The CLI/server only parse and report.

Reuse the small dtype enum and registry default where correct. Keep selected policy separate from actual per-device implementation and fallback. Explicit F32 must reach F32 activation arithmetic. Auto follows the pinned vLLM rule, including the declared-F32 downcast; explicit overrides do not follow auto. Unsupported native precision selects the promised exact emulation or visible F32 fallback, never another 16-bit format silently.

Test resolution and application separately. Hold exact expected choices on mixed devices, and use inputs whose F32, F16, BF16 and block-integer results differ to prove which arithmetic a call ran. Alternate two models with different policies over a shared backend and verify neither changes the other's results. Exercise model reuse, split stages and mixed prompt/decode passes. Keep incomplete paths private; no public F16 capability before conformance.

### B. Make correctness bounds follow execution (step 3)

Implement the one tolerance owner in tests/common.py. Static capability strings cannot select a looser bound. Use actual dispatched-path evidence for the tested operation, phase and shape; F32-only calls keep their old F32 bounds. Preserve independent HF goldens unchanged.

Reuse existing Vulkan kernel names and targeted CPU dispatch checks where sufficient; do not build a new tracing framework or user tuning switch. Test missing or mismatched witnesses, F32-only calls, threshold crossings, and mixed calls. Against the same rounded inputs, check only accumulation error; against original inputs, account separately for input rounding. Retain exact scale, conversion, zero/sign and unsafe-range checks.

### C. Implement conforming CPU paths (step 5, can overlap GPU validation)

Replace the remaining Q4_K/Q5_K block-int8 activation dots in their existing owner with 16-bit arithmetic, covering dense decode, dense prompts at and above width 4096, grouped projections and routed prompt/decode at all supported widths. Reuse the existing 16-bit activation preparation and integer-dot helpers; remove the 8-bit storage/quantizer/helpers when no production caller reaches them. Avoid a second quantization implementation.

The first focused regression uses a unit weight to select a small activation beside a peak, where 8-bit rounding loses the value. It must fail on the old arithmetic and pass on the new path. Cover valid K widths 3840, 4096 and 4352, both signs, mixed row runs, tails and thread counts. Existing Q6_K, Q4_0/Q4_1 and MXFP4 16-bit paths also need conformance evidence; their representation alone is not proof. Keep existing F32 paths where the approved policy allows a reported wider implementation.

Start with a short matched performance screen after the focused numerical check; a candidate with an unresolved performance cost gets diagnosis before expensive full HF runs. Report phase gains and losses together, against both main and mx, with a same-owner layout control and all monitored samples retained.

### D. Implement conforming GPU paths (step 4)

Move the remaining MI50 quantized row paths from block-int8 to the conforming 16-bit path, preserving batching and accumulation order where possible. Verify the Radeon VII's existing behavior independently. Do not infer one device's result from another device or driver.

Complete the MXFP4 tile's exact float-scale copy and unsafe-range fallback from the parked prototype. Check scale bytes 0, 1, 103, 104, 143, 144, 145, 146, 252 and 255; prove the intended tile ran and test both sides of dispatch thresholds. Measure copy, tile and fallback costs separately before the matched end-to-end screen. Preserve overflow/underflow repair when the final result remains representable.

### E. Complete overrides, emulation and reporting (step 6)

Wire --dtype auto|f16|bf16|f32 once into shared execution options for every model command and serve. Update help, USAGE and health in the same change. Valid requests always have a working implementation: native where present, otherwise exact emulation when available, otherwise visible F32 fallback. Invalid spellings remain usage errors.

Implement the missing Vulkan F32-activation row path and BF16 round/widen execution needed for those requests, with the same centralized dispatch. CPU and GPU conversions are checked against independent exact rounding cases. Reports name requested/effective precision and actual implementation; no native claim follows merely from conversion instructions. Test every value on CPU, MI50 and Radeon, including follow-up chat, server requests and split execution.

### Checkpoints and release

A through E are separate commits or dependent branches, with declared dependencies and one owner; do not duplicate the resolver or tolerance owner. CPU and GPU focused work may proceed independently once the application contract is fixed. Host and rig correctness checks may run in parallel; performance arms run without competing work on their assigned resources and with recorded background activity.

During development run only the focused builds and tests the edit needs. Before release, run the applicable model/kernel gates: native tests, complete required-fixture suites, independent file-exact HF NLL/rankings, raw and chat-template 16k checks and chat-00, within-backend batch/split identity, both GPU environments, and matched main/candidate/mx performance at default clocks. Test all dtype values and fallbacks. Preserve every attempt and artifact, review all Markdown, and run hosted CI at the exact integrated head.

LDEV reviews the concrete checkpoints, but the user has already authorized merging completed work that passes its required gates; review is not an added general merge-permission flow. The quant hold remains until this complete approved system is done. Then revalidate and land the held quant branches individually. Every status report distinguishes implemented, validated, held and merged work, and provides a test executable path after a Windows build.

### Calibration checkpoint (2026-09-30)

The calibration tool and fixture builders reproduce the frozen budget. Reproduce the frozen budget in torch 2.5.1+cpu and transformers 4.55.2:

```
python tools/calibrate_dtype.py --output dtype-reproduction.json
```

The output must be a new file. The tool pins the package versions, uses one CPU thread and records fixture seeds and weight hashes. It reuses the HF fixture builders rather than defining another model. The committed `tests/data/dtype_budget.json` preserves the handoff values byte for byte; a reproduction matched every field of all 60 original rows and all maxima and budgets exactly. An independent module-input audit over the six default fixtures in each dtype checked actual rounded inputs, unchanged F32 routers and unchanged weights. These were calibration checks, not llmx kernel-conformance evidence. At that checkpoint the existing tiny-fixture tolerances remained unchanged while actual dispatch witnesses and the shared tolerance owner were being connected. The implementation reads completed matrix-path witnesses through the shared tolerance owner; all-F32 measurements retain F32 bounds, and narrowed paths use the frozen budget. Backend conformance remains a separate qualification.

## 9. Precision and range (clarified 2026-09-30)

The user questioned whether the range work was overengineered. LDEV confirmed at 13:50 that the intended contract uses the selected dtype's precision and range, not the original F32 input range. This clarification narrows the range requirement in rules 3 and 4 above; it does not waive a gate or widen the frozen budget.

The earlier private implementation at 4fc70316 treated F16 as an accuracy class served by block-scaled int16 activations, while keeping the original F32 activation range. Its regression asked an input of 2^-112 to survive a matrix product. An actual F16 conversion makes that input zero: binary16's smallest positive subnormal is 2^-24. Range repair for that original input is therefore stronger than the conversion behavior of ordinary F16. It remains meaningful for a path promising the original F32 inputs; it is not by itself a requirement of literal F16 activation arithmetic.

Source checks distinguish the alternatives:

- [vLLM dtype](https://docs.vllm.ai/en/stable/configuration/engine_args/#--dtype) selects actual model weight and activation types. Quantized methods impose further restrictions; the current [MXFP4 configuration](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/mxfp4.py) lists BF16 activation support. This is not evidence that every vLLM kernel uses the same arithmetic.
- [llama.cpp CUDA/HIP quantized dots](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vecdotq.cuh) include Q8_1 activation paths. The local reference at 407d0bb1f1 uses integer dots for MXFP4 and centers Q6_K weights before its integer dot. Those paths are different from a literal FP16 activation contract.
- [llama.cpp Vulkan dots](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/dot_product_funcs.glsl) include native F16 operand dots with F32 accumulation and widened FMA alternatives. Availability and end-to-end speed on our driver still need measurement; source presence is not a performance result.

Confirmed simplification: define explicit F16/BF16 operation inputs by their actual rounding behavior, with F32 accumulation and the existing F32 operation exceptions. Use native arithmetic where the device supports the required behavior, otherwise round and widen in the existing backend owner. Retain an optimized integer implementation only when its independently measured errors meet that operation contract and its added complexity pays for itself. Do not make every F16 kernel preserve values that F16 conversion already removes.

The frozen calibration uses exact weights and rounds matrix inputs only. Native half dots that also round dequantized weights do not automatically match it. Weight conversion, accepted range, reporting and the error oracle must be settled together; no implicit weight downcast or budget widening is authorized by this proposal. F32 requests keep their F32 behavior. Format decoding, finite weight-scale correctness, zero/sign handling, real-model HF bounds, long-context checks and performance comparisons remain required.

Round-to-nearest conversion includes subnormals and rounding to zero at underflow and infinity at overflow; it is not finite saturation. Exact tests follow those conversion semantics. A qualifying implementation may preserve more precision or range, but no guard is required solely to preserve an input beyond the chosen dtype's range.

The backend repeatability tool, `tools/long_context_check.py`, accepts `--dtype auto|f16|bf16|f32` and forwards it to prompt sizing, fresh generation and scoring on both backends. Its existing backend comparison margin is unchanged; that check is separate from file-exact HF depth acceptance.

The earlier range implementation at 4fc70316 remains in history with its evidence. BF16 rounds activations and widens for F32 arithmetic; F16 uses qualified block-int16 paths or documented wider F32 paths. F16 checks cover the selected precision's range, and explicit F32 retains its original-input range checks. CPU and Vulkan advertise F16 before F32 in their native preference lists. The current evidence, remaining performance work and release state are recorded in [STATUS](STATUS.md).
