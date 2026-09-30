# Precision (planned)

This is the plan for one precision system across every backend and weight type: how a run's activation dtype is chosen, how each device carries it out, how the tests hold each path to its own tolerance, and the order the work lands in.
This design is not approved or merged. Parked prototypes and calibration measurements are prior research, not completed plan steps.
Implementation and further dtype builds wait for the user's approval. Quantization merges remain held until this approved system is complete; any earlier staged release needs an explicit user decision.

## The rule (signed 2026-09-30, planned)

The rule was agreed in the collaboration log on 2026-09-30 and follows vLLM's actual dtype behaviour with a small implementation.

1. **auto** (the default) takes a non-F32 declared dtype where every device can implement it, otherwise a common preferred supported dtype, falling back to F32 when no 16-bit choice is common. A declared F32 model takes the preferred choice under auto, as vLLM does; explicit `--dtype f32` stays F32.
   A GGUF file records no source dtype, so its architecture module declares a documented default: BF16 for qwen3, qwen3moe and qwen35.
   An explicit dtype overrides it.
   On the MI50, the Radeon VII and an AVX2 CPU, auto is F16; on a device that runs BF16 natively it is BF16.
2. **`--dtype auto|f16|bf16|f32`**, after auto lands: a valid value is never refused.
   A device without it natively takes a fast exact emulation, else F32, with a visible warning.
   F16 and BF16 never stand in for each other; F32 is the only wider fallback.
3. **Weights are read exactly and ordinary sums are F32.** Existing wider recomputation for unsafe ranges remains until an equally safe implementation replaces it.
   The F32 intermediates llmx keeps today (norms, softmax, rope, recurrent state, residual) stay until a narrower change is measured and passes the gates.
4. **Implementations.** A kernel may implement a dtype in another form, block-scaled 16-bit integers for F16 for instance, only when it passes that dtype's calibrated budget and exact range checks, with a witness that it ran.
5. **One resolution per run.** One owner resolves the dtype once; the requested, effective and native, emulated or fallback dtype of each device is shown once, on the CLI and in `/v1/health`.
6. **Gates.** One tolerance owner keyed by the dtype that ran; a per-dtype budget calibrated and frozen before any candidate is measured; F32 paths keep their bounds; exact conversion and range checks; the real-model gates and within-backend byte identity unchanged; speed at default clocks on one card, first support may leave speed parity open after this plan's release dependencies are satisfied. Gains and losses on existing paths are still assessed together under AGENTS.md.
7. **Owners.** The resolver, the GPU side, the tolerance owner and the docs are one developer's; the CPU side is the other's.

The ROADMAP entry that states the rule for the gates is `8. Correctness & perf gates`, Precision policy, on the branch `docs/gate-guidelines`.

## 1. The resolver (planned)

**Where it lives.** The resolution reads the model's declared dtype, which the model layer holds, and each device's capabilities, which the backends layer reports, so by the layer rule (AGENTS.md, Architecture) it belongs to the model layer, beside `place_model` in `src/model/place.hpp`, the one entry that already holds a run's weights and backends.
It returns a `DtypePlan` with the placed model and applies that policy to the model's execution context before inference. Every dense, grouped, routed and split matmul consumes that policy through the backend's existing dispatch owner. It must not change another loaded model's policy through shared mutable backend state. The inference layer keeps the resulting record, the CLI writes it and the server reports it. A report alone is not an implementation.
Nothing below the model layer chooses a dtype, and no kernel chooses one by weight type.

**What it reads.**
- The declared dtype: a column of the architecture registry (`src/model/arch/registry.hpp`), BF16 for the three Qwen entries, copied into `ModelWeights` when a file is read.
  A future safetensors loader reads the checkpoint's `torch_dtype` instead, as vLLM reads the HF config.
- Each backend's native dtypes, preferred 16-bit first (section 2).
- The request: `auto` until the flag lands.

**How it resolves**, as vLLM does at its pinned commit 1117140e (`vllm/config/model.py`, `_resolve_auto_dtype`, [source](https://github.com/vllm-project/vllm/blob/1117140edbaf3d1e1f73b01b09cde3cf98ec24ff/vllm/config/model.py)): the platform's supported dtypes, filtered by the model's validity, give a preferred dtype, the first; a float32 model is taken down to the preferred dtype; the declared dtype is kept where it is supported, and the preferred one taken otherwise.
llmx makes one plan for every device of a run, since a split's stages pass activations to one another:
1. For a non-F32 declaration, the declared dtype where every device lists a complete supported implementation.
2. Else the first device's preferred dtype that every device lists as supported. The record identifies the actual implementation and any fallback on each device.
3. Else F32.
A model that declares F32 takes step 2's dtype, as vLLM takes float32 models down.
vLLM's list of models that refuse F16 has no counterpart until an architecture needs one.

**The record.** Per run: the requested dtype, the declared dtype, the effective dtype, and per device its name, how it carries the effective dtype (native, emulated or fallback) and its paths, the activation form of each kernel family as groups `class (members)` (section 2).
The CLI writes it once to stderr as the model is loaded; on the MI50 after step 4 of the order below it would read, with the router's retained F32 operation named as its own path:

    dtype: auto -> f16 (model declares bf16); vulkan:0 native: block-int16 (quantized), f32 (router)

A fallback adds `warning:` and names the dtype it replaced.
`/v1/health` gives the same record as a `dtype` object: `requested`, `declared`, `effective` and `devices`, each with `device`, `how` and `paths`.
Keep the implementation in the placement owner and existing backend dispatch owners. It needs a small plan/record, capabilities and one explicit connection to execution, not a second precision framework. The parked `wip/dtype-auto` resolver only builds a report; it does not yet apply the policy or witness which kernels ran.

## 2. Capabilities and kernel classes (planned)

**A backend's capabilities.** `Backend::native_dtypes()` lists the dtypes whose kernels the backend has, preferred 16-bit first.
The auto-preference list advertises a dtype only after the backend implements that policy across its reached kernel families. An explicit dtype served only by emulation or F32 fallback is not thereby added to the native auto-preference list. Report native arithmetic, emulation and wider fallback accurately; F16C conversion alone is not native F16 arithmetic. The following are target capabilities after conformance, not claims about current kernels:
- CPU (AVX2, F16C, no AVX-512 BF16): F16, F32.
- Vulkan on the MI50 and the Radeon VII: F16 (half arithmetic, and the block-int16 twins below), F32.
- A device with BF16 arithmetic lists BF16 first once its BF16 kernels exist.

**A kernel's class.** `Backend::dtype_path(effective)` names, per kernel family, the form its matmul activations take:
- `f32`, `f16`, `bf16`: activations in that type;
- `block-int16`: blocks of 32 scaled to 16-bit integers, which may implement the F16 class once it passes the F16 budget and its exact range checks (rule 4);
- `block-int8`: blocks scaled to 8-bit integers, which implements no class; under auto a path of this form moves to a 16-bit form (section 3), or is withdrawn.
The groups name weight types (`MXFP4`), `quantized` for every type but F32, or a phase (`prompts`, `decode`).

**Witnesses.** A static catalog describes possible paths; it does not prove which path a call executed. Tests use the actual dispatched path for the measured phase, shape and operation role, through a counter, kernel name or forced-path unit check. The selected plan alone never chooses a looser bound for a path that stayed F32. Check both sides of every threshold and grouped, routed and split callers. Existing Vulkan per-kernel timing names may provide this evidence without a new profiling system.

## 3. Each backend today and under auto (planned)

Today's activation forms, from `main` at 4970a071 and the MXFP4 Vulkan integration branch (`integrate/mxfp4-vulkan`), which is not on `main`:

| Backend | Prompts | Generated rows | Output head | MXFP4 prompts | MXFP4 rows | F32 weights |
|---|---|---|---|---|---|---|
| CPU, AVX2 | f32 except K-quant rows at width >=4096: Q4_K/Q5_K block-int8, Q6_K block-int16 | Q8_0 f32; Q4_0, Q4_1, Q6_K block-int16; Q4_K, Q5_K block-int8 | as generated rows | f32 | block-int16 | f32 |
| Vulkan, MI50 (integer dot preferred) | block-int16 (integer tile) | Q8_0 block-int16; other quantized types block-int8 | Q4_0, Q4_1, Q6_K block-int16 | f32 (predecoded F32 tile, wide range recompute) | block-int16 | f32 |
| Vulkan, Radeon VII | f32 (float tile) | block-int16 | block-int16 | f32 | block-int16 | f32 |

Under auto every Qwen model resolves to F16 on these three, and each path becomes:
- **CPU:** existing F32 prompt paths may remain as a documented wider implementation. Both prompt and generated Q4_K/Q5_K paths that currently read block-int8 activations must move to a qualifying implementation. Q6_K wide prompts and all existing block-int16 paths also need their conformance checks. Calibrating a budget does not itself prove that a path meets it.
- **MI50:** prompts keep the block-int16 tile once it passes the F16 budget with its witness; the block-int8 rows of every quantized type but Q8_0 move to block-int16, measured against main and the reference; MXFP4 prompts take the integer tile through exact copies of their weights (parked on `wip/mxfp4-int16tile-v2`, with a range fallback for weight and activation scales outside the float range), replacing the F32 predecoded tile.
- **Radeon VII:** already F16-class where it is not F32; a block-int16 prompt tile is optional, taken only for speed.
- **A BF16 device:** BF16 kernels; until they exist it resolves to F16 like the others.
- **Operation precision** is not inferred from weight storage. Explicitly retained F32 operations, such as the existing norms and router, stay F32; other matmuls follow the selected policy or report their actual wider fallback. A matrix stored as F32 is not by itself an activation-policy exception.

## 4. The test side (planned)

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

## 5. The dtype flag (planned)

`--dtype auto|f16|bf16|f32` on every model command and `serve`, `auto` by default, after auto lands.
- A misspelt value is a usage error (status 2); a valid value is never refused.
- Where a device lacks the value natively, it takes a fast exact emulation where one exists (BF16 or F16 activations rounded to that type and widened to F32 in the kernel), else F32, and the CLI writes one warning line and the record marks the device `fallback` or `emulated`.
- `f32` means F32 activations on every path, which on Vulkan needs F32-activation row kernels for quantized weights, since today's rows read only the integer twins (open question 2).

## 6. Order of work (planned)

Each step lands on its own branch off `main`, with its gate:

1. **Guidelines.** AGENTS.md's gates and tests section and ROADMAP's rule (branch `docs/gate-guidelines`) and this plan. Gate: the suite's `docs` and `dead-code`, both developers' review, the user's approval.
2. **Resolver, application contract and record.** The registry default, capabilities, resolver, connection to the model's execution context, CLI line and `/v1/health` field. Test resolution and application separately. Intermediate code may describe existing paths, including block-int8, but cannot claim an effective F16 policy until the conformance steps implement it. Keep incomplete policy behavior private and integrate the dependent commits before release. Gate: CTest with a resolver test (declared, preferred, mixed devices, F32 fallback), the CPU suite, Qwen3-0.6B byte identity against main, the hosted run.
3. **Tolerance owner and budgets.** `hf_bounds`, the record parsers, `tools/calibrate_dtype.py` and the frozen budget, every tiny-fixture check witnessing its class. Gate: the suite on the CPU and a device, `reference-consumer` covering the owner, the budget file reviewed before any candidate uses it.
4. **F16 conformance, GPU.** MI50 rows from block-int8 to block-int16 and the Radeon VII unchanged; the MXFP4 Vulkan integration lands with its prompts on the integer tile (`wip/mxfp4-int16tile-v2`: its boundary regressions for scale bytes 0, 1, 103, 104, 143 to 146, 252 and 255, `mxfp4-vulkan` against a reference on the rounded inputs with a witness, and the tile cost of its range guard). Gate: the device tier on both cards, HF error and headroom, speed against main and the reference at default clocks.
5. **F16 conformance, CPU** (the other developer). Q4_K/Q5_K prompt and generated rows from block-int8 to a qualifying implementation, plus the CPU's capabilities, execution-policy application and actual-path witnesses. Gate: the model/kernel tier, independent HF depth checks, batch/split invariance and matched performance against main and mx-llama.cpp.
6. **The flag.** `--dtype`, emulation and F32 fallback on each backend, warnings. Gate: the `cli` component, the suite at each value on the CPU and a device.

What waits:
- Existing CPU MXFP4 support remains on main; its prior evidence is preserved.
- The pending CPU AVX2 decoder optimization, Vulkan MXFP4 integration, int16 tile and new quantization types remain held until the approved unified system is complete, including CPU/GPU conformance and explicit dtype/fallback behavior.
- Steps may be developed in separate dependent branches. Their landing order must leave main truthful about the arithmetic it runs. Any earlier staged quant merge requires an explicit user-approved amendment to this hold.
- Re-run each pending branch's applicable gate against the completed policy and current main; an earlier green run does not approve a later numerical policy.

## 7. Open questions (planned)

1. **MI50 rows at 16 bits.** Moving the MI50's block-int8 rows to block-int16 is what F16 requires; the 16-bit prompt fix cost decode little on these types, but it is to be measured per type. Recommendation: move them in step 4 and record any decode cost as the fix's.
2. **`--dtype f32` on Vulkan.** Quantized rows have no F32-activation kernel. Recommendation: write them in step 6, since a fallback that cannot run F32 would break rule 2.
3. **BF16 emulation.** A device without BF16 emulates it by rounding activations to BF16 and computing in F32 on the float tile. Recommendation: accept that as exact emulation, with its speed recorded.
4. **Existing F32 CPU prompt paths.** Keep a documented F32 implementation where it passes and is faster. Wide K-quant prompts are a separate existing integer path and must also conform; no speed claim for a proposed conversion is made before measurement.
5. **The budget's margin.** Twice the largest error over five weight sets per fixture. Recommendation: keep twice, and recalibrate only with the user's approval, never to admit a failing path.
6. **The record on every run.** One stderr line for every model command. Recommendation: always, since the rule makes precision visible and the tests read it.
7. **GGUF dtype metadata.** A GGUF file carries a quantization type but no source dtype. Recommendation: the architecture's documented default, overridable by the flag; a safetensors path reads `torch_dtype`.
