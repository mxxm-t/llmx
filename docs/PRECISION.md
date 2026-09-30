# Precision (planned)

This is the plan for one precision system across every backend and weight type: how a run's activation dtype is chosen, how each device carries it out, how the tests hold each path to its own tolerance, and the order the work lands in.
Nothing here is built yet; it waits for the user's approval.
Until the plan's first steps land, no quantization branch passes its gates.

## The rule (signed 2026-09-30, planned)

The rule was agreed in the collaboration log on 2026-09-30 and follows vLLM's actual dtype behaviour with a small implementation.

1. **auto** (the default) takes the model's declared dtype where every device of the run runs it natively, else the devices' preferred 16-bit dtype, as vLLM resolves `auto`.
   A GGUF file records no source dtype, so its architecture module declares a documented default: BF16 for qwen3, qwen3moe and qwen35.
   An explicit dtype overrides it.
   On the MI50, the Radeon VII and an AVX2 CPU, auto is F16; on a device that runs BF16 natively it is BF16.
2. **`--dtype auto|f16|bf16|f32`**, after auto lands: a valid value is never refused.
   A device without it natively takes a fast exact emulation, else F32, with a visible warning.
   F16 and BF16 never stand in for each other; F32 is the only wider fallback.
3. **Weights are read exactly and sums are F32.**
   The F32 intermediates llmx keeps today (norms, softmax, rope, recurrent state, residual) stay until a narrower change is measured and passes the gates.
4. **Implementations.** A kernel may implement a dtype in another form, block-scaled 16-bit integers for F16 for instance, only when it passes that dtype's calibrated budget and exact range checks, with a witness that it ran.
5. **One resolution per run.** One owner resolves the dtype once; the requested, effective and native, emulated or fallback dtype of each device is shown once, on the CLI and in `/v1/health`.
6. **Gates.** One tolerance owner keyed by the dtype that ran; a per-dtype budget calibrated and frozen before any candidate is measured; F32 paths keep their bounds; exact conversion and range checks; the real-model gates and within-backend byte identity unchanged; speed at default clocks on one card, first support free to merge with its speed gate open.
7. **Owners.** The resolver, the GPU side, the tolerance owner and the docs are one developer's; the CPU side is the other's.

The ROADMAP entry that states the rule for the gates is `8. Correctness & perf gates`, Precision policy, on the branch `docs/gate-guidelines`.

## 1. The resolver (planned)

**Where it lives.** The resolution reads the model's declared dtype, which the model layer holds, and each device's capabilities, which the backends layer reports, so by the layer rule (AGENTS.md, Architecture) it belongs to the model layer, beside `place_model` in `src/model/place.hpp`, the one entry that already holds a run's weights and backends.
It returns a `DtypePlan` with the placed model; the inference layer keeps it with the loaded model, the CLI writes it and the server reports it.
Nothing below the model layer chooses a dtype, and no kernel chooses one by weight type.

**What it reads.**
- The declared dtype: a column of the architecture registry (`src/model/arch/registry.hpp`), BF16 for the three Qwen entries, copied into `ModelWeights` when a file is read.
  A future safetensors loader reads the checkpoint's `torch_dtype` instead, as vLLM reads the HF config.
- Each backend's native dtypes, preferred 16-bit first (section 2).
- The request: `auto` until the flag lands.

**How it resolves**, as vLLM does at its pinned commit 1117140e (`vllm/config/model.py`, `_resolve_auto_dtype`, [source](https://github.com/vllm-project/vllm/blob/1117140edbaf3d1e1f73b01b09cde3cf98ec24ff/vllm/config/model.py)): the platform's supported dtypes, filtered by the model's validity, give a preferred dtype, the first; a float32 model is taken down to the preferred dtype; the declared dtype is kept where it is supported, and the preferred one taken otherwise.
llmx makes one plan for every device of a run, since a split's stages pass activations to one another:
1. The declared dtype, where every device lists it as native.
2. Else the first device's preferred dtype that every device lists as native.
3. Else F32.
A model that declares F32 takes step 2's dtype, as vLLM takes float32 models down.
vLLM's list of models that refuse F16 has no counterpart until an architecture needs one.

**The record.** Per run: the requested dtype, the declared dtype, the effective dtype, and per device its name, how it carries the effective dtype (native, emulated or fallback) and its paths, the activation form of each kernel family as groups `class (members)` (section 2).
The CLI writes it once to stderr as the model is loaded; on the MI50 after step 4 of the order below it would read:

    dtype: auto -> f16 (model declares bf16); vulkan:0 native: block-int16 (quantized), f32 (F32)

A fallback adds `warning:` and names the dtype it replaced.
`/v1/health` gives the same record as a `dtype` object: `requested`, `declared`, `effective` and `devices`, each with `device`, `how` and `paths`.
The implementation is one struct and one function beside `place_model`, two virtual functions on `Backend` and one registry column; the parked branch `wip/dtype-auto` holds a first version of the resolver and record (without `how`).

## 2. Capabilities and kernel classes (planned)

**A backend's capabilities.** `Backend::native_dtypes()` lists the dtypes whose kernels the backend has, preferred 16-bit first.
A dtype is native only when every kernel family the backend runs implements it at that class; hardware support alone does not make it native.
- CPU (AVX2, F16C, no AVX-512 BF16): F16, F32.
- Vulkan on the MI50 and the Radeon VII: F16 (half arithmetic, and the block-int16 twins below), F32.
- A device with BF16 arithmetic lists BF16 first once its BF16 kernels exist.

**A kernel's class.** `Backend::dtype_path(effective)` names, per kernel family, the form its matmul activations take:
- `f32`, `f16`, `bf16`: activations in that type;
- `block-int16`: blocks of 32 scaled to 16-bit integers, which may implement the F16 class once it passes the F16 budget and its exact range checks (rule 4);
- `block-int8`: blocks scaled to 8-bit integers, which implements no class; under auto a path of this form moves to a 16-bit form (section 3), or is withdrawn.
The groups name weight types (`MXFP4`), `quantized` for every type but F32, or a phase (`prompts`, `decode`).

**Witnesses.** A test proves which path ran twice over: the run's record names the class of every path its weights take (section 4), and a kernel test names the kernel it dispatched (the Vulkan backend's per-kernel times already carry kernel names).
A path that only runs past a threshold, such as a tile from a batch width, is reached by the test's own shapes, never assumed.

## 3. Each backend today and under auto (planned)

Today's activation forms, from `main` at 4970a071 and the MXFP4 Vulkan integration branch (`integrate/mxfp4-vulkan`), which is not on `main`:

| Backend | Prompts | Generated rows | Output head | MXFP4 prompts | MXFP4 rows | F32 weights |
|---|---|---|---|---|---|---|
| CPU, AVX2 | f32 (float dots) | Q8_0 f32; Q4_0, Q4_1, Q6_K block-int16; Q4_K, Q5_K block-int8 | as generated rows | f32 | block-int16 | f32 |
| Vulkan, MI50 (integer dot preferred) | block-int16 (integer tile) | Q8_0 block-int16; other quantized types block-int8 | Q4_0, Q4_1, Q6_K block-int16 | f32 (predecoded F32 tile, wide range recompute) | block-int16 | f32 |
| Vulkan, Radeon VII | f32 (float tile) | block-int16 | block-int16 | f32 | block-int16 | f32 |

Under auto every Qwen model resolves to F16 on these three, and each path becomes:
- **CPU:** prompts stay F32, which is wider than the class and allowed (rule 3); the block-int8 rows of Q4_K and Q5_K move to block-int16, measured as their own change; the rest already meets F16 once calibrated.
- **MI50:** prompts keep the block-int16 tile once it passes the F16 budget with its witness; the block-int8 rows of every quantized type but Q8_0 move to block-int16, measured against main and the reference; MXFP4 prompts take the integer tile through exact copies of their weights (parked on `wip/mxfp4-int16tile-v2`, with a range fallback for weight and activation scales outside the float range), replacing the F32 predecoded tile.
- **Radeon VII:** already F16-class where it is not F32; a block-int16 prompt tile is optional, taken only for speed.
- **A BF16 device:** BF16 kernels; until they exist it resolves to F16 like the others.
- **F32 weights** (the tiny fixtures, routers, some norms) stay F32 everywhere.

## 4. The test side (planned)

**One tolerance owner.** `tests/common.py` holds every tiny-fixture bound, keyed by the dtype class the checked path ran in (`hf_bounds`), which the run's record witnesses (`dtype_record`, and `witnessed_class` for the least precise path a weight type takes).
F32 keeps its bounds, 2e-5 on logits and 1e-5 on NLL.
The per-type constant it replaces, MXFP4's 2e-4, goes; file-exact goldens keep only HF's outputs and their metadata, never a tolerance.

**Calibration.** `tools/calibrate_dtype.py` measures each 16-bit class against HF in float32 in the pinned HF environment: HF with the weights exact, the input of every matrix product but the routers' rounded to the dtype, and F32 sums, over the tiny fixtures (the dense F32 model tied and untied, the MoE model, the three MXFP4 fixtures) at their own weights and four further seeds.
A class's budget is twice the largest error, frozen in `tests/data/dtype_budget.json` before any candidate path is measured against it, and changed only with the user's approval.
The parked branch's first measurement: F16 at most 0.00165 logits and 0.000148 NLL, budgets 0.0033 and 0.00030; BF16 at most 0.0270 and 0.0040, budgets 0.054 and 0.0079.
The F16 budget is fifteen times the MXFP4 integer tile's measured 0.00021952 and was not set from it.

**Kernel checks.** A kernel of a 16-bit class is checked against an independent reference on its actually rounded inputs, within the sum of the weights' magnitudes times each input's rounding error plus the F32 accumulation's, and its conversion and range handling exactly: every scale, zero and sign, the extreme exponents, and the fallback where a product can leave the float range.

**Unchanged gates.** The real-model gates keep their bounds: file-exact HF NLL and rankings, the near-tie rules, the long-context check (`tools/long_context_check.py`, device against itself and within its margin of the CPU), and within-backend byte identity, batch invariance and split identity.
A path whose numerics change reports its HF error and headroom; a path that does not stays byte-identical.

## 5. The dtype flag (planned)

`--dtype auto|f16|bf16|f32` on every model command and `serve`, `auto` by default, after auto lands.
- A misspelt value is a usage error (status 2); a valid value is never refused.
- Where a device lacks the value natively, it takes a fast exact emulation where one exists (BF16 or F16 activations rounded to that type and widened to F32 in the kernel), else F32, and the CLI writes one warning line and the record marks the device `fallback` or `emulated`.
- `f32` means F32 activations on every path, which on Vulkan needs F32-activation row kernels for quantized weights, since today's rows read only the integer twins (open question 2).

## 6. Order of work (planned)

Each step lands on its own branch off `main`, with its gate:

1. **Guidelines.** AGENTS.md's gates and tests section and ROADMAP's rule (branch `docs/gate-guidelines`) and this plan. Gate: the suite's `docs` and `dead-code`, both developers' review, the user's approval.
2. **Resolver and record.** The registry column, the two `Backend` functions, `resolve_dtype`, the CLI line and the `/v1/health` field, with every device's paths recorded as they are today, block-int8 included. No kernel changes. Gate: CTest with a resolver test (declared, preferred, mixed devices, F32 fallback), the CPU suite, Qwen3-0.6B byte identity against main, the hosted run.
3. **Tolerance owner and budgets.** `hf_bounds`, the record parsers, `tools/calibrate_dtype.py` and the frozen budget, every tiny-fixture check witnessing its class. Gate: the suite on the CPU and a device, `reference-consumer` covering the owner, the budget file reviewed before any candidate uses it.
4. **F16 conformance, GPU.** MI50 rows from block-int8 to block-int16 and the Radeon VII unchanged; the MXFP4 Vulkan integration lands with its prompts on the integer tile (`wip/mxfp4-int16tile-v2`: its boundary regressions for scale bytes 0, 1, 103, 104, 143 to 146, 252 and 255, `mxfp4-vulkan` against a reference on the rounded inputs with a witness, and the tile cost of its range guard). Gate: the device tier on both cards, HF error and headroom, speed against main and the reference at default clocks.
5. **F16 conformance, CPU** (the other developer). Q4_K and Q5_K rows from block-int8 to block-int16, the CPU's capabilities and paths. Gate: the CPU tier and speed against main.
6. **The flag.** `--dtype`, emulation and F32 fallback on each backend, warnings. Gate: the `cli` component, the suite at each value on the CPU and a device.

What waits:
- The MXFP4 CPU support on `main` keeps its gates until step 3 re-keys its bound to the witnessed class.
- The MXFP4 Vulkan integration waits for step 3, and its prompt path for step 4.
- `wip/mxfp4-int16tile-v2` waits for steps 3 and 4 and lands with or after the Vulkan integration.
- Any new quantization type waits for step 3.

## 7. Open questions (planned)

1. **MI50 rows at 16 bits.** Moving the MI50's block-int8 rows to block-int16 is what F16 requires; the 16-bit prompt fix cost decode little on these types, but it is to be measured per type. Recommendation: move them in step 4 and record any decode cost as the fix's.
2. **`--dtype f32` on Vulkan.** Quantized rows have no F32-activation kernel. Recommendation: write them in step 6, since a fallback that cannot run F32 would break rule 2.
3. **BF16 emulation.** A device without BF16 emulates it by rounding activations to BF16 and computing in F32 on the float tile. Recommendation: accept that as exact emulation, with its speed recorded.
4. **CPU prompts in F32 under F16.** They are wider than the class and faster to keep than to narrow. Recommendation: keep them F32 until a narrower form is measured faster.
5. **The budget's margin.** Twice the largest error over five weight sets per fixture. Recommendation: keep twice, and recalibrate only with the user's approval, never to admit a failing path.
6. **The record on every run.** One stderr line for every model command. Recommendation: always, since the rule makes precision visible and the tests read it.
7. **GGUF dtype metadata.** A GGUF file carries a quantization type but no source dtype. Recommendation: the architecture's documented default, overridable by the flag; a safetensors path reads `torch_dtype`.
