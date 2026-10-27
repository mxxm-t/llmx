# NVFP4 default depth reference proposal (2026-10-25)

Status: method approved by F2DEV at 10:54 and by the owner through LDEV at 10:55 on 2026-10-25. The separate wording commit is `8e84e4c4f`; it has landed on main. Candidate depth limits remain unchanged. The later reference-only self-control decisions are recorded in [PRECISION](../PRECISION.md#independent-hf-depth-acceptance-agreed-2026-10-01). The separate explicit-F32 raw/chat depth checks pass 512/512 exact each. [STATUS](../STATUS.md) records subsequent capture and comparison results; the planned method below is retained as its design record.

## Planned scope

Only the released Qwen3.5-0.8B file with SHA256 `71a0a29da9481f51d8e301f8ffb67feec1627cd18e92183a1f3918b0a7c35728`, the frozen raw/chat prompts of 16384 tokens, `--ubatch 512`, and their 511 forced continuation steps. HF reads this file's independently decoded and globally scaled F32 weights. Its parameters, matrix sums and recurrent state remain F32. K and V round to F16 at the cache-write boundary, after K normalization and RoPE.

The file has 186 transformer matrices: 108 NVFP4, 22 BF16, 26 Q8_0, nine Q4_K and 21 Q5_K. Eighteen two-dimensional convolution weights use the causal-convolution operation and are excluded from this matrix count. Embedding gather is separate. The tied Q8_0 output head is one additional matrix operation.

## Proposed operation policy

| Matrix role | Radeon prefill | Radeon decode | MI50 prefill | MI50 decode | CPU prefill | CPU decode |
| --- | --- | --- | --- | --- | --- | --- |
| NVFP4 and BF16 transformer weights | F32 | F32 | F32 | F32 | F32 | F32 |
| Q4_K and Q5_K transformer weights | F32 | Block-int16 | Block-int16 | Block-int16 | F32 | Block-int16 |
| Q8_0 transformer weights | F32 | Block-int16 | Block-int16 | Block-int16 | F32 | F32 |
| Tied Q8_0 output head | Block-int16 | Block-int16 | Block-int16 | Block-int16 | F32 | F32 |

The head is not an ordinary prefill projection. The runtime gives its final prompt row an extent of one even when transformer rows keep the full prompt extent. On Radeon the ordinary Q8 row reads the packed activation twin despite the profile not preferring integer-dot instructions; MI50 uses its vector integer kernel. Both represent block-int16 inputs. CPU keeps Q8 products F32.

The source-derived map joins every matrix descriptor to the existing HF parameter map. The runtime's projection owners and dispatch table determine the policy, not comparison outcomes. The local preparation artifact is `build-nvfp4-depth-reference/default-policy/projection-map.json`. It pins the model, source files, shapes and all 186 HF module names. The [initial Radeon dispatch witness](nvfp4-default-dispatch-20261025.json) confirms the profile and both phases: 12131 timed prefill dispatches and 306 after one forced step, each with the Q8 row head present. NVFP4 float tile/row kernels, the generic float prompt tiles and the companion decode rows are present. No logits were emitted or compared. Per-projection attribution combines these phase-specific kernel records with the pinned source map. [Refreshed Radeon evidence](nvfp4-main3-hf-progress-20261025.json) and the [CPU witness](nvfp4-current-cpu-dispatch-20261025.json) cover runtime `4f026d941`. MI50 remains source-derived until its own witness is collected.

Block-int16 reconstruction uses an independently implemented mathematical quantizer over groups of 32 inputs, with F32 scale computation, signed codes from -32767 through 32767, and reconstruction before HF multiplication. Vulkan rounds halfway away from zero; CPU rounds ties to even. Those are distinct fixed policies, not interchangeable references. The producer refuses nonfinite or extreme-range inputs outside the ordinary rule. Existing operator/range checks cover the runtime's separate extreme-range implementation. The approved general input-reconstruction rule covers this method. The separate TQ known-mismatch exception does not apply to NVFP4.

HF chunking must not choose the arithmetic. A 511-row continuation evaluated in one HF call still represents 511 runtime decode steps. Each reference run uses its device's fixed table above throughout the entire workload, including prompt-to-decode transition. Extra unscored HF head rows are discarded and cannot affect subsequent state.

## Required checks before acceptance

- Completed method agreement under [PRECISION](../PRECISION.md#independent-hf-depth-acceptance-agreed-2026-10-01), before comparing default-depth outputs.
- Independent reconstruction controls, exact HF module coverage and F16 cache-write controls, with invocation counts and finite outputs retained.
- An HF-only chunked/full control that separates reference execution effects from runtime errors.
- The unchanged 512-position exact/top-two rule with margin 0.1, preserving every position and all failures. No TQ baseline exception applies.
- The completed F32 reference retained as quality information alongside the selected-policy comparison. Short-model bounds, NLL, calibration, split, range, identity and performance requirements stay unchanged.

## Preparation evidence

The independent implementation controls pass six reconstruction cases, four positive-zero cases and seven refused-input cases. They also check the exact names and shapes of all 187 HF Linear modules on the released configuration, without loading released weights. Separate hybrid-cache controls retain exact F32 convolution/recurrent updates and round only full-attention K/V writes through F16. The [CPU BF16 split record](nvfp4-cpu-bf16-split-20261025.json) pins these control receipts separately from its CPU qualification.

The producer keeps the original strict-F32 producer unchanged. Its HF-only control uses fresh logical F16 caches on both arms, with a 64-token prompt followed by 16 forced continuation tokens. One arm batches each phase; the other splits the prompt into 17-token chunks and feeds continuation tokens singly. Both switch arithmetic at the same prompt boundary. Module call/row counts and cache-write coverage are checked explicitly. The 16K producer uses 32 prompt chunks and one 511-row continuation call under the fixed logical decode policy. The original narrowed self-control failures, their rounding diagnosis and the scoped reference-only decisions are retained in PRECISION and STATUS. Radeon and CPU reference artifacts are verified, and their [operation-matched raw/chat comparisons](nvfp4-cpu-radeon-default-depth-20261025.json) pass all 512 positions in each history. The record retains the Radeon cross-run control hash difference separately from its passing frozen self-control bound. Device witnesses and source guards must cover the runtime being compared.

This is neither a pass nor permission to merge. A failed comparison still requires diagnosis.
