# llmx — Test assets

Where the real models and corpora used for manual verification live. These are
environment-specific paths (this dev machine); the automated suite
(`tests/run_tests.py`) generates its own synthetic fixtures and needs none of
them.

> Superseded once `llmx pull` lands (`docs/ROADMAP.md` #9a): the hardcoded paths
> below become a cache the tool manages. Until then, this file is the record of
> what is on this machine.

## Model locations

Models are kept in the LM Studio model directory:

```
C:\Users\Marko\.lmstudio\models\
```

llmx reads **Q8_0 / Q4_0 / Q4_1 / Q4_K / Q5_K / Q6_K / F32** tensors (see
`docs/src/format-gguf.md`). Dense Qwen3 Q4_K_M and Q5_K_M mixtures are supported;
support for their tensor encodings does not add new architectures. Dense F32
embeddings, matrices and norms are supported, including tied output weights.

| Model                                            | Format | Status                       |
|--------------------------------------------------|--------|------------------------------|
| `Qwen\Qwen3-8B-GGUF\Qwen3-8B-Q8_0.gguf` (8.11 GB)| Q8_0   | **Usable** — the real-model gate |
| `Qwen\Qwen2-0.5B-Instruct-GGUF\...fp16.gguf`     | FP16   | Not yet supported            |
| `lmstudio-community\...\Qwen3-30B...Q4_K_M.gguf` | Q4_K_M | Quant supported; architecture not validated (MoE is unsupported) |
| `lmstudio-community\...\Qwen3-Coder...Q4_K_M.gguf`| Q4_K_M | Quant supported; architecture not validated (MoE is unsupported) |
| `unsloth\...\Qwen3.5-4B-BF16.gguf`               | BF16   | Not yet supported            |
| `unsloth\...\mmproj-F32.gguf`                    | F32    | Multimodal projector (not a main model) |

These assets exercise both tensor-format coverage and architecture support
(`docs/ROADMAP.md` #1-3). Check both before selecting a validation model.

Use the Qwen3-8B Q8_0 model for the manual real-model checks that the suite
can't cover — e.g. the lossless correctness gate (path-controlled perplexity)
and real throughput:

```
llmx.exe generate C:\Users\Marko\.lmstudio\models\Qwen\Qwen3-8B-GGUF\Qwen3-8B-Q8_0.gguf "The capital of France is" -n 32
```

## Gate fixture models (HF cache)

`tests/baseline.py` looks these up in the Hugging Face cache automatically, and
skips if they are absent. `LLMX_BASELINE_GGUF` overrides the lookup.
Keep the fixture's original filename when using the override: it selects the
quantization-specific logit/PPL bounds. Only that model's numerical checks run
when an override is set; unsupported filenames are rejected.

| Repo / file | Why this one |
|---|---|
| `Qwen/Qwen3-0.6B-GGUF` / `Qwen3-0.6B-Q8_0.gguf` | Small enough to gate on, and the tokenizer golden's model |
| `unsloth/Qwen3-0.6B-GGUF` / `Qwen3-0.6B-Q4_0.gguf` | **Load-bearing.** Mixed Q4_0/Q4_1/Q6_K/F32, and its Q6_K `token_embd` has a subnormal super-block scale. The Q8_0 fixture has almost no subnormal scales (0.0061% of blocks against 5.89% in Qwen3-8B), so without this model the logit gate is blind to the f16 subnormal bug class - it passed with that bug deliberately reintroduced until this was added. |

Fetch and SHA-256 verify the pinned snapshots with
`python tools/fetch_test_models.py` (Python standard library only, about 1 GB
combined). Revisions and digests are recorded in `tests/baseline.py`; the
numerical checks use those exact snapshots unless explicitly overridden.

### Fixed-excerpt HF perplexity gate

`tests/data/baseline_perplexity.json` records an HF float32 reference from
`Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`.
Regenerate it with `python tools/gen_baseline.py perplexity` in the isolated
HF environment described by that script. Generation uses CPU eager attention;
the JSON records torch/transformers versions, the exact text and its SHA-256,
token IDs, target count, mean NLL and PPL.

The text is the first 1024 Unicode characters of `wiki.test.raw`, with line
endings normalized to LF before extraction. Its 247 tokens form one continuous
sequence with no added BOS/EOS. Every token after the first is scored against
the preceding tokens (246 targets); log-softmax and the reduction use float64
on the HF float32 logits. The suite writes the stored text bytes to a temporary
file and invokes `perplexity --file`, so checkout newline settings do not change
the test input.

The HF reference is mean NLL **3.360285580**, PPL **28.797413678**. Two repeated
llmx measurements per quant gave Q8_0 NLL **3.36166** / PPL **28.8371** and mixed
Q4_0 NLL **3.49184** / PPL **32.8463**. The absolute mean-NLL bounds are **0.01**
and **0.16**, respectively (about 1.01% and 17.35% relative PPL). These bounds
allow quantization error; they do not establish lossless inference. The gate
also requires exact HF token IDs/count and finite, mutually consistent NLL/PPL.

The same pinned reference also scores disjoint 64-token windows (all four, or
the first two) and 123-token windows (two, omitting the singleton tail).
Positions/KV reset each window, and NLL is weighted by scored targets.

| Context / limit | HF NLL | Q8_0 NLL | Mixed Q4_0 NLL |
|---|---:|---:|---:|
| 64 / all | 4.030360346 | 4.037034329 | 4.197941852 |
| 64 / 2 | 3.710157365 | 3.712550543 | 3.779062509 |
| 123 / all | 3.630793905 | 3.643202014 | 3.751242730 |

Chunked mean-NLL bounds are **0.02** for Q8_0 and **0.20** for mixed Q4_0;
the tighter continuous bounds above remain unchanged. An independent control
running the previous scoring loop on the same token windows matched total
NLL exactly for both quants. This establishes unchanged window arithmetic,
not equivalence to the full-precision weights.

A diagnostic with two-token windows gave HF NLL **11.543536540**, Q8_0
**11.737387723**, Q4_0 **9.244354366**, identically under old/new scoring.
Those large quantized/full-precision differences are not covered by the
useful bounds above; the real-model gate uses contexts 64 and 123. Minimum
context handling is tested separately using mathematically known, nonuniform
probabilities in `tests/perplexity.py`. These short excerpts do not establish
full-corpus or long-context numerical correctness.


## F32 reference coverage

`tests/f32.py` creates a small dense Qwen3 fixture using deterministic binary
fractions. `tools/gen_baseline.py f32` generates its committed full-logit and
NLL references with HF `Qwen3ForCausalLM`, float32 eager attention, torch
2.5.1+cpu and transformers 4.55.2. No HF dependency is needed to run the tests.
The golden file records weight hashes and configuration so fixture drift fails.

Two layers, hidden width 37, FFN width 19, head width 10, GQA 2:1 and vocabulary
257 exercise scalar tails and partial row blocks. Five prompt lengths, physical
batches 1/2/3/5/16, threads 1/4 and tied/untied output weights cover prefill;
continuous and four-token-window PPL cover sequential decode and resets.
Bounds are 2e-5 absolute for every logit and 1e-5 for mean NLL. Observed maximum
logit error for the original head-width-8 fixture was 7.2e-7 on Windows MSVC
and Linux GCC, including UBSan. The attention refactor changes head width to
10, exercising vector attention and its scalar tail against regenerated HF
goldens with unchanged tolerances.
A preceding 34-byte quantized tensor checks the loader's float alignment under
UBSan; the old packed blob layout fails with a misaligned float load.

For a real-model check, Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca` was converted with llama.cpp
`convert_hf_to_gguf.py --outtype f32` at converter commit
`407d0bb1f12eaa49882d2714a93fe774eb6806f2`. This converter is an external
fixture-generation tool, not a runtime dependency. It also required
sentencepiece 0.2.2 in the isolated reference environment.

The resulting `Qwen3-0.6B-F32.gguf` is 3,012,480,832 bytes, SHA-256
`41583f438fd2af4ac7fa30c63701a4d1f4010bc68f6cabf773a924c23d377265`.
All 311 tensor arrays were compared to the original safetensors: each is an
exact BF16-to-F32 widening, with no quantization. On the workstation it is at
`%TEMP%/Qwen3-0.6B-F32.gguf`; regenerate rather than relying on this scratch path.

It passed all 20 tokenizer and six logit-ranking reference cases. The existing
247-token excerpt gave PPL 28.7974; all three chunked cases also matched HF
within 1e-4 mean NLL (largest printed delta 4.5e-6). This is bounded agreement,
not bit-identical arithmetic with HF or full-corpus validation.

A control that copies F32 rows before calling the same float kernels produced
identical matrix output hashes to direct access. For a 1,943-token wikitext
prompt plus 32 greedy tokens, both paths also produced identical token IDs and
all-logit hash `11549319204765092529` (64-bit FNV-1a). This checks the storage
access change at that depth, not the model's maximum context.

F32 1024x1024 microbenchmarks on Ryzen 5800X, MSVC /O2 /arch:AVX2, with one
warmup followed by eight alternating process pairs:

| Threads / columns | Copy rows: mean / median ms | Direct rows: mean / median ms |
|---|---:|---:|
| 1 / 1 | 0.07901 / 0.07834 | 0.04327 / 0.04091 |
| 1 / 3 | 0.10408 / 0.10278 | 0.06024 / 0.05969 |
| 1 / 128 | 2.47139 / 2.46553 | 2.36848 / 2.35731 |
| 6 / 1 | 0.03640 / 0.03720 | 0.03101 / 0.03052 |
| 6 / 3 | 0.04454 / 0.04486 | 0.03300 / 0.03303 |
| 6 / 128 | 0.73990 / 0.73273 | 0.71517 / 0.68938 |

The six-thread, 128-column ranges overlap substantially; no end-to-end speedup
or mx-llama.cpp parity is inferred. The previous runtime rejected F32 weights,
so the comparison above is against a working copy-buffer control. Existing Q8
bench mean throughput was 114.53 -> 114.44 GFLOPS, synthetic prefill
4108 -> 4250 tok/s and decode 4126 -> 4141 tok/s under the same eight-pair
measurement; these guardrails do not establish a quantized-path speedup.

### Matched external CPU benchmark

`tools/compare_cpu.cpp` builds against either llmx or the public mx-llama.cpp
C API. `tools/compare_cpu.py` feeds both binaries the committed HF token IDs:
215 prompt tokens followed by 32 forced continuation tokens. This measures
model execution, with loading, tokenization and sampling excluded. It is not
a greedy-generation or numerical-correctness test. Both arms default to six threads,
ubatch 128, F32 KV and causal attention; the reference disables GPU offload and
flash attention. Each process runs one warmup sequence and one measured
sequence with cleared KV state. The driver alternates arm order, rejects
failed runs and invalid timings, and saves raw stdout/stderr, hashes and
mean/median/range summaries. The default is eight process pairs.

Pass `--threads N` to the Python driver to set the same positive thread count
in both arms (up to 64). The C++ wrapper accepts the same optional flag after
the model and token-file arguments. Both wrappers report the requested count
with each sequence, and the driver rejects a missing or mismatched count.
Rebuild both wrappers when updating this tool; older wrappers are rejected.

The reference is public mx-llama.cpp
`5542318e748c154b634211def405ae95da3dfaa9`, built in a clean detached worktree.
The eight-pair measurement on Ryzen 7 5800X, Windows, MSVC 19.50,
llmx `2131c1b` and the exact F32 model above found:

| Phase | llmx mean / median tok/s | mx mean / median tok/s |
|---|---:|---:|
| Prefill | 275.95 / 276.81 | 391.85 / 394.26 |
| Decode | 13.08 / 12.99 | 14.49 / 14.52 |

Raw per-process timings, artifact hashes and ranges are committed in
[`benchmarks/f32-cpu-20260919.json`](benchmarks/f32-cpu-20260919.json).
Prefill ranges are 251.37-290.44 vs 373.97-398.54 tok/s; decode ranges are
12.75-13.41 vs 14.13-14.73 tok/s. An earlier five-pair run also failed the
floor (285.14 vs 375.98 prefill, 13.14 vs 14.27 decode); the runs are not
pooled, and changes between runs are not treated as code speedups.

The external floor is not met. A three-pair diagnostic of 16x12, 32x12 and
16x24 output-row/activation tiles did not show a consistent prefill benefit.
A fully spinning worker-pool diagnostic worsened prefill. Neither experiment
is in the runtime. Profiling the original path attributes about 170-180 ms
of a 720-770 ms prefill to scalar attention. A first AVX2 attention diagnostic
reached about 600 ms, but changes summation order and has not passed the HF
gate or a repeated performance comparison. It is an investigation target,
not a validated speedup.

To reproduce on Windows, use a Visual Studio x64 developer command prompt.
The example assumes the two repositories are siblings, with the clean
reference checkout named `llmx-ref`. Run from the llmx root:

```bat
cmake -S ../llmx-ref -B ../llmx-ref/build -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=OFF -DGGML_BLAS=OFF -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_VULKAN=OFF -DGGML_SYCL=OFF
cmake --build ../llmx-ref/build --config Release --target llama --parallel 6
cl /nologo /std:c++17 /O2 /EHsc /W4 /arch:AVX2 /I src /Fe:"%TEMP%\llmx-compare.exe" /Fo:"%TEMP%\llmx-compare.obj" tools\compare_cpu.cpp
cl /nologo /std:c++17 /O2 /EHsc /W4 /arch:AVX2 /DLLMX_COMPARE_REFERENCE /I ..\llmx-ref\include /I ..\llmx-ref\ggml\include /Fe:..\llmx-ref\build\bin\Release\llmx-compare.exe /Fo:"%TEMP%\mx-compare.obj" tools\compare_cpu.cpp /link ..\llmx-ref\build\src\Release\llama.lib
python -X utf8 tools/compare_cpu.py --llmx "%TEMP%\llmx-compare.exe" --reference ../llmx-ref/build/bin/Release/llmx-compare.exe --reference-revision 5542318e748c154b634211def405ae95da3dfaa9 --model "%TEMP%\Qwen3-0.6B-F32.gguf" --output "%TEMP%\llmx-cpu-comparison"
```

Use a new output directory for each run. The reference executable lives beside
its DLLs, which the driver also hashes on Windows. Record the actual source
revision and build flags when changing either build; the revision argument
is provenance supplied by the caller, not a source-to-binary verification.

### CPU attention validation

The CPU attention refactor shares decode and prefill behind
`Backend::attention`, with backend-owned scratch and AVX2 dots/value sums.
The new dot reduction is not bit-identical to the old scalar attention.
Windows/MSVC and Linux/GCC full suites passed with required real Q8_0 and
mixed Q4_0 fixtures. Linux UBSan passed the synthetic suite, including the
head-width-10 HF fixture; the largest tiny-model logit error was 6.2e-7.
Changing the causal limit to include future batch tokens in a temporary build
made this fixture fail with error 0.03787 under its unchanged 2e-5 bound.
Real Qwen3-0.6B F32 excerpt and chunked NLL checks stayed within 4.5e-6 of HF
under the existing 1e-4 bound.

A separate HF float32/eager check used the first 8,192 normalized Unicode
characters of `wiki.test.raw`, followed by
`\nSummarize the main topics above.\n`. The UTF-8 prompt SHA-256 is
`b6b45b11937fb5a556ba2b29545100d2cbdc8749e3f60ad40de54830d26b9b22`;
both tokenizers produce 1,943 tokens without added special tokens. The same
pinned HF snapshot and exact F32 GGUF described above were used, with six
threads and llmx ubatch 128. HF ran `use_cache=True, logits_to_keep=1` for
prefill and then 32 greedy one-token steps. Binary float32 logits from both
paths were compared after prefill and each step: all 5,013,888 values were
finite, maximum absolute error 0.00012517, RMS error 0.00001581, under a
preselected 0.001 absolute bound. All 32 greedy tokens matched exactly.
This validates that depth and these inputs; it does not cover maximum context,
full-corpus perplexity or bit-identical arithmetic. Scratch outputs and the
reference scripts are under `%TEMP%/llmx-attention-long` and
`%TEMP%/llmx-attention-hf-long.py` on the validation workstation.
The prompt provenance, generated IDs and error summary are committed in
[`benchmarks/attention-hf-20260919.json`](benchmarks/attention-hf-20260919.json).

Eight interleaved before/after/mx rounds, each with a warmup sequence, using
the same 215+32-token F32 benchmark and settings above:

| Phase | Before mean / median tok/s | After mean / median tok/s | mx mean / median tok/s |
|---|---:|---:|---:|
| Prefill | 286.84 / 286.84 | 357.88 / 356.79 | 385.64 / 395.13 |
| Decode | 13.34 / 13.35 | 13.57 / 13.63 | 14.20 / 14.30 |

Prefill improves 24.8%; its before/after ranges do not overlap. Decode's mean
change is 1.7%, but ranges overlap, so it is not a strong speedup claim.
The candidate remains 7.2% below the reference mean for prefill and 4.4%
below for decode. Neither F32 nor attention is merged to main yet.

Q8 synthetic guardrails used eight alternating pairs after a warmup pair,
`bench --size 2048 --iters 2000 --threads 6`. Mean/median matmul GFLOPS were
123.07/125.29 -> 121.80/121.80; prefill 4828/4808 -> 4800/4755 tok/s;
decode 4522/4511 -> 4875/4868 tok/s. All ranges overlap; this is a regression
guard, not proof of a quantized-model speedup. An initial 20-iteration
matmul sample had wide 90.96-140.95 GFLOPS variation, so the matrix run was
lengthened to resolve that concern. Raw timings, hashes and the normalized
source hashes for the candidate are in
[`benchmarks/attention-cpu-20260919.json`](benchmarks/attention-cpu-20260919.json).

### CPU row streaming and parallel prefill

Single-column F32 matrices now use contiguous `dot_f32` rows; batched matrices
retain fused row/column kernels. Independent batched norm, per-head norm/RoPE
and SiLU rows use the existing pool when there are at least two rows per
worker. Smaller batches and single-thread execution stay serial. Weights and
quantized decode kernels are unchanged. F32 dot reductions change order.

The final guarded build passed Windows/MSVC and Linux/GCC full suites with
required real Q8/Q4 HF fixtures, plus Linux UBSan's synthetic suite (real models
intentionally skipped in that sanitizer run). Real F32 HF logits, excerpt and
chunked NLL pass under the existing bounds. Against the pinned long-prompt HF
reference above, all 5,013,888 logits are finite, maximum error is 0.00012636
and RMS error 0.00001616 under 0.001; all 32 greedy IDs match. Guarded and
initial-candidate long outputs are byte-identical. This is bounded agreement
at that depth, not maximum-context coverage or bit identity with HF.

Eight final interleaved rounds on the same F32 model, tokens and settings,
with `637bbdd` as the before arm:

| Phase | Before mean / median tok/s | Guarded mean / median tok/s | mx mean / median tok/s | Guarded vs mx mean |
|---|---:|---:|---:|---:|
| Prefill | 371.65 / 372.48 | 383.06 / 386.48 | 398.97 / 400.67 | -4.0% |
| Decode | 13.87 / 13.96 | 14.68 / 14.70 | 14.89 / 14.90 | -1.4% |

Versus the control, mean prefill improves 3.1% with overlapping ranges, so
this is not a strong prefill speedup claim. Decode improves 5.8% with disjoint
control/candidate ranges. Both external floors remain unmet; main is unchanged.
These final measurements supersede the initial unconditional candidate's run,
which had appeared above mx for prefill. The runs are not pooled, and neither
code changes nor reference drift are inferred from cross-run absolute rates.

The batched Q8 guard calls `Model::prefill()` on the existing deterministic
2-layer synthetic model (E=256, FF=1024, H=8, HK=2, D=32, vocab=512, seed=12345).
It covers 1/2/4/8/16/64 tokens and 1/6 threads, ten warmups and 100 timed
resets/prefills per cell, across four alternating pairs. Unconditional
parallel dispatch regressed B=2/4 latency by 19%/17%, motivating the guard.
After the fix, small-batch and one-thread ranges overlap the control; B=64
retains disjoint ranges:

| Batch, six threads | Before mean ms | Guarded mean ms |
|---|---:|---:|
| 2 | 0.279 | 0.284 |
| 4 | 0.283 | 0.283 |
| 64 | 1.302 | 1.075 |

Printed full-vector sums match in every cell; HF checks separately gate
numerics. The legacy `bench` prefill label measures repeated `Model::step()`,
so it does not cover this batched path. Its eight-pair Q8 regression guard
passes with overlapping ranges: mean matmul 128.55 -> 123.45 GFLOPS, reported
prefill 5120 -> 5108 tok/s and decode 5027 -> 5112 tok/s.

Raw timings, artifact/source hashes, diagnostic selection, the rejected
unconditional small-batch result, activation witnesses and HF errors are in
[`benchmarks/row-scheduling-cpu-20260919.json`](benchmarks/row-scheduling-cpu-20260919.json).
Scratch harnesses and logs are under `%TEMP%/llmx-cpu-gap`, with final artifacts
under `guarded/`. Reader alignment was not adopted. Subsequent instrumented
pool profiling is diagnostic only: roughly 26-28 ms of the 32-token decode
occurs after the final worker callback finishes, motivating investigation of
bounded completion polling. It is not a validated optimization.

### Bounded completion polling: not adopted

Caller-side bounded polling was tested against `c4436fd`, retaining the mutex
and condition-variable fallback. An atomic-counter-only arm separated the
counter change from polling. The initial three-round diagnostic covered zero,
128 and 512 pauses; the shorter budget proceeded to eight interleaved rounds:

| Phase | Current mean tok/s | Atomic only | Poll 128 | mx |
|---|---:|---:|---:|---:|
| Prefill | 382.30 | 392.78 | 385.23 | 400.76 |
| Decode | 14.42 | 14.59 | 14.58 | 14.87 |

Ranges overlap. Polling does not establish a benefit over the atomic-only
control and still misses both mx means. No worker-pool code was adopted.
The workstation had unrelated interactive CPU activity; no user processes
were changed and no timing outliers were removed. This is insufficient
evidence for a small speedup, not proof that polling can never help.

A separate instrumented build witnessed 15,746 calls: 4,392 completed within
the polling budget and 11,354 used the blocking path. A 64,000-job stress run
passed across changing thread counts, yields and deliberately delayed workers.
That checks publication/lifetime behavior, not HF numerical correctness. The
rejected candidate did not proceed to a full HF gate. Raw timings, hashes,
the exact patch and the stress source are preserved in
[`benchmarks/completion-polling-20260919.json`](benchmarks/completion-polling-20260919.json).

### Attention query scheduling: not adopted

A scratch change schedules independent head/query pairs cyclically across
workers, with one score row per worker. Per-query arithmetic is unchanged
in source. Three interleaved rounds against `c4436fd` show:

| Phase | Current mean tok/s | Candidate | mx |
|---|---:|---:|---:|
| Prefill | 388.08 | 399.48 | 395.48 |
| Decode | 14.57 | 14.46 | 14.83 |

Prefill control/candidate ranges are disjoint; decode ranges overlap. A
subsequent eight-round run did not establish external parity:

| Phase | Current mean tok/s | Candidate | mx |
|---|---:|---:|---:|
| Prefill | 378.01 | 391.81 | 405.30 |
| Decode | 14.66 | 14.66 | 14.87 |

Scheduling remains scratch-only. Combining it with the value kernel below
added no clear diagnostic benefit. Only finite outputs/top-token/printed-sum
smoke checks ran for scheduling; it did not proceed to a full HF gate.
The exact patch and raw samples are in
[`benchmarks/attention-scheduling-diagnostic-20260919.json`](benchmarks/attention-scheduling-diagnostic-20260919.json).
Scratch files are under `%TEMP%/llmx-attention-balance`.

### Register attention value accumulation

The next candidate normalizes attention coefficients once and retains output
sums in registers across the KV sequence. It uses 32-lane blocks, eight-lane
remainders and scalar tails, preserving sequence order per output lane.
Head scheduling and worker-pool behavior are unchanged.

Final measurements use the same Ryzen 7 5800X, F32 Qwen3-0.6B weights, pinned
215 prompt plus 32 forced continuation tokens, six threads, ubatch 128 and
F32 KV as the previous comparison. Control is `c4436fd`; mx remains
`5542318e748c154b634211def405ae95da3dfaa9`. Eight rounds rotate/reverse arm
order; each process runs one warmup and one measured sequence. Loading,
tokenization and sampling are excluded. No builds/tests ran concurrently.
All samples are retained; unrelated interactive activity was not controlled.

| Phase | Control mean / median tok/s | Candidate mean / median | mx mean / median | Mean gap vs mx |
|---|---:|---:|---:|---:|
| Prefill | 384.86 / 387.00 | 397.63 / 396.58 | 397.94 / 398.27 | -0.08% |
| Decode | 14.56 / 14.58 | 14.57 / 14.61 | 14.89 / 14.97 | -2.10% |

Prefill improves 3.32% against its control, with narrowly overlapping ranges.
Decode differs by 0.11%, within overlapping ranges. Prefill is close to mx,
but external parity remains unproven and decode still trails. This is a
development checkpoint, not a merge or an external-floor pass. These final
results supersede the three-round candidate-selection diagnostic; runs are
not pooled or compared by their absolute throughput across sessions.

The deterministic HF fixture now uses head width 42, exercising the full
32-lane block plus eight-lane and scalar tails. Tensor shapes and HF head
configuration derive from the same width. The generator still uses original
HF model code, and numerical bounds are unchanged.

| Numerical gate | Measured | Bound / expected |
|---|---:|---:|
| Tiny HF maximum logit error | 0.00000070 | 0.00002 |
| Forced scalar-value branch maximum error | 0.00000070 | 0.00002 |
| Missing SIMD-block output mutant error | 0.19888665 | Must fail 0.00002 |
| Long-prompt HF maximum logit error | 0.00012636 | 0.001 |
| Long-prompt HF RMS logit error | 0.00001616 | Recorded diagnostic |
| Greedy IDs matching HF | 32/32 | 32/32 |
| Long logits byte-identical to c4436fd | 5,013,888 | All recorded values |

The long case covers 1,943 prompt tokens plus 32 greedy tokens, not the model's
maximum context or the full corpus. All values were checked finite. The
concatenated vector hash is
`986be83255fa7e25c17b987c3d6f27f58b0f65e6dd84d90568bf4ac819b9d702`.
Windows and Linux full suites pass with required Q8/Q4 HF fixtures; real F32
HF logits and excerpt/window NLL pass. Linux UBSan synthetic tests pass;
real model fixtures were deliberately skipped in that sanitizer run.
The scalar-value test still enables AVX elsewhere and is not an ISA portability
test. The root CLI rebuild has an identical code section to the tested CLI.

The eight-pair Q8 step guard and four-pair batched guard have overlapping
control/candidate ranges. The latter calls real `Model::prefill()` at batches
1/2/4/8/16/64 and one/six threads, with ten warmups and 100 timed iterations.
The legacy `bench` prefill label still calls repeated `step()`.

| Q8 step guard | Control mean | Candidate mean |
|---|---:|---:|
| Matmul, GFLOPS | 126.56 | 127.61 |
| Reported prefill, tok/s | 5088.48 | 5066.90 |
| Decode, tok/s | 5155.80 | 5080.45 |

Instrumented profiling separately places about 2,073 ms of the 2,187 ms decode
in matrix operations, including 528 ms in the output projection; attention
takes about 87 ms. This diagnostic identifies the next investigation target;
it is not a comparative benchmark or proof of achievable savings.

All final/diagnostic timing samples, the scheduling follow-up, hashes,
validation logs and reproduction harnesses are in
[`benchmarks/attention-values-cpu-20260919.json`](benchmarks/attention-values-cpu-20260919.json).
Scratch artifacts are under `%TEMP%/llmx-attention-values/validation`.

### Paired decode and packed prefill: not adopted

Two further matrix-kernel investigations use `5a9518c` as the control and
the same pinned mx build, F32 model, 215+32 tokens, six threads and ubatch 128.
Each diagnostic has three interleaved rounds with a warmup per process.

The paired-row decode kernel shares activation loads while retaining four
independent accumulation chains per row. All 12,642 synthetic outputs are
byte-identical to the single-row control across odd rows, dimension tails and
one/four/six threads, but decode throughput does not improve:

| Mean tok/s | Control | Paired rows | mx |
|---|---:|---:|---:|
| Prefill | 372.33 | 401.07 | 398.85 |
| Decode | 14.25 | 14.25 | 14.57 |

A large control prefill outlier prevents attributing the apparent prefill gain
to this decode change. All samples were retained. The candidate was rejected
without proceeding to a full HF gate. Its patch, harness and raw samples are
in [`benchmarks/paired-decode-diagnostic-20260919.json`](benchmarks/paired-decode-diagnostic-20260919.json).

Packed F32 prefill uses activation panels and a six-row by sixteen-token
kernel. Scalar packing first regressed mean prefill from 403.48 to 334.59 tok/s.
A vectorized transpose improved the prototype; an intrusive profile measured
about 19 ms packing time. A follow-up comparison separated vectorized packing,
packing weights as well, and processing only full panels with the existing
kernel for remaining tokens:

| Mean tok/s | Control | Vectorized activation packing | Also pack weights | Full panels + tail fallback | mx |
|---|---:|---:|---:|---:|---:|
| Prefill | 379.36 | 348.64 | 300.56 | 293.27 | 371.42 |
| Decode | 14.23 | 14.04 | 14.34 | 14.24 | 14.27 |

All packed variants are slower on prefill and were rejected. Their smoke
checks establish finite logits and matching top tokens only, not full-vector
HF correctness. No runtime or fixture change was adopted. Raw samples,
patches and profiling sources are in
[`benchmarks/packed-prefill-diagnostic-20260919.json`](benchmarks/packed-prefill-diagnostic-20260919.json).
These runs occurred on an interactive workstation; results from separate
sessions are not pooled or used to infer a change in the reference.

That artifact also records a separate intrusive profile of unchanged `5a9518c`
arithmetic. During 2,237.50 ms decode, SiLU takes 7.45 ms caller wall time and
softmax takes 9.42 ms summed worker time. Softmax runs across workers, so its
sum is not elapsed time. These components are small relative to matrix work;
an approximate exponential was not implemented on this evidence.

### Q8 external floor on two model sizes

The validated `5a9518c` runtime was compared directly with public mx
`5542318e748c154b634211def405ae95da3dfaa9` on Qwen3-0.6B Q8_0 and the real
Qwen3-8B Q8_0 asset. Both arms receive the same pinned 215-token prompt and
32 forced continuation IDs used above, with six threads, ubatch 128 and F32
KV. The benchmark bypasses tokenization; the same token stream is reused
for both model sizes. Model loading and sampling are excluded. The reference
uses context 512, no GPU offload, no flash attention and no BLAS.
Three alternating pairs each run one warmup and one measured sequence;
no other agent builds/tests ran concurrently and no samples were removed.

| Q8_0 model / phase | llmx mean / median tok/s | mx mean / median | Mean gap vs mx |
|---|---:|---:|---:|
| Qwen3-0.6B prefill | 380.50 / 383.68 | 265.35 / 265.79 | +43.39% |
| Qwen3-0.6B decode | 42.73 / 42.86 | 46.81 / 47.06 | -8.71% |
| Qwen3-8B prefill | 25.94 / 26.21 | 20.07 / 20.03 | +29.27% |
| Qwen3-8B decode | 4.10 / 4.11 | 4.39 / 4.36 | -6.64% |

Each phase has disjoint llmx/mx sample ranges on each model. Prefill exceeds
the reference in these runs, while decode still misses the floor. This does
not establish parity for other models, quants or hardware. Historical 8B
stand-in measurements used different prompts/thread counts and are not
pooled with these runs. No throughput improvement is claimed from comparing
their absolute values.

The 0.6B model hash is the required HF Q8 fixture digest recorded above.
The 8B file is 8,709,518,112 bytes, SHA-256
`408b955510e196121c1c375201744783b5c9a43c7956d73fc78df54c66e883d6`.
Source/fixture/benchmark hashes still match the validated `5a9518c` artifact.
Its Windows/Linux required HF fixture checks cover 0.6B; the 8B timing smoke
check does not add an independent 8B HF correctness baseline.

At the pinned reference revision, CPU Q8 type traits select activation
conversion through `quantize_row_q8_0` and an integer Q8-by-Q8 dot. llmx's
current fused dot consumes float activations. This source difference
motivates an experiment, not a claim that activation quantization will close
the gap or preserve quality. Any such candidate needs a measured numerical
cost bound in addition to performance validation.

Raw output, samples, hashes, flags and reproduction scripts are in
[`benchmarks/q8-external-floor-20260919.json`](benchmarks/q8-external-floor-20260919.json).
Scratch artifacts are under `%TEMP%/llmx-q8-floor`.

### Integer activation experiments: not adopted

Scratch AVX2 kernels consume the existing Q8_0 weights with quantized
activations, using either signed 8-bit or signed 16-bit activation values and
F32 block scales. Activation conversion happens once per matrix. Invalid or
too-small activation ranges use the existing float path. The 16-bit scratch
layout is internal to the CPU kernel and does not add a GGUF weight format.

Scalar activation packing erased the initial integer-dot benefit. Vectorized
packing produced a modest gain; unrolling the dot added no further benefit.
The 16-bit representation performed similarly to 8-bit activations while
retaining much closer excerpt numerics. Final diagnostic comparisons use the
same `5a9518c` control, pinned mx build, models and 215+32 tokens as above,
six threads, ubatch 128 and F32 KV. Each has three interleaved rounds with
per-process warmup; all samples are retained and no builds/tests compete.

| Q8_0 model / phase | Control mean tok/s | Q16 activation candidate | mx | Candidate vs mx |
|---|---:|---:|---:|---:|
| Qwen3-0.6B prefill | 417.79 | 412.84 | 275.50 | +49.85% |
| Qwen3-0.6B decode | 43.79 | 45.01 | 47.44 | -5.12% |
| Qwen3-8B prefill | 28.65 | 28.31 | 21.25 | +33.23% |
| Qwen3-8B decode | 4.29 | 4.47 | 4.52 | -1.21% |

Decode improves 2.79% / 4.06% against the respective controls, with disjoint
control/candidate ranges. Prefill control/candidate ranges overlap. Neither
decode comparison establishes the external floor; no candidate is adopted.
Absolute rates from other sessions are not pooled with these runs.

Independent scalar packing and signed-product controls match 5,504 activation
values and 2,352 outputs exactly for each precision. Cases include -128 weight
bytes, half subnormal scales, zero and extreme-sign blocks, one/four/six
threads, and nonfinite/tiny-input fallbacks. Test compilation exposes private
members in a copied header only; production APIs are unchanged. A separately
instrumented model run witnesses 197 / 253 packed calls on 0.6B / 8B, with no
fallback calls, for the one-token `hello` prompt.

Existing Windows HF tokenizer, ranking, continuous NLL and chunked NLL checks
pass for both activation precisions without changing tolerances. This is not
a full platform-suite certification. The continuous excerpt shows:

| Absolute NLL error versus HF | Measured | Existing bound |
|---|---:|---:|
| Current float activations | 0.001374 | 0.010 |
| Q8 activations, vectorized packing | 0.009316 | 0.010 |
| Q16 activations, vectorized packing | 0.001354 | 0.010 |

A path-controlled comparison uses identical weights, token IDs and scoring
for current llmx, a control that reconstructs Q16 activations and calls the
original float dot, and the Q16 integer kernel. Over four continuous/chunked
excerpt cases:

| Maximum absolute NLL shift | Measured |
|---|---:|
| Activation conversion with the original float dot | 0.00003266 |
| Integer kernel versus that control | 0.00005188 |
| Complete candidate versus current llmx | 0.00003155 |

These are whole-model effects: later activations can diverge, so this is not
an isolated per-layer rounding decomposition. They do not prove full-corpus
quality or losslessness.

The longer Q8 test feeds identical 1,943 prompt tokens and 32 forced HF
continuation tokens to every arm. Prompt IDs were independently checked
against the pinned HF tokenizer. All 33 full vectors (5,013,888 finite logits)
were compared, including the prefill output:

| Comparison | Maximum absolute logit error | RMS error | Top token agrees |
|---|---:|---:|---:|
| Current Q8 llmx vs original HF | 0.999404 | 0.110809 | 33/33 |
| Reconstructed-Q16 float control vs HF | 0.998318 | 0.110775 | 33/33 |
| Q16 integer candidate vs HF | 0.999252 | 0.110796 | 33/33 |
| Q16 candidate vs current Q8 llmx | 0.002213 | 0.000263 | 33/33 |

The candidate's maximum and RMS distance to HF are slightly smaller in this
case, but its added precision change is nonzero. No new global acceptance
bound is inferred from the result. Full-corpus, independent 8B HF,
maximum-context and cross-platform validation are still unproven.

All prototype patches, raw samples, numerical controls, activation witnesses,
vector hashes and reproduction sources are in
[`benchmarks/q8-integer-activation-20260919.json`](benchmarks/q8-integer-activation-20260919.json).
Scratch artifacts are under `%TEMP%/llmx-q8-integer`. Runtime remains the
validated `5a9518c` code; the next investigation targets projection dispatch
while preserving its float arithmetic.

## Grouped CPU projections checkpoint (2026-09-19)

The selected implementation groups independent Q/K/V and FFN gate/up decode
projections into one pool dispatch. Existing native F32/Q8_0/Q4_K row kernels
and per-matrix partitions are preserved; other types, batches and small jobs
use sequential matmul. The scratch both-phase experiment offered no advantage.
No integer activation quantization or TLS dispatcher is included.

Final matched Qwen3-0.6B comparison, Ryzen 7 5800X / MSVC AVX2, six threads,
ubatch 128, F32 KV, the same 215 HF prompt IDs and 32 forced continuation IDs.
Each process warms up; an outer warmup round is also discarded, followed by
eight interleaved rounds. Loading, tokenization and sampling are excluded.
All samples, including outliers, are retained.

| Model / phase, tok/s | Previous mean | Grouped mean | mx mean | Previous median | Grouped median | mx median |
|---|---:|---:|---:|---:|---:|---:|
| Q8_0 prefill | 418.59 | 419.06 | 267.76 | 419.13 | 422.01 | 266.69 |
| Q8_0 decode | 43.92 | 45.18 | 48.34 | 44.02 | 45.43 | 48.88 |
| F32 prefill | 397.41 | 394.25 | 390.07 | 399.12 | 396.92 | 390.94 |
| F32 decode | 14.47 | 14.64 | 14.86 | 14.59 | 14.70 | 14.89 |

The Q8 decode mean improves, but remains below mx; the F32 decode floor is
also unmet. F32 control/candidate ranges overlap. This checkpoint is unmerged.
The follow-up grouped 8B comparison is recorded below; its decode floor also
remains unmet.

| Validation | F32 | Q8_0 |
|---|---:|---:|
| Exact excerpt/window NLL cases vs previous runtime | 4/4 | 4/4 |
| Long-prompt full logits byte-identical vs previous | 5,013,888 | 5,013,888 |
| Maximum added logit difference | 0 | 0 |

Long checks use the previously pinned 1,943-token prompt and 32 continuation
steps. F32 greedy output matches HF; Q8 uses forced HF continuation for matched
inputs. F32 maximum error vs HF remains 0.00012636 under 0.001. These are scoped
checks, not full-corpus or maximum-context proof. Independent HF tests remain
mandatory; equality against llmx alone would not establish correctness.

Windows/Linux full suites pass with required real Q8/Q4 HF fixtures, and the
real F32 HF gate passes. UBSan synthetic suite and backend CTest pass; UBSan
intentionally skips real-model fixtures. Backend CTest checks 540 cases and
141,750 outputs on each platform, including mixed formats, uneven rows,
batches, single/empty groups, output boundaries and invalid types. A missing-row
mutant fails. Instrumented Q8 model execution observes 3,584 groups and 5,376
avoided dispatches across warmup plus measurement; its timings are not used.

Synthetic matmul median is 129.78 -> 129.51 GFLOPS with overlapping ranges;
step-based synthetic prefill/decode means improve 4998/4928 -> 6861/6921 tok/s.
The separate batched guard has overlapping ranges except its faster
six-thread single-token case (0.232 -> 0.173 ms). No test tolerance changed.

Evidence, raw samples, prototype patches, source/binary hashes, controls and
reproduction harnesses are in
[`benchmarks/grouped-projections-cpu-20260919.json`](benchmarks/grouped-projections-cpu-20260919.json).
Scratch artifacts: `%TEMP%/llmx-grouped-projections/validation`. Comparator source
is `tools/compare_cpu.cpp`; reference remains public mx commit `5542318e74`.
Run `ctest --test-dir build -C Release --output-on-failure` after a CMake build
for the new backend tests. CI runs them on each configured CPU job; hosted CI
for this unmerged checkpoint has not run.

## Grouped projections on Qwen3-8B (2026-09-19)

Same Q8_0 model and pinned 215+32 token IDs, six threads, ubatch 128 and F32 KV.
Three measured interleaved rounds after outer and per-process warmups; loading
is excluded. These are diagnostics, not proof of a sub-percent speedup.

| Qwen3-8B Q8_0, mean tok/s | Previous 5a9518c | Grouped b6a890f | mx |
|---|---:|---:|---:|
| Prefill | 29.12 | 29.25 | 21.20 |
| Decode | 4.31 | 4.33 | 4.49 |

Control/group ranges overlap; the external decode floor remains open. A
separate harness loads the same weight object once and runs the previous and
current implementations on identical histories, comparing every full vector.

| Equality check vs previous llmx | Result |
|---|---:|
| Prompt tokens | 215 |
| Forced continuation tokens | 32 |
| Full vectors compared | 33 |
| Finite logits byte-identical | 5,013,888 |
| Maximum added error | 0 |

This is same-weight equality, not an independent 8B HF reference or a
maximum-context test. The binary hash, output hash, raw timing samples and
reproduction sources are in
[`benchmarks/grouped-projections-8b-20260919.json`](benchmarks/grouped-projections-8b-20260919.json).
Scratch: `%TEMP%/llmx-grouped-projections/validation/eight-b-*`.

## Q8 scale/load scheduling (2026-09-19)

The selected native Q8 row kernel broadcasts the stored half scale directly
from memory and loads signed byte groups directly into widening instructions.
It preserves float activations and the existing per-lane FMA/reduction order.
Assembly and a real-model activation witness confirm the intended path. Feature
specialization is excluded: its decode median was similar and prefill lower.

Matched comparisons use the same models, reference commit, settings and token
histories documented above. These are final interleaved means after outer and
per-process warmups; loading, tokenization and sampling are excluded.

| Qwen3-0.6B mean tok/s | Before b6a890f | Candidate | mx |
|---|---:|---:|---:|
| Q8 prefill | 422.99 | 418.43 | 273.51 |
| Q8 decode | 45.30 | 46.54 | 48.35 |
| F32 prefill | 398.17 | 395.23 | 394.92 |
| F32 decode | 14.70 | 14.77 | 14.98 |

| Comparison scope | Result |
|---|---:|
| Measured rounds per arm/model | 8 |
| Q8 decode mean improvement vs before | 2.74% |
| Q8 decode mean gap vs mx | -3.73% |
| F32 decode mean gap vs mx | -1.39% |

Q8 prefill and F32 timing ranges overlap. No external decode parity is claimed.
All outliers are retained; do not pool absolute rates across separate sessions.

| Numerical check | Observed | Bound / requirement |
|---|---:|---:|
| Tiny F32 HF maximum logit error | 0.00000070 | 0.00002 |
| Long F32 HF maximum logit error | 0.00012636 | 0.001 |
| Long F32 HF greedy IDs | 32/32 | 32/32 |
| Long F32 values byte-identical to b6a890f | 5,013,888 | All compared values |
| Long Q8 values byte-identical to b6a890f | 5,013,888 | All compared values |
| Excerpt/window NLL cases identical, per format | 4/4 | 4/4 |
| 8B Q8 values byte-identical to b6a890f | 5,013,888 | All compared values |
| Finite half-scale/signed-weight exact cases | 253,952 | All cases |

The long smaller-model histories, NLL scopes and HF references are as above.
The larger-model equality case uses the pinned comparison history and compares
full vectors with explicit old/new backend objects sharing the same weights.
It is not an independent HF 8B reference or a full-corpus/max-context claim.
The scale regression rejects a wrong-half-offset mutant. Windows and Linux
full suites require the real Q8/Q4 HF fixtures; real F32 HF checks pass. Backend
CTest passes on both platforms. UBSan synthetic/backend tests pass, with real
fixtures intentionally skipped in that sanitizer configuration. Production
comparator and root CLI code hashes match the measured/validated binaries.

The larger-model diagnostic uses the same settings with fewer measured rounds:

| Qwen3-8B Q8 mean tok/s | Before b6a890f | Candidate | mx |
|---|---:|---:|---:|
| Prefill | 28.97 | 29.04 | 21.13 |
| Decode | 4.29 | 4.45 | 4.49 |

| Diagnostic scope | Value |
|---|---:|
| Measured rounds per arm | 3 |
| Decode mean improvement vs before | 3.65% |
| Decode mean gap vs mx | -0.94% |

Before/candidate decode ranges are disjoint, while candidate/mx ranges overlap.
This short diagnostic does not establish external parity.

The synthetic hot-path guard improves the Q8 matrix-vector kernel; the CLI
labels it "matmul" but it calls `matvec_q8_0`. Its "prefill" is repeated
`step()`, so the separate batched guard remains necessary.

| Synthetic guard, mean | Before | Candidate |
|---|---:|---:|
| Q8 matrix-vector GFLOPS | 128.52 | 158.98 |
| Repeated-step prefill tok/s | 6919.28 | 7165.46 |
| Decode tok/s | 7077.54 | 7528.79 |

The short batched guard initially showed disjoint slower ranges in the
threaded single-token case. A longer follow-up retains all samples and shows
substantial overlap, with paired differences in both directions. The higher
threaded candidate means remain recorded; this does not prove zero slowdown.

| Longer synthetic prefill, mean ms | Before | Candidate |
|---|---:|---:|
| 1 thread, 1 token | 0.16995 | 0.12580 |
| 1 thread, 2 tokens | 0.23972 | 0.23734 |
| 6 threads, 1 token | 0.15864 | 0.16167 |
| 6 threads, 2 tokens | 0.23358 | 0.23651 |

| Follow-up settings | Value |
|---|---:|
| Interleaved measured pairs | 8 |
| Discarded outer warmup pairs | 1 |
| Warmup iterations per case | 200 |
| Timed iterations per case | 2000 |
| Six-thread single-token mean latency change | +1.91% |
| Six-thread single-token paired change range | -5.18% to +7.86% |

Raw samples from both guards, model timings, exact-output hashes, HF logs,
platform results, prototype patches, assembly and reproduction scripts are in
[`benchmarks/q8-scale-load-cpu-20260919.json`](benchmarks/q8-scale-load-cpu-20260919.json).
Scratch: `%TEMP%/llmx-q8-scale-load/validation`. The checkpoint is unmerged;
external performance floors and hosted CI remain open.

## Q8 row instruction studies (2026-09-19)

Control is `475f312`, with native memory-scale broadcasts and direct signed
byte loads already selected. These scratch experiments retain per-row float
arithmetic and accumulator order. None is adopted.

Paired rows share each activation load while processing adjacent weight rows.
Assembly confirms that sharing and no inner-loop accumulator spills. Decode
regresses with disjoint ranges in the initial matched model diagnostic.

| Paired-row study, mean tok/s | Control | Paired | mx |
|---|---:|---:|---:|
| Qwen3-0.6B Q8 prefill | 416.65 | 403.09 | 271.13 |
| Qwen3-0.6B Q8 decode | 46.00 | 43.81 | 48.31 |

Direct pointer increments remove repeated block-address multiplication from
the native loop. They do not rescue paired-row decode. The longer single-row
comparison does not establish a decode gain; its prefill ranges overlap.

| Pointer study, mean tok/s | Control | Pointer | Paired + pointer | mx |
|---|---:|---:|---:|---:|
| Initial prefill | 413.38 | 421.58 | 420.93 | 272.19 |
| Initial decode | 45.91 | 46.10 | 44.41 | 47.79 |
| Longer prefill | 413.06 | 419.71 | Not retested | 273.86 |
| Longer decode | 46.22 | 45.86 | Not retested | 48.22 |

Explicit inlining removes all assembly calls to the native Q8 row helper but
increases code size and does not establish a model decode gain. Adding pointer
increments to that variant also fails to establish a gain.

| Inlining study, mean tok/s | Control | Inline | Inline + pointer | mx |
|---|---:|---:|---:|---:|
| Prefill | 418.15 | 416.33 | 418.29 | 271.44 |
| Decode | 46.56 | 46.04 | 46.38 | 48.46 |

| Comparator code / scoped checks | Control | Each applicable prototype |
|---|---:|---:|
| Q8 row-helper calls in assembly | 3 | 0 with explicit inlining |
| Comparator virtual code bytes | 316,008 | 320,392 with explicit inlining |
| Exact matrix cases per prototype | Reference | 504 passed |
| Exact matrix outputs per prototype | Reference | 14,328 matched |
| Finite scale/weight output checks per prototype | Expected products | 507,904 passed |
| Grouped backend cases, paired prototype | Double oracle / separate calls | 540 passed |

The direct paired-scale test checks both rows, all finite half scales and
signed extremes with one-hot activations. Matrix cases include zero/odd row
counts, varied block lengths, output sentinels and different thread counts.
Native AVX2/F16C, forced software-half SIMD and forced scalar paths are checked
on AVX2 hardware; this does not prove a non-AVX binary target. A passing printed
model sum is only diagnostic, not a model-level numerical gate. These rejected
prototypes were not promoted to cross-platform, full HF, long-context or
larger-model validation.

| Measurement setting | Value |
|---|---:|
| Short-study measured rounds per arm | 3 |
| Longer pointer-study measured rounds per arm | 8 |
| Discarded outer warmup rounds | 1 |
| Per-process warmup sequences | 1 |
| Threads | 6 |
| Ubatch | 128 |
| Prompt / forced decode tokens | 215 / 32 |

Model, token hashes and mx reference are unchanged from the preceding study.
All samples are retained; loading, tokenization and sampling are excluded.
Do not combine absolute rates across these separate sessions. Evidence is in
[`benchmarks/q8-row-instructions-20260919.json`](benchmarks/q8-row-instructions-20260919.json),
including patches, assembly extracts, source/binary hashes, raw samples and
reproduction scripts. Scratch: `%TEMP%/llmx-q8-paired`. Runtime remains `475f312`.

## Matched thread scaling and projection shapes (2026-09-19)

The shared comparator and runner now accept `--threads`; both arms use the
requested value for prefill and decode, report it per sequence, and the runner
rejects missing or mismatched values. The default is unchanged. Windows llmx
and mx wrappers build, as does the Linux llmx wrapper. Invalid argument checks,
thread-metadata rejection checks and a real-model runner smoke pass. Both
runtime sources are unchanged; llmx remains `475f312`, with the same pinned mx
revision and model/token hashes used above.

| Measurement setting | Value |
|---|---:|
| Default threads | 6 |
| Accepted thread range | 1-64 |
| Measured rounds per model/count/arm | 3 |
| Discarded outer warmups per setting | 1 |
| Per-process warmup sequences | 1 |
| Prompt / forced decode tokens | 215 / 32 |
| Ubatch | 128 |

Both models use F32 KV and the prior reference flags. Arm order alternates and
thread-setting order rotates. Default C++ argument handling is exercised in
outer warmups at the default count. All samples, including the slower F32
samples, are retained. These are scaling diagnostics, not a universal parity
claim or a reason to discard the established external floor.

| Qwen3-0.6B Q8_0, mean tok/s | llmx prefill | mx prefill | llmx decode | mx decode |
|---|---:|---:|---:|---:|
| Threads 1 | 105.54 | 83.95 | 25.82 | 29.87 |
| Threads 2 | 189.88 | 161.61 | 39.93 | 44.53 |
| Threads 4 | 311.32 | 249.15 | 47.47 | 47.92 |
| Threads 6 | 423.46 | 274.01 | 45.84 | 48.65 |
| Threads 8 | 428.27 | 328.42 | 46.50 | 48.04 |
| Threads 16 | 513.28 | 453.54 | 45.15 | 43.46 |

| Qwen3-0.6B F32, mean tok/s | llmx prefill | mx prefill | llmx decode | mx decode |
|---|---:|---:|---:|---:|
| Threads 1 | 102.51 | 106.38 | 12.28 | 12.22 |
| Threads 2 | 189.22 | 202.10 | 14.93 | 14.95 |
| Threads 4 | 301.86 | 321.55 | 14.61 | 15.33 |
| Threads 6 | 383.07 | 390.20 | 13.99 | 14.63 |
| Threads 8 | 371.79 | 430.43 | 14.10 | 14.15 |
| Threads 16 | 496.01 | 497.88 | 13.99 | 13.73 |

Decode saturates earlier than prefill, and the best observed count differs
between workloads. Selecting a favorable setting does not establish the floor
at other counts. Runtime defaults and scheduling have not changed.

Correctness was checked separately from timing using the same prompt and
forced continuation. Every full output vector matches the default-count
control byte for byte. Continuous-excerpt NLL is exactly unchanged from the
prior validated runtime and passes the independently generated HF fixture.

| Numerical check | F32 | Q8_0 |
|---|---:|---:|
| Thread settings checked | 6 | 6 |
| Counts checked | 1, 2, 4, 6, 8, 16 | 1, 2, 4, 6, 8, 16 |
| Full vectors per setting | 33 | 33 |
| Finite values per setting | 5,013,888 | 5,013,888 |
| Added error across counts | 0 | 0 |
| HF excerpt NLL absolute error | 0.000000078466 | 0.001378187469 |
| Existing HF NLL bound | 0.0001 | 0.010 |

These full-precision NLL errors differ slightly from older CLI-rounded
summaries. The HF check here is excerpt NLL; vector equality across thread
counts is a separate consistency check. Earlier independent long F32 HF
validation remains documented above. No full-corpus, maximum-context or new
independent larger-model correctness claim is made.

Instrumented projections and attention use the existing runtime backend,
with timing around each top-level operation. Nested fallback calls are not
double-counted. These timings include pool dispatch/wait within an operation;
total time also includes instrumentation and is excluded from benchmark data.

| Instrumented decode, mean ms, 6 threads | F32 | Q8_0 |
|---|---:|---:|
| Attention | 87.86 | 76.22 |
| Q/K/V projections | 408.31 | 115.93 |
| FFN gate/up | 600.66 | 169.35 |
| Attention output projection | 210.91 | 63.52 |
| FFN down | 308.13 | 89.06 |
| Vocabulary output | 534.59 | 138.06 |
| Total including instrumentation | 2172.73 | 673.69 |

The matrix probes cycle through each layer's actual weights with fixed
synthetic activations. Q/K/V and gate/up use grouped projections in llmx and
a shared CPU graph in mx. The reference uses a persistent threadpool and
preallocated tensor outputs/workspace; Q8 activation conversion is timed.
Loading and graph construction are excluded. These probes isolate kernel and
dispatch costs, not full-model quality or throughput.

| Matrix/group, mean ms, 6 threads | F32 llmx | F32 mx | Q8 llmx | Q8 mx |
|---|---:|---:|---:|---:|
| Q/K/V | 0.45645 | 0.45001 | 0.12258 | 0.10952 |
| Attention output | 0.22749 | 0.22488 | 0.05409 | 0.04372 |
| FFN gate/up | 0.67401 | 0.67320 | 0.18083 | 0.17291 |
| FFN down | 0.33850 | 0.34063 | 0.08395 | 0.07811 |
| Vocabulary output | 16.30322 | 16.98024 | 4.16707 | 3.99393 |

| Matrix-probe scope | Value |
|---|---:|
| Thread counts | 1, 4, 6, 16 |
| Interleaved process pairs per model/count | 3 |
| Warmup cycles per matrix case | 4 |
| Timed cycles per matrix case | 64 |
| Layer jobs per cycle | 28 |
| Vocabulary-output jobs per cycle | 1 |

At the recorded default count, F32 arm ranges overlap for every matrix case;
Q8 llmx is slower with disjoint ranges. This makes more F32 dot-instruction
changes a lower-priority hypothesis. Attention still has measurable cost in
both models. Source inspection shows token-major K/V storage makes successive
positions of a head strided across token records. A contiguous per-head cache
is the next bounded layout experiment; this is a hypothesis, not a measured
win. Capacity must grow with used context rather than allocating the model's
full maximum context up front. Preserve arithmetic order and validate growth,
reset, batched prefill, full vectors and independent HF outputs before adoption.

Raw samples, wrapper/DLL/model hashes, exact-vector hashes, HF checks,
projection profiles, matrix ranges and reproduction sources are in
[`benchmarks/cpu-thread-scaling-20260919.json`](benchmarks/cpu-thread-scaling-20260919.json).
Scratch: `%TEMP%/llmx-cpu-scaling`. The measurement checkpoint does not merge
or publish the unlanded runtime stack; external performance floors remain open.

## Initial head-major KV diagnostic (2026-09-19)

Scratch `%TEMP%/llmx-kv-head-major` compares control `342960a` (runtime
`475f312`), a concrete host-cache helper with contiguous history per head,
and the same public mx reference used above. No runtime code was adopted at
that initial checkpoint.
There is an outer warmup round and a warmup sequence in every process; three
measured rounds rotate arm order. Same pinned model, tokens, six threads,
ubatch 128, F32 KV and inference-only timing. All samples, including the slower
Q8 candidate decode sample, are retained.

| Mean tok/s | Current | Scratch KV candidate | mx |
|---|---:|---:|---:|
| Q8 prefill | 423.69 | 461.67 | 289.01 |
| Q8 decode | 47.08 | 48.51 | 50.24 |
| F32 prefill | 403.41 | 427.26 | 430.29 |
| F32 decode | 15.20 | 15.33 | 15.32 |

Q8 prefill arm ranges are disjoint; decode ranges overlap and the candidate
mean still trails mx. F32 ranges overlap for both phases. This short diagnostic
does not establish the external floors. Growth/reset/mixed-history outputs
match the control exactly across all 229,758 values, including multiple KV
heads and tail widths. Tiny independent HF fixtures pass at maximum logit
error 7e-7. Real-model HF, long-context and platform gates remain pending.
The investigation paused for the follow-up chat correctness fix, then resumed
with the validation and selection recorded below.

[`benchmarks/head-major-kv-initial-20260919.json`](benchmarks/head-major-kv-initial-20260919.json)
contains all samples, hashes, prototype sources, harnesses and initial checks.

## Head-major KV validation and selection (2026-09-19)

The selected layout centralizes CPU storage in `model/host_kv_cache.hpp` and
passes an explicit head stride to backend attention. It grows with used
context, preserving each layer/head prefix, and retains capacity across resets.
The logical sequence length remains in `Model`. No arithmetic order changes.
The source headers adopted into the repository are byte-identical to the
validated candidate; the final MSVC build also has the same `.text` hash.

The longer matched run uses the same pinned weights, token IDs, public mx
revision and CPU settings as the initial run: six threads, ubatch 128, F32 KV,
prompt 215 plus 32 forced continuation IDs. Nine measured rounds rotate the
three arms, with an outer warmup round and a warmup sequence in each process.
Loading, tokenization and sampling are excluded. No validation/build work
overlaps timing, and all outliers are retained. These sessions are not pooled
with the earlier short run.

| Mean tok/s | Control 475f312 | Head-major KV | mx | Change vs control | Gap vs mx |
|---|---:|---:|---:|---:|---:|
| Q8 prefill | 366.46 | 416.37 | 262.34 | +13.62% | +58.72% |
| Q8 decode | 42.36 | 44.41 | 45.51 | +4.82% | -2.42% |
| F32 prefill | 345.42 | 361.38 | 362.44 | +4.62% | -0.29% |
| F32 decode | 13.06 | 13.47 | 13.27 | +3.16% | +1.55% |

Control/candidate ranges overlap. Paired candidate wins are respectively
9/9, 7/9, 7/9 and 8/9. Q8 decode and F32 prefill still trail mx in the means;
selection is an incremental improvement, not closure of the external floor.
The previous table remains a separate diagnostic, not a substitute for these
longer-run results.

A separate eight-pair synthetic comparison uses `bench --size 2048 --iters 10
--threads 6 --p 64 --n 64`, with an outer warmup pair. Its legacy prefill
metric is repeated `step()`, not batched prefill. All ranges overlap, and
the lower candidate decode mean remains a limitation rather than evidence
of universal non-regression.

| Synthetic mean | Control | Head-major KV |
|---|---:|---:|
| Matmul GFLOPS | 116.34 | 117.22 |
| Step-prefill tok/s | 5994.64 | 6005.85 |
| Decode tok/s | 6314.19 | 6132.35 |

| Correctness / integration | Result | Scope |
|---|---:|---|
| F32 long logits byte-identical to control | 5,013,888 | Prompt 1943 + 32 decode steps |
| Q8 long logits byte-identical to control | 5,013,888 | Same prompt + forced HF continuation |
| NLL cases identical, per format | 4 / 4 | Continuous and disjoint windows |
| F32 maximum error vs independent HF | 0.00012636185 | Bound 0.001 |
| F32 greedy IDs matching HF | 32 / 32 | Long prompt |
| Mixed-history values identical, per platform | 229,758 | Windows / Linux, distinct KV heads |
| Direct storage relocation mutant | Rejected | Wrong head copied on growth |

Windows/Linux full suites pass with required real-model HF fixtures. Linux
UBSan native/synthetic suites pass, and the separate Linux mixed-history
harness uses nonrecovering UBSan. The final CTest integration includes the
direct `kv-cache` oracle alongside backend and chat-template tests. All pass;
the Windows oracle also passes. The long gate is not a full-corpus or
maximum-context claim. Growth keeps old and replacement allocations alive
together until successful completion, so future memory budgets must account
for that temporary peak.

All samples, ranges, hashes, compiler commands, platform logs, exact-vector
hashes and reproduction harnesses are in
[`benchmarks/head-major-kv-cpu-20260919.json`](benchmarks/head-major-kv-cpu-20260919.json).
Scratch lives at `%TEMP%/llmx-kv-head-major`, including `validation/` and
`bench-nine/`. The validated runtime stack remains unmerged until its external
performance requirements are met.

## Follow-up chat fixtures (2026-09-19)

`tests/data/baseline_chat.json` uses the same deterministic untied tiny weights
as the F32 gate. `tools/gen_chat_baseline.py` generates greedy replies with
HF Qwen3ForCausalLM in float32/eager mode, recomputing the complete transcript
at each turn. Independent Jinja2 rendering supplies the transcript. Byte
tokens map directly to IDs, including generated bytes outside ASCII.

`tests/chat.py` checks follow-up replies with appendable, rewritten, shortened
and identical prompts, templates without generation suffixes, stop strings,
EOS and token-limit endings. A separate malformed-empty-prompt case checks
the error path. Reintroducing token-count-only cache reuse changes a reply in
the `rewrite-longer` fixture and is rejected. The original binary also crashes
on the template without a generation suffix.

`tests/data/baseline_chat_template.json` contains the actual template extracted
from the pinned Qwen3-0.6B Q8 GGUF, its hash, and Jinja2-rendered histories.
CTest's `chat-template` test checks initial/follow-up and completed-assistant
turns, with and without generation headers. This caught both double-consumed
block terminators and the first-keyword-argument parsing bug affecting Qwen's
namespace state and old reasoning removal.

Regeneration requires the same isolated HF environment documented above:
`python tools/gen_chat_baseline.py` regenerates both fixtures using the
committed template. Running tests requires neither torch nor Jinja2. These chat fixtures
prove transcript/reply behavior; the existing full-logit and NLL gates remain
the numerical runtime tests.

Windows and Linux full suites pass with required real Q8/Q4 HF fixtures;
Linux UBSan native and synthetic suites pass (real-model fixtures intentionally
skipped for the sanitizer run). The standalone renderer test passes on MSVC
and GCC. The initial GCC compile failure exposed a missing `<cmath>` include
in the renderer header; it was fixed before the passing run. Numerical kernels
and model forward paths are unchanged. Validation logs, fixture/source hashes,
commands and the rejected reuse mutant are in
[`benchmarks/chat-followup-validation-20260919.json`](benchmarks/chat-followup-validation-20260919.json).

## Wiki text location

The wikitext corpus used for corpus-level perplexity is committed to the test
data directory:

```
tests\data\wiki.test.raw
```

This is the wikitext-2-raw test split (4358 articles, ~1.28 MB). It is a plain
text dump — the `@-@` split tokens are present, matching the wikitext corpus
format expected by the path-controlled perplexity gate in `docs/ROADMAP.md`.

`llmx.exe perplexity <model.gguf> --file tests/data/wiki.test.raw --ctx-size 512`
reads UTF-8 text and scores disjoint 512-token windows. Add `--chunks 4` to
evaluate only the first four windows. See `docs/USAGE.md` for target selection;
line endings are preserved, so use identical bytes and scoring policies for
both arms of a comparison.

## CPU worker exception checkpoint (2026-09-19)

Branch `fix/cpu-worker-errors`, control `3a82284`. Evidence:
[`worker-errors-cpu-20260919.json`](benchmarks/worker-errors-cpu-20260919.json).
The archive includes exact commands, source hashes/patch, fault tests, numerical
harnesses, full logs and every timing sample. Scratch: `%TEMP%/llmx-worker-errors`.

Dispatch waits for all participants before propagating caller/worker exceptions;
partial startup joins already-created threads. Failed outputs are not rolled
back. Windows/Linux full suites with required real-model HF fixtures pass;
Windows/Linux native checks and Linux UBSan native tests pass. Allocation fault
sweeps cover construction/reconfiguration; OS thread exhaustion is not separately
forced. The original pool terminates in both initial task and final startup
regressions. Final test allocator suppresses a GCC inlining false positive;
the final Windows/Linux/UBSan fault tests were rebuilt and rerun afterwards.

| Correctness | Observed | Requirement |
|---|---:|---:|
| Long F32 max absolute logit error vs HF | 0.0001263618469 | <= 0.001 |
| Long F32 greedy continuation IDs | 32/32 | 32/32 |
| F32 full values identical to control | 5,013,888 | Exact |
| Q8 full values identical to control | 5,013,888 | Exact |
| F32 continuous/window NLL cases | 4/4 exact | Exact |
| Q8 continuous/window NLL cases | 4/4 exact | Exact |

Matched Ryzen 7 5800X Windows CPU run: same pinned Qwen3-0.6B F32/Q8 models,
215 prompt tokens + 32 forced tokens, six threads, ubatch 128, F32 KV,
mx commit `5542318e748c154b634211def405ae95da3dfaa9`. Nine measured rounds,
rotating arm order, one excluded outer warmup and each process's warmup sequence.
Model loading/tokenization are excluded. No build/test work overlaps timing.

| Mean tok/s | Control | Candidate | mx | Candidate vs control |
|---|---:|---:|---:|---:|
| Q8 prefill | 408.39 | 386.18 | 263.80 | -5.44% |
| Q8 decode | 43.88 | 43.94 | 45.81 | +0.14% |
| F32 prefill | 362.85 | 377.11 | 370.26 | +3.93% |
| F32 decode | 13.91 | 14.06 | 13.69 | +1.10% |

All candidate/control ranges overlap, but Q8 prefill loses eight of nine paired
rounds: a performance concern that must be investigated after the user's reboot
pause. Q8 decode remains 4.09% below mx. F32 means lead in this session; this is
not an equivalence test or proof of all-workload parity. No merge/push, and root
`llmx.exe` stays on the previous KV build.

Separate eight-pair synthetic `bench --size 2048 --iters 10 --threads 6 --p 64
--n 64`, alternating order with outer warmup excluded:

| Mean synthetic metric | Control | Candidate |
|---|---:|---:|
| Matmul GFLOPS | 90.12 | 91.45 |
| Step-prefill tok/s | 5,165.91 | 4,999.78 |
| Decode tok/s | 5,283.41 | 5,142.40 |

Synthetic ranges overlap and all outliers are retained. The legacy prefill
metric repeatedly calls step; it is not batched real-model prefill.

## Worker caller invocation study after reboot (2026-09-19)

Evidence: [`worker-invocation-cpu-20260919.json`](benchmarks/worker-invocation-cpu-20260919.json).
Production source remains `c072af2`; the scratch-only `fn(0)` to `job(0)` change
is not adopted. Post-reboot unchanged-binary controls and an exploratory run
precede the full gate; their separate sessions are archived, not pooled.

The stored-call candidate passes real F32 long HF logits (max absolute error
0.0001263618469 <= 0.001), exact F32/Q8 long vectors and four NLL cases each,
Windows/Linux full suites with real HF fixtures required, and Linux UBSan native
checks. MSVC allocation/task fault and grouped-kernel tests pass. NLL controls
reuse the archived c072af2 results with identical pinned weights and token IDs;
candidate NLL is freshly computed. Long coverage remains 1943 prompt tokens
plus 32 continuation steps, not full corpus or maximum context.

| Mean tok/s, nine rounds | 3a82284 | c072af2 | Stored-call candidate | mx |
|---|---:|---:|---:|---:|
| Q8 prefill | 422.70 | 406.76 | 404.32 | 262.84 |
| Q8 decode | 44.69 | 44.93 | 44.53 | 46.32 |
| F32 prefill | 354.95 | 346.77 | 343.63 | 369.07 |
| F32 decode | 13.44 | 13.43 | 13.30 | 13.33 |

Same Ryzen 7 5800X, pinned 0.6B models, six threads, ubatch 128, F32 KV,
215 prompt plus 32 forced tokens, mx 5542318e748c154b634211def405ae95da3dfaa9.
Four rotating arms, one outer warmup and each process's warmup; all samples
retained, and no build/test overlap. Candidate/control ranges overlap, but
candidate Q8 prefill loses all pairs against 3a82284. No mean beats c072af2.
The early five-round apparent win did not hold in the longer comparison.

| Separate synthetic mean, nine rounds | 3a82284 | c072af2 | Stored call |
|---|---:|---:|---:|
| Matmul GFLOPS | 118.89 | 112.27 | 105.32 |
| Step-prefill tok/s | 6270.34 | 6171.41 | 6097.43 |
| Decode tok/s | 6441.56 | 6007.77 | 6313.81 |

Synthetic ranges overlap and outliers remain in the archive. Prefill here is
repeated step rather than batched model prefill. Next investigation moves
exception_ptr construction/destruction off successful dispatches: the installed
MSVC 14.50.35717 `include/exception` confirms calls to __ExceptionPtrCreate and
__ExceptionPtrDestroy even for empty exception_ptr objects. That source fact
identifies removable work; it does not by itself quantify performance impact.
