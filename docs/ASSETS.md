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
a greedy-generation or numerical-correctness test. Both arms use six threads,
ubatch 128, F32 KV and causal attention; the reference disables GPU offload and
flash attention. Each process runs one warmup sequence and one measured
sequence with cleared KV state. The driver alternates arm order, rejects
failed runs and invalid timings, and saves raw stdout/stderr, hashes and
mean/median/range summaries. The default is eight process pairs.

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

### Attention query scheduling: diagnostic candidate

A scratch change schedules independent head/query pairs cyclically across
workers, with one score row per worker. Per-query arithmetic is unchanged
in source. Three interleaved rounds against `c4436fd` show:

| Phase | Current mean tok/s | Candidate | mx |
|---|---:|---:|---:|
| Prefill | 388.08 | 399.48 | 395.48 |
| Decode | 14.57 | 14.46 | 14.83 |

Prefill control/candidate ranges are disjoint; decode ranges overlap. This
is a lead for the next full gate, not adopted runtime code or an external
parity claim. Only finite outputs/top-token/printed-sum smoke checks ran.
The exact patch and raw samples are in
[`benchmarks/attention-scheduling-diagnostic-20260919.json`](benchmarks/attention-scheduling-diagnostic-20260919.json).
Scratch files are under `%TEMP%/llmx-attention-balance`.

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
