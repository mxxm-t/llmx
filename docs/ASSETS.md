# llmx - Test assets

Where the real models and corpora used for manual verification live. These are
environment-specific paths (this dev machine); the automated suite
(`tests/run_tests.py`) generates its own synthetic fixtures; its real-model HF
checks need the pinned fixtures below and otherwise skip.

Dated validation/research sections preserve the source revision and state at
that checkpoint. Their old next steps and binary revisions are historical,
not current instructions. STATUS.md is the current branch/merge tracker; later
sections record follow-up results without pooling separate timing sessions.

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
| `Qwen\Qwen3-8B-GGUF\Qwen3-8B-Q8_0.gguf` (8.11 GiB)| Q8_0   | **Usable** - optional independent HF checks pass on Windows and Linux |
| `Qwen\Qwen2-0.5B-Instruct-GGUF\...fp16.gguf`     | FP16   | Not yet supported            |
| `lmstudio-community\...\Qwen3-30B...Q4_K_M.gguf` | Q4_K_M | Quant supported; architecture not validated (MoE is unsupported) |
| `lmstudio-community\...\Qwen3-Coder...Q4_K_M.gguf`| Q4_K_M | Quant supported; architecture not validated (MoE is unsupported) |
| `unsloth\...\Qwen3.5-4B-BF16.gguf`               | BF16   | Not yet supported            |
| `unsloth\...\mmproj-F32.gguf`                    | F32    | Multimodal projector (not a main model) |

These assets exercise both tensor-format coverage and architecture support
(`docs/ROADMAP.md` #1-3). Check both before selecting a validation model.

Use the Qwen3-8B Q8_0 model for the manual real-model checks that the suite
can't cover - e.g. the lossless correctness gate (path-controlled perplexity)
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

### Generating pinned HF references

`tools/gen_baseline.py` accepts `all`, `tokenizer`, `logits`, `perplexity` or
`f32` (default `all`). Real-model modes default to `Qwen/Qwen3-0.6B` at commit
`c1899de289a04d12100db370d81485cdf75e47ca`. Both model and tokenizer loaders
receive that revision. Logits and PPL use CPU float32 eager attention with six
threads by default; `--threads N` selects another positive count. Generated
numerical metadata records the revision, execution settings and package versions;
logit cases also record the exact input token IDs.

For review, generate into a separate directory before replacing any goldens:

```
python tools/gen_baseline.py all --output-dir reference-review
```

Another model requires `--repo`, `--revision` (a full 40-character commit SHA),
`--output-dir`, `--gguf-repo` and `--gguf-file`. The last two are associated
GGUF labels; independently verify that the GGUF derives from the intended HF
weights. A changed model or revision cannot write into the default `tests/data`
directory. Local model-directory paths are rejected because they bypass Hub
revision selection. Explicit output directories are caller-owned and can be
overwritten on subsequent runs.

`f32` generates the independent deterministic tiny model, uses one thread and
accepts only `--output-dir`. `all` includes this same tiny fixture even when
another real-model repository is selected. Tokenizer mode does not accept
`--threads`; it does no numerical inference. Use `--help` for the flag reference.
Generation needs the optional HF tooling described in the script; running the
ordinary suite still needs only Python's standard library and the built runtime.

The local 8B GGUF is not an independent HF reference. On 2026-09-20, original
`Qwen/Qwen3-8B` weights/config at revision
`b968826d9c46dd6066d109eabc6255188de91218` were verified and used on the rig
to generate separate CPU FP32 eager references. The official GGUF repository
declares that base model and its Q8 LFS digest matches the local file, but its
exact original conversion revision remains undocumented. The optional consumer
below uses predeclared bounds; reference generation alone does not establish
8B llmx correctness or broaden CI downloads.

The owned CPU-only `llmx-hf-reference` container uses six CPUs and a 40 GiB
memory cap with no extra swap. FP32 parameters occupy 30.513 GiB. Measured
cgroup accounting reached 40 GiB, including retained file cache, with 3,098
limit/reclaim events and zero OOM or OOM-kill events. All three modes exited
successfully; the container is now idle. The earlier loading estimate did not
bound retained cache accounting. Minimum sampled host available memory was
7.94 GiB. Future runs should use these measured figures.

| Independent 8B HF case | Targets | Mean NLL | Perplexity |
|---|---:|---:|---:|
| Continuous 247 tokens | 246 | 2.401654413 | 11.041428357 |
| Context 64, all windows | 243 | 3.090417353 | 21.986252097 |
| Context 64, two windows | 126 | 2.829422962 | 16.935685467 |
| Context 123, all windows | 244 | 2.670115355 | 14.441635018 |

Twenty tokenizer cases and six top-10 logit cases were also generated. This
is one generation, not a repeatability result. All three complete goldens,
commands, settings, provenance limits, hashes and memory evidence are in
[`benchmarks/hf-8b-reference-20260920.json`](benchmarks/hf-8b-reference-20260920.json).
Original weights and raw logs remain under `/zpool1/llmx-hf-reference`; outputs
are in its `goldens/qwen3-8b-b968-20260920` directory. Existing small-model
fixtures and default CI downloads are unchanged.

Validation on 2026-09-19 used cached original 0.6B weights with network access
disabled in the HF tooling. Two complete generations reproduced all numerical
fields. Compared with the committed fixtures:

| Regenerated reference check | Result |
|---|---:|
| Tokenizer cases identical | 20/20 |
| Ordered top-10 token ID lists identical | 6/6 prompts |
| Maximum rounded top-10 logit difference | 0.0001 |
| PPL token IDs identical | 247/247 |
| Continuous/windowed NLL values identical | 4/4 |
| Synthetic F32 fixture JSON | Exact |
| Windows required-HF suite components | 10/10 pass |
| Linux generator safeguard tests | 2/2 pass |

The small rounded-logit difference is an observation, not a new acceptance
bound; the old generator did not explicitly select the revision, eager
attention, no-cache execution or thread count. Existing goldens are preserved.
The Windows suite reused validated runtime 9cfe43f because no C++ source or
build setting changed. Full runtime Linux/native validation remains that
checkpoint's result; this tooling change adds the targeted Linux check above.
Commands, metadata, output hashes and comparison results are archived in
[`benchmarks/hf-reference-tools-20260919.json`](benchmarks/hf-reference-tools-20260919.json).

### Optional Qwen3-8B HF consumer

`tests/baseline_8b.py` consumes the three separate goldens in
`tests/data/qwen3-8b/`, generated from the pinned original HF revision above.
It requires the existing Q8_0 GGUF with SHA-256
`408b955510e196121c1c375201744783b5c9a43c7956d73fc78df54c66e883d6`.
It verifies the fixture hashes after normalizing checkout line endings to LF.
No HF packages or network access are needed to run it.

The canonical Linux copy is:

```
/root/.cache/llmx-models/Qwen/Qwen3-8B-GGUF/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q8_0.gguf
```

The official download at that GGUF revision was verified as 8,709,518,112
bytes with the digest above. Its temporary staged duplicate was removed; the
original Windows model is preserved. In this transfer session, copying the
existing model plus verification took 59.17 seconds, versus 486.61 seconds for
download plus verification. These are operational I/O timings, not inference
benchmarks. Prefer an existing verified copy when it is faster, and verify the
destination's identity before use.

From the repository root, select the built executable and a new output path:

```
python -X utf8 tests/baseline_8b.py --exe build/Release/llmx.exe --model C:/Users/Marko/.lmstudio/models/Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q8_0.gguf --output-dir hf-8b-review
```

For Linux use `--exe build/llmx` and the local model path. Existing output
directories are rejected. Missing/wrong models and failed commands fail the
run; there is no download or skip. Numerical commands use six workers and
ubatch 128. `report.json` records the executable hash/version, model/fixture
hashes, commands, bounds and check results. Per-command stdout/stderr and the
exact PPL excerpt remain beside it, including partial output on timeout.

The bounds were declared before the first llmx 8B comparison, prospectively
reusing the strict 0.6B Q8 budget, not calibrated from 8B results:

| Check | Frozen acceptance |
|---|---|
| Tokenizer | All 20 cases, six logit prompt ID sequences and the PPL input IDs exact |
| Six short-prompt rankings | Top-1 exact; top-5 set overlap 5/5 |
| Each top-10 output | Ten unique valid IDs; finite, nonincreasing logits; absolute magnitude <= 100; exact prompt count |
| Continuous 247-token mean NLL | Absolute HF delta <= 0.01 |
| Each of three windowed mean NLL cases | Absolute HF delta <= 0.02 |
| PPL accounting | Exact input/used/target/window/context counts; finite consistent NLL/PPL |

The context-123 case omits its singleton tail: 246 used tokens, 244 targets
and two windows. PPL runs the serial-step path; rankings exercise short batched
prefill. These checks do not bound all logits or establish full-corpus,
deep-context, lossless or performance parity. The exact original revision used
for the GGUF conversion remains undocumented. Keep failures and investigate
them without relaxing these bounds to fit observations.

The Windows and Linux consumers each pass all 37 checks with the validated
`bf122fd` runtime.
This includes 20 tokenizer cases, six prompt-ID/ranking pairs, PPL input IDs
and four numerical/counter checks. The printed NLL values and HF deltas are
identical on both platforms. The Linux ordinary suite also passes 11/11 with
`--no-perf-floor`; these results do not establish a hosted CI pass.

| Case | Windows / Linux llmx mean NLL | HF FP32 mean NLL | Absolute delta | Limit |
|---|---:|---:|---:|---:|
| Continuous | 2.400160 | 2.401654413 | 0.001494413 | 0.01 |
| Context 64, all | 3.090200 | 3.090417353 | 0.000217353 | 0.02 |
| Context 64, two | 2.829430 | 2.829422962 | 0.000007038 | 0.02 |
| Context 123, all | 2.667930 | 2.670115355 | 0.002185355 | 0.02 |

Values reflect the CLI's printed precision. The Windows executable was reused
after checking its hash and unchanged runtime source; its version records the
pre-checkpoint build (`ea1e727.dirty`). No runtime path changed in this consumer
feature, and these correctness runs are not throughput measurements.

`tests/reference_consumer.py` checks fixture tampering, token mismatch,
malformed/nonfinite/duplicate/unsorted logits, damaged PPL counters/bounds and
failed launches using the standard library. It is the eleventh ordinary suite
component. The real 8B run is optional and separate; `--require-baseline` and
`tools/fetch_test_models.py` still cover only the two pinned 0.6B models.

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

Two layers, hidden width 37, FFN width 19, head width 42, GQA 2:1 and vocabulary
257 exercise scalar tails and partial row blocks. Five prompt lengths, physical
batches 1/2/3/5/16, threads 1/4 and tied/untied output weights cover prefill;
continuous and four-token-window PPL cover sequential decode and resets.
Bounds are 2e-5 absolute for every logit and 1e-5 for mean NLL. Observed maximum
logit error for the original head-width-8 fixture was 7.2e-7 on Windows MSVC
and Linux GCC, including UBSan. Attention initially expanded head width to
10; the value-accumulation checkpoint later expanded it to 42 to cover 32-lane,
eight-lane and scalar tails. HF goldens were regenerated with unchanged bounds.
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
a shared CPU graph in mx. The reference reuses a ggml threadpool state object
and preallocated tensor outputs/workspace. The pinned Windows DLL uses OpenMP
workers and barriers; the custom ggml polling path is not active. Q8 activation
conversion is timed.
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

This is the wikitext-2-raw test split (~1.28 MB). It is a plain
text dump - the `@-@` split tokens are present, matching the wikitext corpus
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
rounds: a performance concern investigated in the post-reboot studies below.
Q8 decode remains 4.09% below mx. F32 means lead in this session; this is
not an equivalence test or proof of all-workload parity. The feature branch is
backed up on Gitea, but no main merge or GitHub publication has occurred; root
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
repeated step rather than batched model prefill. The follow-up below moved
exception_ptr construction/destruction off successful dispatches: the installed
MSVC 14.50.35717 `include/exception` confirms calls to __ExceptionPtrCreate and
__ExceptionPtrDestroy even for empty exception_ptr objects. That source fact
identifies removable work; it does not by itself quantify performance impact.

## Worker dispatch follow-up studies (2026-09-19)

Two scratch variants remain unadopted. Evidence includes source patches,
initial MSVC allocation/task-fault and grouped-kernel checks, build commands,
hashes and all samples:
[Failure-only bookkeeping](benchmarks/worker-cold-errors-cpu-20260919.json),
[shared dispatch](benchmarks/worker-shared-dispatch-cpu-20260919.json).
Neither proceeded to full HF/platform gates. Production stays at c072af2.

Each study has five measured rotating four-arm rounds plus excluded outer and
per-process warmup, matched Qwen3-0.6B Q8, 215 prompt and 32 forced tokens,
six threads, ubatch 128, F32 KV on Ryzen 7 5800X. The reference is mx
5542318e748c154b634211def405ae95da3dfaa9. No builds/tests overlap timing.
Keep the separate sessions separate; all outliers remain in the artifacts.

| Study / mean tok/s | Before errors (3a82284) | Existing fix (c072af2) | Variant | mx |
|---|---:|---:|---:|---:|
| Failure-only / Q8 prefill | 391.48 | 377.07 | 381.29 | 259.18 |
| Failure-only / Q8 decode | 41.35 | 42.66 | 41.51 | 43.59 |
| Shared dispatch / Q8 prefill | 378.79 | 349.13 | 375.20 | 260.25 |
| Shared dispatch / Q8 decode | 41.03 | 41.42 | 41.18 | 41.41 |

Failure-only bookkeeping loses every decode pair to c072af2. Shared dispatch
improves exploratory prefill but shows no decode gain. Five rounds with
variable timings do not establish sustained external parity or identify a
compiler cause. Retain the simpler existing implementation, consistent with
the user's request to avoid overcomplication. Further runtime changes require
profiling evidence; no additional dispatch variants are planned. Merge and
GitHub publication remain authorized once the project requirements pass.

## CLI thread controls checkpoint (2026-09-19)

Evidence: [cli-threads-20260919.json](benchmarks/cli-threads-20260919.json).
Source base is 05fce2c (runtime c072af2). Generate/chat now restore the resolved
CPU decode count after prefill, including auto selection and follow-up turns;
bench zero/omitted threads preserves backend auto selection. Verbose output
reports actual phase counts. Kernel arithmetic and the worker pool are unchanged.

| Validation | Windows MSVC | Linux GCC |
|---|---:|---:|
| Generation/chat thread configurations | 32 pass | 32 pass |
| Benchmark thread settings | 4 pass | 4 pass |
| HF-golden replies in thread checks | 64 match | 64 match |
| Tiny HF maximum logit error (bound 0.00002) | 0.00000070 | 0.00000070 |
| Required real Q8/Q4 HF checks | Pass | Pass |
| Native CTest | Unchanged; prior checkpoint evidence | 4/4 pass |

Three MSVC mutants reintroduce missing auto-decode restoration, ignored chat
batch count and serial bench auto; all fail the new actual-count assertions.
Windows full suite passes. Linux's first suite fails only the synthetic prefill
floor: 777.7 tok/s against 1000 with auto threads. The previous benchmark default
silently used one worker. `tests/perf.py` now explicitly uses one worker to
preserve its established workload and unchanged floors; targeted reruns pass
on both platforms (7851/8241 prefill tok/s respectively). Other Linux suite
components passed before this test-command correction and were not repeated.
No new sanitizer run or hosted CI run is claimed for this CLI-only checkpoint.

Separate five-round alternating synthetic pairs use explicit counts, size 2048,
10 iterations, 64 step-prefill/64 decode tokens and one excluded warmup pair.
All samples remain; no builds/tests overlap timing. These are not batched real
model measurements and do not establish external performance parity.

| Mean metric | Before, 1 thread | After, 1 thread | Before, 6 threads | After, 6 threads |
|---|---:|---:|---:|---:|
| Matmul GFLOPS | 41.51 | 38.82 | 108.99 | 112.75 |
| Step-prefill tok/s | 7863.96 | 7819.64 | 6094.68 | 6460.92 |
| Decode tok/s | 7461.92 | 7378.54 | 5942.72 | 6559.86 |

All respective ranges overlap; no speedup or universal non-regression is claimed.
The rebuilt matched-model comparator has identical .text SHA-256 to c072af2:
`5ed1addb00f9038153f958f38340a6d9b0acde23ca0f5a3c26fce1dcb6eb19d7`.
Thus the prior external comparisons remain the evidence for that unchanged
benchmark path; the CLI fix does not close the existing external floor.

## Build identification and documentation checkpoint (2026-09-19)

Evidence: [build-version-20260919.json](benchmarks/build-version-20260919.json).
Base is 62223a4. Builds now report release plus Git revision, with tracked-dirty
state and an unknown fallback. Build identifiers are refreshed by CMake's
build dependency and by the plain Windows build; the runtime has no Git call.
Integration fixtures use isolated repositories, including directories with
spaces. Their test commit IDs are fixture provenance, not llmx release commits.

| Integration case | Windows build.bat | Windows CMake | Linux CMake |
|---|---|---|---|
| Clean checkout, including ignored untracked files | Pass | Pass | Pass |
| Tracked change produces dirty marker | Pass | Pass | Pass |
| Commit refresh without reconfiguration | Pass | Pass | Pass |
| Archive nested inside another repo stays unknown | Pass | Pass | Not repeated |
| No-op rebuild avoids recompilation | Always compiles | Pass | Not repeated |

The separate CMake metadata script also passes with Git discovery disabled.
The actual project build reports `62223a4d81ce.dirty` before this checkpoint's
commit. A first Windows trial exposed for/f splitting an unquoted --short=12;
quoting the argument fixed it, and the complete isolated matrix then passed.
The failed trial is retained in the evidence rather than omitted.

Windows full required-HF suite passes all nine components, including the new
--version/banner check, thread controls, chat and real Q8/Q4 fixtures. Linux
build and the version check pass; unchanged inference checks retain the prior
CLI checkpoint's evidence. No new arithmetic or throughput claim is made.

README was reviewed against the roadmap and now separates current CPU support
from planned GPU/device/cluster/server/HF work. All 25 Markdown files were
reviewed, links checked, and source/comments/docs normalized to ASCII. Unicode
fixture data remains intact. Existing commit history was not rewritten; new
commit messages follow the ASCII rule. Live token delivery and reusable loading
progress are recorded as the next requested feature, not claimed implemented.

## Live generation and reusable loading progress (2026-09-19)

`benchmarks/live-generation-20260919.json` records callback/CLI changes over
8226e17, Windows/Linux full required-HF suites, final native 7/7 checks on both
platforms, and four deliberately broken variants rejected by the new tests.
The late-seek completion fix was followed by final native tests and Linux
HF-backed chat/version checks; it changes notification ordering, not tensor
bytes or inference arithmetic. Broad model validation remains scoped as above.

| Qwen3-0.6B Q8_0, 64 greedy tokens, 6 workers, stdout pipe | Before | Streaming |
|---|---:|---:|
| First visible text, median seconds from process start | 2.663 | 0.662 |
| Total elapsed, median seconds | 2.738 | 2.741 |
| CLI generation, median tok/s | 31.48 | 31.77 |

Three alternating measured pairs follow one excluded warmup pair. Both arms use
--think; every response byte is identical. The first-text measurement includes
loading/prefill and the parent's pipe-read scheduling. This demonstrates live
output, not an inference-kernel speedup or external performance-floor pass.
Legacy reasoning filters remain buffered to preserve retroactive filtering.

## CPU worker operation profile (2026-09-19)

`benchmarks/cpu-worker-profile-20260919.json` compares pre-error 3a82284 with
retained runtime 9cfe43f through the same instrumented backend wrapper. Pinned
Qwen3-0.6B Q8_0/F32, 215 HF prompt IDs plus 32 forced IDs, ubatch128 and one/six
workers are checked in three alternating process pairs. Each process uses one
uninstrumented warmup and two instrumented sequences. No competing build/test
or user inference ran during admitted samples.

| Mean instrumented milliseconds | Before, 1 worker | Current, 1 worker | Before, 6 workers | Current, 6 workers |
|---|---:|---:|---:|---:|
| Q8 prefill | 2009.89 | 2017.30 | 562.43 | 573.38 |
| Q8 decode, 32 steps | 1261.32 | 1271.89 | 764.09 | 762.44 |
| F32 prefill | 2105.09 | 2059.57 | 617.72 | 626.08 |
| F32 decode, 32 steps | 2941.53 | 2881.95 | 2443.77 | 2431.62 |

Most of the mean prefill difference lies inside Q/K/V and gate/up projections,
which include their worker scheduling and completion waits. For Q8, medians
reverse the six-worker mean ordering (569.89 vs565.93 ms). These diagnostics do
not prove a stable regression magnitude or a runtime speedup. Per-operation
means, phase medians, process ranges and every sample are in the artifact.

The instrument's initial total-time assertion caught double-counting of
attention and its nested parallel_for. The rejected sample/source is retained;
the corrected probe excludes nested operations. Final-vector FNV64 hashes
match the uninstrumented warmup, across source arms and thread counts. This
is a harness consistency witness, not an independent HF gate. The next probe
separates worker-entry delays, callback spans and completion waits.

The pinned reference DLL uses OpenMP: Release defines GGML_USE_OPENMP and the
DLL imports VCOMP140.DLL, _vcomp_fork, _vcomp_barrier and OpenMP thread functions.
Its SHA256 ac7f8ef50edc9d4d729930b669c81ddefdb56dff7e2c0121d559a02a830035d2
matches the original cpu-thread-scaling artifact. The original samples and
harness code remain intact with a dated interpretation correction appended.

Root llmx.exe is now the validated streaming build 9cfe43f after the user's
first-text delay report. The old executable is backed up in
%TEMP%/llmx-live-generation/root-before-streaming.exe; the root replacement's
hash and version verification are recorded in the profile artifact. No main
merge or GitHub publication occurred.

## CPU worker participant spans (2026-09-19)

The follow-up scratch probe independently instruments the worker pools from
3a82284 and 9cfe43f. It records publication, entry/exit for the caller and five
background workers, and an internal return boundary. Pinned Qwen3-0.6B Q8_0/F32,
the same 215 prompt plus 32 forced HF IDs, six participants, ubatch 128 and F32
KV are used throughout. Three alternating pairs interleave plain/instrumented
builds for each source arm: 24 processes, each with one warmup and one measured
sequence. No loading, tokenization, sampling or output writing is timed.

| Mean phase ms | Before plain | Retained plain | Before instrumented | Retained instrumented |
|---|---:|---:|---:|---:|
| Q8 prefill | 632.945 | 567.707 | 591.131 | 574.741 |
| Q8 decode | 803.943 | 762.698 | 779.716 | 783.178 |
| F32 prefill | 638.665 | 627.730 | 645.636 | 635.732 |
| F32 decode | 2463.925 | 2479.028 | 2458.478 | 2562.645 |

Instrumentation changes the comparison materially. Q8 decode reverses arm
ordering; F32 decode differs by +0.61% in plain builds and +4.24% instrumented.
Observed plain/instrumented differences include noise, ordering and code/cache
layout effects; they cannot be subtracted as a calibrated clock overhead.
No mx run or external-floor claim belongs to this probe.

For each dispatch, the exact decomposition selects the participant whose
callback exits last. Its entry delay, callback span and remaining completion
interval add to publication-to-return without overlapping. Mean phase sums, ms:

| Phase / arm | Last finisher entry | Last finisher callback | Final completion |
|---|---:|---:|---:|
| Q8 prefill / before | 6.002 | 572.383 | 2.836 |
| Q8 prefill / retained | 5.985 | 555.590 | 2.940 |
| Q8 decode / before | 46.909 | 689.583 | 19.104 |
| Q8 decode / retained | 47.775 | 689.672 | 19.879 |
| F32 prefill / before | 8.809 | 623.372 | 3.315 |
| F32 prefill / retained | 6.209 | 615.108 | 3.374 |
| F32 decode / before | 61.491 | 2346.495 | 21.489 |
| F32 decode / retained | 116.025 | 2385.459 | 27.653 |

Publication is sampled inside the existing mutex before unlock/notify. Entry
therefore includes wakeup and locking; callback elapsed time includes stalls.
Exit precedes worker completion bookkeeping. The return scope guard runs after
existing caller cleanup, before the function epilogue. Last-background-worker
exit to return can include remaining caller computation; the archive reports
that interval separately from last-participant completion. Summing callbacks
across participants would double-count concurrent work. Preallocated records
and distinct padded participant fields use the existing synchronous completion
ordering; no additional worker barrier or logging is introduced.

All 24 processes pass. Each model's 151,936 finite final logits are byte-identical
across all 12 processes, with matching warmup hashes. Every instrumented trace
has 673 prefill and 4512 decode dispatches, ordered timestamps, no overflow and
sequential dispatch totals fitting its phase time. A focused instrumented-current
check preserves caller error priority, completion and reuse after task failures.
These are instrumentation controls; independent HF, maximum-context and corpus
gates were not rerun. No competing agent build/test/timing ran, and every
before/after user llmx.exe check was empty; OS background work is not excluded.

The original probe did not label operations. The offline analysis below now
resolves whole decode-operation labels; members inside groups remain unresolved.
Retain the current implementation; this evidence
does not justify another rejected dispatch variant or establish a stable worker
regression. Complete process distributions, per-participant timing summaries,
decompositions, source patches, scripts, build logs, tokens, artifact hashes and
limitations are in
[`benchmarks/cpu-worker-spans-20260919.json`](benchmarks/cpu-worker-spans-20260919.json).
Full dispatch arrays and final vectors remain under `%TEMP%/llmx-worker-spans`
with hashes in the archive. Model digests reuse the pinned asset records; no
large model was rehashed. No production source or executable changed.

## Ordered prefill accumulator reductions (2026-09-20)

The four-row/three-column float prefill kernel now reduces its accumulators
explicitly in the original lane order. Both versions keep the FMA loop in
registers. MSVC control assembly stores all 12 accumulators afterward and
builds a pointer table; the candidate leaves one post-loop spill/reload.
The tradeoff is a larger function: 1,119 -> 2,931 bytes. This changes neither
the per-lane FMA sequence nor the scalar tails. Decode kernels are unchanged.

Fixed nine-round comparisons use control `ea1e727`, candidate from
`feat/prefill-ordered-reduction`, and the pinned mx reference above. Each
process warms up; an additional outer warmup round is discarded. Both models
use the same 215 prompt plus 32 forced HF token IDs, six threads, ubatch 128
and F32 KV. Local timing runs had no overlapping builds/tests/user inference.
All llmx warmup/final full-vector hashes agree across arms and rounds.

| Initial session, mean tok/s | Control | Candidate | mx | Change vs control |
|---|---:|---:|---:|---:|
| Q8 prefill | 394.772 | 412.872 | 260.000 | +4.59% |
| Q8 decode | 44.054 | 43.256 | 43.612 | -1.81% |
| F32 prefill | 357.852 | 382.921 | 364.721 | +7.01% |
| F32 decode | 13.747 | 13.735 | 13.524 | -0.09% |

Prefill wins 8/9 Q8 and 9/9 F32 pairs. The Q8 decode dip prompted a separate
fixed nine-round Q8 follow-up with identical binaries and workload:

| Follow-up session, mean tok/s | Control | Candidate | mx | Change vs control |
|---|---:|---:|---:|---:|
| Q8 prefill | 368.050 | 391.063 | 261.877 | +6.25% |
| Q8 decode | 41.840 | 42.137 | 40.513 | +0.71% |

Prefill wins 9/9 follow-up pairs; decode wins 6/9. The initial decode decrease
does not repeat, so these samples do not establish a stable decode regression.
Both sessions remain separate. The first still misses the Q8 decode mx floor;
these results do not clear the broader external gate or historical 8B gap.
The consistent prefill improvement supports retaining this feature checkpoint.

| Numerical check | Result | Acceptance |
|---|---:|---:|
| Ordered scalar-FMA oracle | 1,824 exact outputs per arm | Byte equality |
| F32 long full logits vs control | 5,013,888 exact values | Byte equality |
| Q8 long full logits vs control | 5,013,888 exact values | Byte equality |
| F32 maximum full-logit error vs HF | 0.000126362 | <= 0.001 |
| F32 prefixed continuation NLL delta vs HF | 0.000000645 | <= 0.0001 |
| Q8 prefixed continuation NLL delta vs HF | 0.007011817 | <= 0.01 |
| Swapped final lane additions mutant | Rejected | Required rejection |

The long case prefills 1,943 tokens and compares 32 continuation steps. Its
F32 greedy IDs match all 32 HF IDs. Four fresh serial-step continuous/windowed
NLL cases also equal control and pass the existing independent HF bounds;
those serial cases alone do not exercise the changed prefill kernel. This is
not full-corpus, maximum-context or independent 8B llmx coverage.

Windows native tests pass 7/7 and the required-HF Python suite passes 10/10.
Linux candidate/control native tests pass 7/7; the candidate required-HF suite
passes 10/10 with both real quantized fixtures. Linux synthetic timings are
diagnostic only (`--no-perf-floor`). The Windows perf smoke changes from
34.04 to 41.61 GFLOPS, 7,180 to 7,713 prefill tok/s and 6,960 to 7,393 decode
tok/s; these single samples do not establish causal speedups.

Both GCC arms also match the new scalar-FMA oracle. Separate Linux real-F32
checks pass 20 tokenizer cases, six logit rankings and all four NLL cases;
maximum NLL error at CLI precision is 0.000004420 <= 0.0001. Linux did not run
the long full-vector comparison, which remains the Windows result above.

Raw timing samples from both sessions, summaries, commands, harness sources,
assembly evidence, numerical results, logs and hashes are archived in
[`benchmarks/prefill-ordered-reduction-20260920.json`](benchmarks/prefill-ordered-reduction-20260920.json).
No API, model format, worker behavior or default CI fixture changes.

## Current Q8 external floor follow-up (2026-09-20)

A fresh comparison measures the unchanged validated `bf122fd` runtime against
mx-llama.cpp `5542318e748c154b634211def405ae95da3dfaa9` on the same Windows CPU.
Each Qwen3 Q8_0 model has nine measured alternating pairs after one discarded
warmup pair. All 40 processes completed successfully. Both arms use six workers,
ubatch 128, F32 KV, 215 prompt tokens and 32 forced continuation tokens. Model
loading is excluded; each process also warms up. No builds, tests, transfers
or other inference overlap the timing. These samples are not pooled with any
historical session.

| Model / phase | Current mean tok/s | mx mean tok/s | Current median tok/s | mx median tok/s | Mean vs mx | Paired wins |
|---|---:|---:|---:|---:|---:|---:|
| 0.6B prefill | 440.479 | 274.766 | 448.159 | 277.473 | +60.31% | 9/9 |
| 0.6B decode | 47.280 | 47.589 | 47.362 | 48.315 | -0.65% | 3/9 |
| 8B prefill | 30.170 | 22.124 | 30.481 | 21.994 | +36.37% | 9/9 |
| 8B decode | 4.403 | 4.440 | 4.465 | 4.501 | -0.82% | 2/9 |

Decode means and medians remain below mx for both models; the median deficits
are 1.97% for 0.6B and 0.80% for 8B. The external performance gate remains open.
The smaller gaps do not establish parity or justify selecting favorable rounds.
Full-vector warmup/final hashes are stable within each model/arm across rounds;
this supports repeatability, not independent HF correctness or cross-arm equality.
The separate HF results above retain their stated scope. Raw samples, ranges,
paired ratios, commands, source/binary identity and process exits are archived in
[`benchmarks/q8-current-floor-20260920.json`](benchmarks/q8-current-floor-20260920.json).

The separate grouped Q16 scratch candidate has preliminary synthetic validation:
MSVC and GCC ordinary/witness builds each pass 611,192 scalar/control output
comparisons, with shared-pack, separate-pack, failure and reuse witness checks.
The refined oracle checks dispatch without pre-seeding packing state and sweeps
all 63,488 finite half patterns with signed weight extremes and unaligned input.
Grouped and standalone stale-packing mutants compile and fail the expected
unseeded numerical comparisons.
Windows real-model activation witnesses also pass on both 0.6B and 8B: a
16-token prefill uses one standalone final-vocabulary pack, while decode,
one-token prefill and a 129-token prefill at ubatch 128 exercise both grouped
and standalone paths. All tested phases produce finite logits with zero float
fallback rows. These checks do not establish HF numerical cost, losslessness
or performance, and no Q16 code is adopted.
Evidence is tracked in
[`benchmarks/q16-group-validation-20260920.json`](benchmarks/q16-group-validation-20260920.json).

### Grouped Q16 rejected by the existing native accuracy contract

The unchanged native suite rejects this candidate on both platforms:

| Release native CTest | Control | Q16 candidate | Failing check |
|---|---:|---:|---|
| Windows MSVC | 7/7 pass | 6/7 pass | `backend-group` double-precision dot oracle |
| Linux GCC | 7/7 pass | 6/7 pass | `backend-group` double-precision dot oracle |

Both candidate builds succeed, but `backend-group` reports
`output differs from double-precision dot oracle`. Tests, tolerances and
candidate source remain unchanged. The earlier 611,192-output oracle passes
check the proposed integer arithmetic, including its fallbacks; they do not
establish compliance with the existing float-activation accuracy contract.
The route witnesses likewise establish activation, not numerical acceptance.

No model/HF numerical-cost or performance comparison was run for this grouped
candidate; validation stopped at the mandatory native failure. Prepared bounds
are unexecuted research criteria; they neither override the failed contract nor
authorize adopting a precision-changing implementation as lossless. Q16 remains
rejected and outside production. Build commands, source/binary hashes, raw logs
and terminal exits are in
[`benchmarks/q16-group-native-rejection-20260920.json`](benchmarks/q16-group-native-rejection-20260920.json).

## Exact Q8 reduction validation (2026-09-20)

An isolated candidate replaces the two horizontal additions at the end of
`dot_row_impl` with a shuffle/add sequence preserving the contributing
addition order. Weights, activation precision, FMA chains, other quant types
and worker behavior are unchanged. MSVC emits the intended different epilogue.
At the validation checkpoint, the candidate remained outside production while
matched timing was in progress. The completed rejection is recorded below.

| Correctness check | Control | Candidate |
|---|---:|---:|
| Windows MSVC native CTest | 7/7 | 7/7 |
| Linux GCC native CTest | 7/7 | 7/7 |
| Windows required-HF suite | 11/11 | 11/11 |
| Linux required-HF suite | 11/11 | 11/11 |
| MSVC finite scalar/control bit comparisons | Reference | 612,267 pass |
| MSVC nonfinite classification checks | Reference | 1,939 pass |
| Windows 8B HF consumer | Prior validated runtime | 37/37 |
| Windows long-prompt logits | Two identical runs | 5,013,888 bit-identical |
| Windows serial NLL cases | Two identical runs | 4/4 exactly equal |

The independent scalar oracle implements half conversion, signed-byte decoding,
four scalar FMA chains and the original rounded addition tree. It covers all
63,488 finite half patterns, unaligned inputs, signed zero, block tails, grouped
and standalone dispatch, F16C/software conversion and forced scalar fallback.
A deliberately wrong shuffle compiles and is rejected numerically. NaN payload
identity and alternate rounding modes are outside the claim.

The long comparison uses 1,943 prompt tokens and 32 forced HF continuation
tokens at ubatch 128. All 33 full vectors match the repeated control. Q8 mean
continuation NLL is 0.211345796380 versus HF 0.204333978960: absolute difference
0.007011817420 is within the unchanged 0.01 bound. The four serial excerpt
cases also pass the existing HF continuous/window bounds. This is scoped
prompt/excerpt coverage, not a full-corpus or maximum-context claim.

Windows 8B NLL differences are 0.001494412682 for the continuous excerpt and
0.000217353359 / 0.000007038207 / 0.002185355474 for the three window cases,
within the existing 0.01 / 0.02 bounds. The verified model files are reused.

The separate local perf smoke floors pass for both arms; all samples are
retained. They do not establish an optimization benefit. The fixed external
comparison was still running at that validation checkpoint, with one discarded
warmup and nine measured control/candidate/mx rounds on each Q8 model. No
partial-result verdict or main/GitHub publication was justified at that point. Source hashes, commands, raw logs,
independent reviews, numerical records and limits are archived in
[`benchmarks/q8-exact-reduction-validation-20260920.json`](benchmarks/q8-exact-reduction-validation-20260920.json).


### Exact Q8 reduction rejected after complete timing

The fixed session completed all 60 processes: one discarded warmup and nine
measured rounds per arm/model. Each process has an internal warmup; only its
second iteration contributes to these rates. All arms use six workers, ubatch
128, F32 KV and identical 215+32 token IDs. No builds/tests/model validation
competed with timing. All samples remain, without pooling historical sessions.

| Mean tok/s | Current | Candidate | mx | Candidate/current | Candidate/mx |
|---|---:|---:|---:|---:|---:|
| 0.6B prefill | 473.669 | 453.120 | 277.206 | -4.34% | +63.46% |
| 0.6B decode | 48.966 | 48.723 | 49.969 | -0.50% | -2.49% |
| 8B prefill | 29.646 | 29.936 | 21.348 | +0.98% | +40.23% |
| 8B decode | 4.579 | 4.549 | 4.610 | -0.65% | -1.31% |

| Median tok/s | Current | Candidate | mx | Candidate wins/current | Candidate wins/mx |
|---|---:|---:|---:|---:|---:|
| 0.6B prefill | 476.152 | 471.589 | 275.085 | 2/9 | 9/9 |
| 0.6B decode | 49.307 | 48.794 | 50.018 | 3/9 | 0/9 |
| 8B prefill | 29.624 | 29.760 | 21.384 | 8/9 | 9/9 |
| 8B decode | 4.582 | 4.576 | 4.613 | 3/9 | 0/9 |

The candidate establishes no decode benefit and loses every paired decode
comparison to mx on both models. It is rejected; production and the native
regression test stay unchanged. The low 0.6B prefill sample (357.111 tok/s)
remains included, with the median shown separately. Do not interpret its effect
on the mean as a universal causal slowdown or discard it to favor the candidate.

The local three-round micro smoke also retains every sample. Its matmul means
are 43.05/39.97 GFLOPS for current/candidate, synthetic prefill 7,862.67/8,030.33
tok/s and decode 7,612.00/7,749.67 tok/s. These short variable samples pass the
smoke floors but do not override the matched real-model rejection.

The current runtime still trails mx decode in this session by 2.01%/0.67% mean
on 0.6B/8B; both medians trail mx too. This does not replace the separate prior
session's numbers with pooled rates. Full samples, raw output, hashes, commands,
independent audit and rejection scope are in
[`benchmarks/q8-exact-reduction-performance-20260920.json`](benchmarks/q8-exact-reduction-performance-20260920.json).
The preceding validation artifact is preserved unchanged. Main/GitHub remains
gated on external decode performance.


### Native Q8 block scheduling rejected at codegen gate

A scratch two-block schedule retains the native float-activation arithmetic,
original HADD reduction and odd-block tail. It changes only the block loop to
call a captured accumulator lambda twice per iteration. No production source
or tests are changed.

| Independent oracle | MSVC | GCC |
|---|---:|---:|
| Finite bit comparisons | 612,267 pass | 612,267 pass |
| Nonfinite classifications | 1,939 pass | 1,939 pass |

A deliberately repeated-block mutant compiles successfully and fails the
numerical oracle. The existing oracle includes unaligned inputs, empty/odd/even
block counts, grouped and standalone dispatch, F16C/software-half conversion
and scalar fallback. NaN payload bits and alternate rounding modes are outside
its claim.

MSVC comparator assembly contains eight unconditional 32-byte accumulator
stores per two-block iteration, including on the hardware F16C path. Software
half conversion also becomes out-of-line. The formulation fails the planned
no-spill codegen gate and is rejected before model timing or HF validation.
This does not establish a measured performance regression or prove that every
possible native unrolling formulation would fail. Production remains unchanged;
the two external decode cases remain open. Source patch, assembly, compiler
commands, oracle source and raw results are archived in
[`benchmarks/q8-native-block-scheduling-rejection-20260920.json`](benchmarks/q8-native-block-scheduling-rejection-20260920.json).


### Bounded native Q8 inner-loop follow-up rejected

The separate follow-up keeps the original body in an ordinary two-trip inner
loop with local accumulators. MSVC emits two bodies without the lambda's stack
stores, but fails the second gate: useful instruction scheduling.

| F16C path per two blocks | Current | Follow-up |
|---|---:|---:|
| Executed instructions | 54 | 55 |
| FMA instructions | 8 | 8 |
| Feature tests / conditional branches | 2 / 2 | 2 / 2 |
| Termination comparisons / branches | 2 / 2 | 2 / 2 |
| Accumulator loop stack accesses | 0 | 0 |

The extra instruction reloads the F16C flag inside the paired loop. The second
body remains after the first block's final FMA and termination check. One
backedge becomes a forward exit branch, but control count is unchanged. These
are static path instruction counts, not micro-ops or measured latency. Reject
this formulation and stop this unrolling exploration; no numerical/model/HF or
performance runs were performed for the follow-up, and no production code was
changed. The earlier lambda oracle passes do not validate this new source.
Commands, source manifest, exact patch, assembly and independent review are in
[`benchmarks/q8-bounded-inner-loop-rejection-20260920.json`](benchmarks/q8-bounded-inner-loop-rejection-20260920.json).


### Archived decode worker spans labeled by operation

The saved worker-span session can be decoded without rerunning its workloads.
For its 0.6B models and six workers, each step dispatches QKV, attention,
attention output, gate/up and FFN down for each of 28 layers, then vocabulary
output. Source, matrix shapes and guards establish 141 records per token.
All 12 traces contain 4,512 ordered records for 32 decode steps: 54,144 total.
The original traces and binaries match the committed archive.

| Historical Q8 instrumented current, ms/token | Last-finisher entry | Callback | Final completion | Dispatch total |
|---|---:|---:|---:|---:|
| QKV | 0.30118 | 3.85901 | 0.12856 | 4.28875 |
| Attention | 0.27566 | 2.08291 | 0.13509 | 2.49366 |
| Attention output | 0.30408 | 1.86238 | 0.11660 | 2.28306 |
| Gate/up | 0.28816 | 5.75930 | 0.12129 | 6.16876 |
| FFN down | 0.31246 | 2.82346 | 0.11390 | 3.24982 |
| Vocabulary output | 0.01143 | 5.16518 | 0.00578 | 5.18239 |

Values are three-run means across all layers per token. Each dispatch is
partitioned using its last-finishing participant: time from publication to that
participant's entry, its callback, then time to return. The parts sum exactly
before display rounding; overlapping participant spans are not added together.
The 737 tied last-exit timestamps use the lowest participant index. Entry can
include time overlapping other workers' computation; callback intervals can
include descheduling, so neither isolates scheduler cost or kernel compute.
Dispatches total 23.66644 ms/token within 24.47431 ms/token instrumented phase
time. The remaining 0.80787 ms/token is outside the recorded dispatch spans.
These parts are descriptive and are not all necessarily recoverable overhead.

Gate/up and vocabulary output are 26.07% and 21.90% of dispatch time. All five
layer-operation control/current differences change sign across the three
pairs. Vocabulary output is slower in all three current-arm pairs, but this
instrumented comparison does not isolate its cause. Original plain/spans
ordering reversals remain relevant; no worker change is justified by this
analysis alone.

This is historical `3a82284`/`9cfe43f` data, not a fresh `bf122fd` performance
measurement. The current model/decode source is unchanged, but prefill code
and potentially compiler layout differ. Labels identify whole grouped calls,
not individual Q/K/V or gate/up projections. Original trace provenance is in
[`benchmarks/cpu-worker-spans-20260919.json`](benchmarks/cpu-worker-spans-20260919.json).
All 12 original raw outputs are now preserved as losslessly compressed records,
with SHA256 checks and standard-library decode instructions. These records,
verified mapping, analysis source, exact breakdowns and review are in
[`benchmarks/worker-decode-operation-attribution-20260920.json`](benchmarks/worker-decode-operation-attribution-20260920.json).


### Current 8B worker-span diagnostic

A fresh `bf122fd` runtime snapshot compares plain and instrumented builds on
verified Qwen3-8B Q8_0, six workers, ubatch 128, F32 KV and pinned 215+32 token
IDs. The fixed session completed all eight processes: one discarded outer
warmup pair and three alternating measured pairs. Every process also warms up
internally. All builds and fault tests finished before isolated timing; every
measured sample is retained. This is not an mx comparison or an HF gate.

| Phase time, ms | Plain mean | Plain median | Spans mean | Spans median | Mean change |
|---|---:|---:|---:|---:|---:|
| Prefill, 215 tokens | 7224.912 | 7215.189 | 7188.779 | 7186.463 | -0.50% |
| Decode, 32 tokens | 6955.863 | 6934.154 | 6887.725 | 6864.781 | -0.98% |

Instrumented/plain paired time changes are -0.676%, -1.815%, +1.016% for
prefill and -2.522%, +0.126%, -0.516% for decode. The signs vary, so lower means
do not establish a speedup caused by instrumentation. Do not pool these samples
with the earlier 0.6B traces or external-floor measurements.

All eight saved final vectors have 151,936 finite floats and match exactly as
bytes. Internal warmup equivalence uses FNV64 rather than saved byte comparison.
The instrumented fault check retains caller error priority, completion and
worker reuse. Each of four instrumented invocations has 865 prefill and 5,792
decode records, valid timestamps and no overflow. Decode has five dispatches
per layer across 36 layers plus vocabulary projection: 181 per token.

| Instrumented decode, ms/token | Entry | Callback | Completion | Dispatch | Dispatch share |
|---|---:|---:|---:|---:|---:|
| QKV | 0.31670 | 24.80917 | 0.14641 | 25.27228 | 11.85% |
| Attention | 0.34743 | 2.57661 | 0.18305 | 3.10709 | 1.46% |
| Attention output | 0.29727 | 16.59627 | 0.13086 | 17.02441 | 7.98% |
| Gate/up | 0.26121 | 99.45746 | 0.12008 | 99.83874 | 46.82% |
| FFN down | 0.32646 | 50.20885 | 0.12887 | 50.66418 | 23.76% |
| Vocabulary | 0.00836 | 17.30919 | 0.00635 | 17.32390 | 8.12% |

These are three-run means across all layers per token. The exact last-finisher
partition sums to 213.23060 ms/token dispatch within 215.24142 ms/token phase
time. Entry can overlap other workers' computation; callback intervals can
include descheduling. The 149 tied final exits select the lowest participant
index. Do not interpret this partition as scheduler-versus-kernel cost or as entirely
recoverable overhead. Members within grouped calls remain unseparated.

Matrix projections account for 98.54% of recorded decode dispatch time. Gate/up
and FFN down together account for 70.58%, directing the next investigation to
large matrix costs. This does not select an implementation or establish a
causal regression; prior kernel and worker nulls still apply. Production source,
tests and the user executable are unchanged. Source identities, commands, raw
outputs, exact-vector identity, reviews and scope are in
[`benchmarks/current-8b-worker-spans-20260920.json`](benchmarks/current-8b-worker-spans-20260920.json).


### Separate Q8 scale/payload storage screened out

A scratch layout separates each block's original two half-scale bytes from its
32 signed weight bytes, aligning payload rows to 32 bytes. It preserves weight
bits, float products, four FMA chains and the HADD reduction. Each packed matrix
requests its original byte count plus 47 bytes of alignment/load padding. The
diagnostic retains the original matrix as well; it does not change GGUF or
introduce a production backend/model interface.

| Correctness check | MSVC | GCC |
|---|---:|---:|
| Finite bit comparisons | 612,267 pass | 612,267 pass |
| Nonfinite classifications | 1,939 pass | 1,939 pass |
| Packing cases | 141 pass | 141 pass |
| Packing bytes round-tripped | 2,789,224 | 2,789,224 |
| Explicit grouped / standalone path calls | 132 / 132 | 132 / 132 |

Packing checks include all 65,536 half patterns, all signed weight bytes, odd
and empty block counts, unaligned input, aligned rows and overflow rejection.
The unchanged scalar oracle retains its original arithmetic coverage and
F16C/software-half/scalar modes. Its adapter packs one row per dot only for
correctness; that adapter is never timed. The matrix harness prepares persistent
packed rows once. A scale-bit mutant compiles and is rejected numerically.
Nonfinite payload identity, alternate rounding modes and model/HF validation
are outside the claim.

MSVC assembly preserves four FMA chains and the original reduction without
accumulator loop spills. The matrix harness uses the existing six-worker pool
and per-matrix partitions, including two matrices in one grouped dispatch.
Its free row kernels use a scratch calling convention, not the production
method; split entry's extra argument loads remain included in timing.

| Shape (input width, rows, matrices) | Original mean ms/call | Split mean ms/call | Mean change | Median change | Split wins |
|---|---:|---:|---:|---:|---:|
| 1024, 3072, 1 | 0.047025 | 0.047276 | +0.53% | -0.51% | 4/7 |
| 1024, 3072, 2 | 0.078415 | 0.078916 | +0.64% | +0.69% | 1/7 |
| 3072, 1024, 1 | 0.042013 | 0.041720 | -0.70% | -1.84% | 3/7 |
| 4096, 12288, 1 | 0.865281 | 0.844761 | -2.37% | -6.07% | 5/7 |
| 4096, 12288, 2 | 2.406585 | 2.359071 | -1.97% | -2.83% | 6/7 |
| 12288, 4096, 1 | 0.859395 | 0.850048 | -1.09% | -0.20% | 3/7 |

Each shape uses one discarded timed warmup pair and seven fixed alternating
pairs. Small cases repeat 256 calls per sample; large cases repeat 64. All 96
samples remain archived and every timed result matches its baseline bytes.
This is a short synthetic matrix screen, with small weights fitting cache;
these are not whole-model rates or an external mx comparison.

Before timing, advancement required at least 3% lower mean and median latency
and 5/7 paired wins for every large case, without small-case mean/median
regression above 3%. All three large means miss that threshold. The study is
screened out without changing the rule; no model integration is justified by
this result. Numerical validity is retained as scoped evidence.

| Packing, including allocation/initialization | Time ms | Original bytes | Packed buffer bytes |
|---|---:|---:|---:|
| Small up | 0.577 | 3,342,336 | 3,342,383 |
| Small gate/up | 1.231 | 6,684,672 | 6,684,766 |
| Small down | 0.528 | 3,342,336 | 3,342,383 |
| Large up | 8.962 | 53,477,376 | 53,477,423 |
| Large gate/up | 19.038 | 106,954,752 | 106,954,846 |
| Large down | 9.092 | 53,477,376 | 53,477,423 |

Packing timings are single observations outside matrix timing, not model-load
measurements. Both original and packed buffers remain resident; an additional
original-sized unpack vector exists transiently during round-trip validation.
Requested buffer bytes exclude vector/allocator overhead and are not peak RSS.
Production source, tests and executable remain unchanged. Exact sources,
commands, oracle/mutant logs, assembly, prospective plan and every raw sample
are archived in
[`benchmarks/q8-split-storage-screening-20260920.json`](benchmarks/q8-split-storage-screening-20260920.json).


## Exact Q8 decode SwiGLU callback fusion screening (2026-09-20)

Scratch branch `research/q8-swiglu-fusion` starts at research checkpoint
`248ed64`, with unchanged production source `bf122fd`. Source/history review
found that each Q8 gate/up worker owns the same row interval for both matrices.
It can run the original SwiGLU expression after its two projection loops and
before synchronous completion, without another dispatch or cross-worker reads.
Existing prefill SiLU parallelism did not implement this decode fusion.

The scratch helper preserves gate, up and output buffers, gate-then-up order,
original CPU dot arithmetic and `g = gate / (1 + exp(-gate)); out = g * up`.
Both timed arms use the same concrete helper; only the location of this final
loop differs. The production generic dispatcher is not timed here. No model,
Backend API, quant format or root executable changes were made.

| Synthetic numerical check, per compiler | MSVC | GCC |
|---|---:|---:|
| Matrix arm comparisons | 1,440 pass | 1,440 pass |
| Finite bit comparisons | 126,990 pass | 126,990 pass |
| Nonfinite classifications | 27,054 pass | 27,054 pass |
| Fused row witnesses in separate copy | 16,920 pass | 16,920 pass |

The reference calls actual unchanged `matmul_group` and an independently
written original expression. It checks both scratch arms and a separate atomic
row witness with one/six workers, zero/tiny/uneven/eligible partitions, changing
inputs, unaligned weights/activations and output guards. An elementwise grid
covers 441 pairs per thread count, including signed zero, subnormal/normal
extremes, infinities and NaNs. Finite values require identical bits; nonfinite
values require matching classification and infinity sign. All run in round to
nearest with FTZ/DAZ disabled; cross-libm, NaN payload and alternate rounding
claims are excluded. Dot arithmetic itself is unchanged, not independently
re-proven by this study. A separately compiled mutant adds one to the output;
it exits 1 on a numerical mismatch. MSVC/GCC builds and normal runs exit 0.
Build logs retain nonfatal vcvars diagnostics and pre-existing GCC warnings.

MSVC `/std:c++17 /O2 /arch:AVX2 /EHsc` assembly confirms original dots and
scalar `expf`, add, divide, multiply. Source snapshots and immutable identities
are archived. All correctness/build jobs are terminal before timing; Windows
process guards before/after are empty and WSL was separately inspected.

One isolated Windows process uses six workers and completes both shapes. Small
is 1024 inputs x 3072 rows per matrix, 512 calls/sample; large is 4096 x 12288,
64 calls/sample. Each shape has one discarded warmup pair and nine alternating
measured pairs, for 40 raw samples plus two shape headers. The timed interval
includes both projections, SwiGLU and complete worker return. Allocation,
fixture setup and exact gate/up/output checks are outside the clock. Every
sample is retained; only the predeclared warmup pair is excluded from statistics.

| Synthetic shape | Serial mean ms/call | Fused mean ms/call | Mean change | Serial median ms/call | Fused median ms/call | Median change | Fused wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| Small gate/up + SwiGLU | 0.089504 | 0.085051 | -4.98% | 0.089841 | 0.084535 | -5.91% | 7/9 |
| Large gate/up + SwiGLU | 2.442995 | 2.383465 | -2.44% | 2.399619 | 2.405680 | +0.25% | 7/9 |

**Screened out.** The prospectively frozen rule requires lower mean AND median
in both shapes, with at least 6/9 paired wins each, and at least one shape with
at least 2% lower mean AND median. The large median fails even though both means
improve and each shape has seven wins. The rule is not changed after observing
results. This is insufficient evidence to advance under that screen, not proof
of a model-level slowdown. No model integration or HF/mx run follows.

The diagnostic repeats synthetic weights and does not reproduce a full model's
working set. The prior 8B outside-dispatch interval includes all serial work;
it is not a measured SwiGLU cost or a promised recoverable speedup. Existing
external decode deficits and prior model correctness evidence are unchanged.
Commands, source, raw logs, fixed plan, results and independent reviews:
[`benchmarks/q8-swiglu-fusion-screening-20260920.json`](benchmarks/q8-swiglu-fusion-screening-20260920.json).


## Topology-derived CPU worker placement (2026-09-20)

Scratch branch `research/cpu-worker-placement` starts at checkpoint `dce1066`
with unchanged production source `bf122fd`. A source/history review across
44 prior artifacts found no explicit worker-affinity measurement; older SMT
claims came from thread-count sweeps. The Windows query identifies eight
physical cores, each with two logical processors, under one processor group.

One immutable MSVC `/std:c++17 /O2 /arch:AVX2 /EHsc` model harness compares
scheduler-selected placement and a single fixed mapping. Each owned child
queries physical core masks and its permitted process mask, sorts eligible
cores by lowest allowed logical processor, and chooses one processor from each
of the first six distinct cores. The observed mapping is 0/2/4/6/8/10, including
caller worker zero. Every invocation reports the same topology and `0xffff`
process mask; no preferred-core ranking or alternate mapping search is used.

The helper changes only its own calling/pool threads through the existing
synchronous pool. It saves previous thread masks and checks each setter return,
current mask, actual group/processor and worker identity. Separate snapshots
preserve placement before model execution, after execution and after restoring
original masks. The scheduler arm makes no mask changes. Normal explicit
restoration is checked; destructor cleanup on unwinding is best effort if an
OS API itself fails. No production flag, Backend API or kernel changes exist.

| Scratch helper check | Result |
|---|---|
| Fixed/scheduler placement and explicit restoration | Pass |
| Injected apply failure after caller zero / worker three | Both restore before rethrow |
| Body-exception destructor restoration / pool reuse | Pass |
| Grouped Q8 exact value comparisons | 1,818 pass (9 vectors x 202) |
| Corrupted placement report checks | 11 rejected |
| Immutable source snapshot | 17 files match production source |

Tests cover the observed single-group topology and reject a five-participant
session. Other processor-group layouts and OS cleanup failures are not executed.
The runner parser tests compose synthetic snapshots and expected verification
counts from the helper logs; these are parser fixtures, not measured model
records. Independent preflight caught and resolved lost active-placement reports
and insufficient process-name prefix matching before timing.

The fixed real-model run uses the existing verified 0.6B/8B Q8_0 models,
215 pinned prompt IDs plus 32 forced decode IDs, six workers, ubatch 128 and F32
KV. There is one discarded outer pair plus five measured pairs per model,
with model order and scheduler/fixed order reversed on odd pairs. Each process
also performs one internal warmup before its measured iteration. All 24
invocations and both internal records are retained; no samples are dropped.
All builds/checks are terminal before timing, WSL was separately inspected and
named Windows process guards are empty before/after each model invocation.

Weights, model and pool are created before placement is applied. Apply,
witnesses, final-vector checks and restoration are outside the clocks; model
reset is also outside the clock. Prefill and the subsequent 32 forced steps
are timed directly. The same executable/kernel/data/thread count is used in
both modes. Pinning also affects caller work and warmup KV/scratch first-touch,
so this measures whole-execution placement, not isolated worker wakeup cost.

| Model / phase | Scheduler mean tok/s | Fixed mean tok/s | Mean change | Scheduler median tok/s | Fixed median tok/s | Median change | Fixed wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.6B prefill | 479.071849 | 541.704220 | +13.07% | 480.110046 | 535.723941 | +11.58% | 5/5 |
| 0.6B decode | 49.314603 | 47.022156 | -4.65% | 49.455816 | 46.638018 | -5.70% | 0/5 |
| 8B prefill | 30.183415 | 40.902861 | +35.51% | 30.196669 | 40.982961 | +35.72% | 5/5 |
| 8B decode | 4.564495 | 4.649338 | +1.86% | 4.553530 | 4.644439 | +2.00% | 5/5 |

**The all-phase candidate is screened out.** The prospective rule requires
at least 0.5% higher decode mean AND median with at least 4/5 wins in BOTH
models, and no prefill mean/median regression above 3%. Small-model decode
fails all its advancement conditions despite the other improvements. The
mapping and rule are not changed after observing results. Scheduler-selected
production behavior remains unchanged.

All 24 invocations pass placement/mask restoration checks. Each compares its
full final warmup/measured vector byte-for-byte. All twelve saved vectors per
model are independently identical, with 151,936 finite floats each; the archive
stores one common full vector per model and each invocation's hash. This
covers the final position only. It is neither independent HF validation nor a
large-context/full-corpus losslessness claim. No mx comparator ran, so do not
compare these absolute rates with historical mx numbers or mark a floor passed.

The consistent prefill benefit supports reviewing prefill-only placement as a
distinct future experiment, restoring normal scheduling before decode and
including transition costs in timing. It is not implemented or validated by
this all-phase run. No claim is made about migration versus SMT contention,
reserved cores, other mappings, machines or production multi-user scheduling.
Source, commands, raw logs, exact final vectors and independent reviews:
[`benchmarks/cpu-worker-placement-20260920.json`](benchmarks/cpu-worker-placement-20260920.json).


## Prefill-only CPU placement (2026-09-20)

Scratch branch `research/cpu-prefill-placement` starts at `248f658`, using
unchanged runtime source `bf122fd`. This is a separate hypothesis and frozen
run from the rejected all-phase placement experiment above. It uses the same
queried physical-core mapping but restores each participant's original mask
before decode. No production source, public API, CLI flag or executable changed.

The Windows Ryzen 7 5800X comparison uses MSVC /O2 /arch:AVX2, Qwen3-0.6B
and Qwen3-8B Q8_0, six threads, ubatch 128, F32 KV, 215 prompt tokens and
32 forced continuation tokens. The same executable runs both modes. All 24
invocations are retained: a discarded outer pair and five measured pairs per
model, with alternating model/arm order and an internal warmup per process.
Means and medians use individual throughput samples; no outliers are removed.
The original all-phase samples are not pooled into this result.

A fresh candidate Session is constructed for each prefill. Its entire lifecycle
(topology discovery, apply, verification, restoration, report serialization and
destruction) is inside the prefill clock. Common unbound observers check worker
identities and masks before/after phases outside both clocks. Those dispatches
can influence subsequent scheduling, so this diagnostic harness is not the
production performance-floor comparator. Mask restoration does not reset cache,
boost or scheduling state, and does not prove restored decode CPU residency.

The prospective advancement rule requires both models' prefill mean and median
throughput to improve at least 5%, with at least 4/5 paired wins. Both decode
means and medians must remain at least 99.5% of control. All conditions pass:

| Model / phase | Scheduler mean tok/s | Prefill-only mean tok/s | Mean change | Scheduler median tok/s | Prefill-only median tok/s | Median change | Candidate wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.6b / pp | 467.836623 | 535.466490 | +14.46% | 476.035277 | 532.833577 | +11.93% | 5/5 |
| 0.6b / tg | 49.293600 | 49.418118 | +0.25% | 49.599945 | 49.793156 | +0.39% | 3/5 |
| 8b / pp | 29.521998 | 40.948992 | +38.71% | 29.230321 | 41.027020 | +40.36% | 5/5 |
| 8b / tg | 4.594650 | 4.623768 | +0.63% | 4.643793 | 4.629796 | -0.30% | 3/5 |

The independent lifecycle checks cover 18 comparisons of 202 Q8 outputs
(3,636 exact values), seven fresh iterations, body failure while pinned,
application failure at caller zero and worker three, and successful same-pool
reuse after each failure. Every changed mask is checked after restoration.
The report parser accepts seven actual lifecycle records and rejects fifteen
corruptions. Real-model execution checks 48 internal records and byte identity
of all 12 finite final vectors per model (151,936 logits each), including
internal warmup/measured equality. This is self-consistency, not HF proof or
full-context logit coverage. OS-level restoration failure is not injected.

The screening pass supports further integration work; it does not justify a
public affinity policy, heterogeneous-core or multi-group support, arbitrary
thread counts, thread-pool recreation, or concurrent model execution. A lean
implementation still needs phase/thread lifecycle coverage without diagnostic
observer overhead, independent HF gates and fresh matched mx timing. Both
external decode requirements remain open; historical mx rates cannot be
compared against this session to close them.

Full sources, frozen plan, identities, lifecycle/parser checks, raw timings,
placement witnesses and exact final vectors are archived in
[`benchmarks/cpu-prefill-placement-20260920.json`](benchmarks/cpu-prefill-placement-20260920.json).


## Observer-free prefill CPU placement (2026-09-20)

Branch `research/cpu-prefill-observer-free` starts at `9769833`, using unchanged
runtime source `bf122fd` and the same placement helper. It removes the shared
observer dispatches from the preceding diagnostic. No production source, CLI,
Backend API, tests or root executable changed.

The same binary runs scheduler and prefill-only modes. Scheduler constructs
no placement Session or observer. Candidate construction/topology, apply and
its verification, prefill, final active verification and checked restoration
are inside prefill timing. Decode starts immediately at the prefill clock
boundary with no intervening dispatch or reporting. Candidate JSON is emitted
after decode from stored apply/verify/restore fields. The restored destructor
performs no affinity/pool call; diagnostic formatting and inert disposal are
outside both clocks. These are pre-decode witnesses, not observations made
throughout or after decode. Scheduler mask state is not separately observed.

The prospective plan retains Windows Ryzen 7 5800X, MSVC /O2 /arch:AVX2,
Qwen3-0.6B and Qwen3-8B Q8_0, six threads, ubatch 128, F32 KV, 215 prompt
and 32 forced continuation tokens. All 24 processes and 48 internal records
are retained: one discarded outer pair and five measured pairs per model,
alternating model/arm order, plus internal warmup per process. No samples are
removed, rerun or pooled with earlier sessions. Means and medians use each
measured process's throughput, not inverse mean latency.

The unchanged prospective screen requires each model's prefill mean and median
throughput to improve at least 5%, with at least 4/5 paired wins, and both
decode means and medians to remain at least 99.5% of scheduler. All pass:

| Model / phase | Scheduler mean tok/s | Prefill-only mean tok/s | Mean change | Scheduler median tok/s | Prefill-only median tok/s | Median change | Candidate wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.6b / pp | 479.228537 | 529.632147 | +10.52% | 476.660380 | 529.184136 | +11.02% | 5/5 |
| 0.6b / tg | 49.518222 | 49.633298 | +0.23% | 49.557314 | 49.609096 | +0.10% | 3/5 |
| 8b / pp | 29.027817 | 40.985134 | +41.19% | 29.063798 | 40.774485 | +40.29% | 5/5 |
| 8b / tg | 4.581855 | 4.600768 | +0.41% | 4.568602 | 4.604604 | +0.79% | 4/5 |

Fresh validation includes the MSVC build, independent boundary review, seven
accepted archived lifecycle records, 34 rejected report corruptions and six
acceptance-rule boundary cases. The unchanged helper's prior 3,636 exact Q8
value checks and failure/reuse cases are reused evidence, not freshly run.
All 48 frozen identities are checked. Each model's 12 saved finite vectors
(151,936 final logits each) match byte-for-byte, and each process checks its
internal warmup/measured identity. This remains self-consistency rather than
independent HF or full-context proof. No OS restore failure is injected.

The result supports a distinct implementation experiment: CPU-local placement
inside existing nonempty parallel batched-matmul callbacks, preserving the
Backend interface and model layer. It has not been implemented. Per-operation
apply/restore would recur thousands of times and omit attention, serial work,
one-token/tail-one operations and final vocabulary projection. Its costs and
activation cannot inherit this phase-wide result. Restricted masks, fallback,
exceptions, pool recreation, short prompts and follow-up generation also need
coverage before adoption. No new independent HF/mx gate passes here; both
external decode requirements remain open.

The archive retains the source, frozen inputs, reused lifecycle provenance,
fresh checks, raw results, stored witnesses and common exact vectors:
[`benchmarks/cpu-prefill-observer-free-20260920.json`](benchmarks/cpu-prefill-observer-free-20260920.json).


## CPU-local batched-matmul placement (2026-09-20)

Branch `research/cpu-matmul-placement` starts at `88eed5f`. This distinct scratch
prototype keeps all placement code inside the CPU backend: the existing parallel
batched-matmul callback applies and restores its own thread mask around row work.
The Backend interface, model graph, kernels, row partitions and pool protocol
are unchanged. The production source tree, tests, build configuration and root
executable are unchanged; the prototype lives in the archived source snapshot.

Initial eligibility is Q8_0, nbatch > 1, six threads, nonempty parallel row ranges,
and homogeneous single-group topology with at least six allowed physical cores.
A fresh map queries topology for each eligible operation. It selects each core's
lowest allowed logical processor, sorts those IDs and takes the first six cores.
There is no cache or hardcoded CPU-ID map. Unsupported/query/apply cases fall
back, respecting pre-existing thread restrictions. Serial/tiny/single-column
work, other formats, attention/model-side work and final vocabulary projection
stay unbound. Every topology/query/apply/restore cost is part of model timing.

Allocation failure while building the map propagates before dispatch or affinity
changes. A task exception triggers checked restoration before propagation; if
both task and restoration fail, the cleanup failure may replace the task payload.
Normal restore failures are visible, and the destructor attempts best-effort
cleanup. Persistent OS refusal cannot be claimed as recovered: the synthetic
oracle detects the pinned state and uses explicit test-only rescue afterward.

| Validation | Result / scope |
|---|---|
| Windows lifecycle/activation oracle | 41 cases, 11,849 exact float comparisons |
| Windows native backend-group, each arm | 540 cases, 141,750 outputs against separate calls/double dots |
| Windows native scale/reduction oracles, each arm | 253,952 exact finite Q8 scale/weight cases; 1,824 ordered reductions |
| Linux GCC backend-group / backend-errors | Both pass; Windows placement is a no-op |
| Linux standalone helper | 20 no-op checks pass |
| Fresh model and witness builds | MSVC pass |
| Timing runner AST checks | Sample/iteration/order checks and eight threshold boundary cases pass |

The lifecycle oracle covers actual callback activation/caller participation,
excluded routes with no affinity calls, empty ranges, restricted caller masks,
thread counts 1 -> 6 -> 4 -> 6, query/apply fallbacks, caller/worker body errors,
one-time restore failure/retry and persistent refusal. These are synthetic
contracts, not HF proof or multi-user scheduling support.

Both unchanged `tests/perf.py` one-thread smoke checks pass. Placement is inactive
at one thread, and these single smoke invocations are not a statistical timing
comparison or evidence about the six-thread policy:

| Smoke metric | Disabled prototype | Enabled-default CLI |
|---|---:|---:|
| Matmul GFLOPS | 43.36 | 38.00 |
| Synthetic prefill tok/s | 7,804 | 7,838 |
| Synthetic decode tok/s | 7,566 | 7,532 |

A separate instrumented model executable intercepts affinity setters to verify
actual CPU/mask, original masks, paired restoration, caller participation and
no setters outside prefill. All four processes pass, with two internal iterations
each; their timing fields are ignored:

| Model / mode | Applies | Restores | Caller applies | Final vector vs prior control |
|---|---:|---:|---:|---|
| 0.6B disabled | 0 | 0 | 0 | Exact |
| 0.6B enabled | 4,704 | 4,704 | 784 | Exact |
| 8B disabled | 0 | 0 | 0 | Exact |
| 8B enabled | 6,048 | 6,048 | 1,008 | Exact |

Each enabled witness has six equally populated target-CPU counters and no
errors, active scopes after decode or setters outside prefill. Counts derive
from two iterations, two multi-token chunks, seven projections per layer,
six workers and 28/36 layers. The normal timing binary excludes instrumentation:
its measurements include permitted fallback, and do not prove that every timed
callback pinned. The exact executed witness plan is retained; a later limits-only
addition documents this distinction without changing inputs, order or thresholds.

The fixed comparison uses Windows Ryzen 7 5800X, MSVC /O2 /arch:AVX2, the same
Qwen3-0.6B/8B Q8_0 files and 247 IDs, six threads, ubatch 128, F32 KV,
215 prompt tokens and 32 forced decode steps. Both arms use the same prototype
binary with a concrete scratch switch disabled/enabled. Disabled control includes
inert callback scaffolding and is not the exact production binary. All 24
processes and 48 internal records are retained: discarded outer pair -1 and
five measured alternating pairs per model, plus internal warmup. No pooling,
retry or sample removal. Means and medians use individual throughput samples.

| Model / phase | Disabled mean tok/s | Enabled mean tok/s | Mean change | Disabled median tok/s | Enabled median tok/s | Median change | Enabled wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.6b / pp | 471.700388 | 504.994000 | +7.06% | 470.245799 | 511.307992 | +8.73% | 4/5 |
| 0.6b / tg | 49.597335 | 48.299841 | -2.62% | 49.513507 | 48.166628 | -2.72% | 1/5 |
| 8b / pp | 30.040204 | 40.211011 | +33.86% | 30.027637 | 40.269777 | +34.11% | 5/5 |
| 8b / tg | 4.502460 | 4.575066 | +1.61% | 4.495369 | 4.567486 | +1.60% | 5/5 |

The prospective rule requires both prefill means/medians >=5% faster with at
least 4/5 wins, and both decode means/medians >=99.5% of disabled control.
Prefill passes in both models; **0.6B decode fails both limits**. The candidate
is rejected despite 8B gains. No outlier is removed and no rerun is used to
rescue it. These observations do not identify the cause of the decode regression.

Every saved final vector is finite and equals the corresponding prior runtime
control (151,936 floats per vector), with internal warmup identity checked too.
All 130 frozen identities are verified. Independent active-path HF and fresh
mx gates were not run for this rejected candidate. A conditional HF plan records
that default auto-thread/F32 tests may miss the new path, while serial PPL never
activates it; that plan is not completed validation. Both external decode floor
requirements remain open. The prior observer-free phase-wide result is separate
and does not authorize this operation-local implementation or a new Backend API.

Sources, build logs, native/synthetic checks, instrumented witness, frozen plans,
raw timing and common vectors with witness aliases are archived in
[`benchmarks/cpu-matmul-placement-20260920.json`](benchmarks/cpu-matmul-placement-20260920.json).

## Synchronous CPU prefill placement callback (2026-09-20)

Branch `research/cpu-prefill-callback` starts at `731dd5c`, with production
runtime source still `bf122fd`. Scratch is
`%TEMP%/llmx-cpu-prefill-callback`. This is the one bounded callback integration
specified after the smaller operation-local candidate failed. It is rejected;
production Backend/Model/CPU, tests, build files and root executable are unchanged.

The scratch Backend adds a synchronous `run_prefill` callback. Model calls it
once around scratch-buffer preparation and all forward chunks, including the
final vocabulary projection. CPU owns topology, original worker masks and
checked cleanup; successful supported execution uses one apply and one restore
pool dispatch. Work executes once on the caller between completed dispatches.
The concrete scratch switch defaults off; a separate CLI include overlay changes
only that default for conditional validation. No production flag or API is added.

Eligibility is six workers on homogeneous single-group Windows topology with
at least six allowed physical cores. Query the process mask, sort the lowest allowed LP
per physical core, and choose the first six. Per-thread restrictions must also
permit the target. Unsupported or partially applied setup restores every changed
participant before unbound fallback. Each restore checks the original owner and
mask. Nested scopes and effective thread-count changes reject; same-count calls
are no-ops. Cleanup retries once but propagates its first failure, which may
replace a body exception. Persistent OS restore refusal is reported and only
the synthetic test performs manual rescue; no safe-reuse claim follows. Linux
is pass-through. This is not an async/GPU or concurrent-session contract.

Fresh Windows native checks pass 7/7. Callback/tiny tied/untied F32 model checks
pass 17 cases/4,626 exact values, including empty, one-token, tail-one and
follow-up behavior. Windows mask/lifecycle checks pass 27 cases/1,176 exact
values with 241 setter events audited, including partial setup, body and cleanup
failures. Setup allocation failure is injected through the topology wrapper,
not a global allocator sweep. The executed Windows test recipe retains a
nonfatal `vswhere.exe` setup diagnostic; both builds and runs exit 0, with no
compiler warning or rerun. Linux unchanged backend-group/backend-errors and
callback/no-op guards pass; platform binding is not claimed there.

Both unchanged one-thread `tests/perf.py` smoke floors pass; placement is inactive
at one thread. Control/candidate matmul is 37.86/38.46 GFLOPS, synthetic prefill
7,858/7,757 tok/s and decode 7,498/7,593 tok/s. These single smoke invocations
are not a statistical placement comparison.

All 18 separate real-model witness processes exit 0: both pinned Q8_0 models,
initial 215-token prompt, one-token suffix after a 214-token prefix, and nine-token
suffix after a 206-token prefix, with production/disabled/enabled arms. Each
process runs two internal iterations and 32 forced decode tokens after the
suffix. Enabled primary witnesses record 12 applies and 12 restores; follow-up
witnesses record 24 and 24 because both prefix and suffix have scopes. Each of
six queried LPs and the caller participates equally. Disabled counts are zero;
production is uninstrumented. No setters occur during decode. All six final
vector groups are finite and exact against fresh production; primary also matches
historical unchanged controls. Instrumented times are ignored. The output's
caller identity is an opaque C++ thread-ID hash, not a Win32 thread ID; actual
owner/caller identity is checked separately by lifecycle and mask witnesses.

Before timing, independent preflight passes 103 AST/data checks. An exact
threshold-rounding bug was corrected prospectively: decisions compare rates
with control * 1.05 or control * 0.995; percentages are reporting only. This
preserves equality at the original bounds without adding tolerance. The final
runner also checks terminal witness rows, plan and binary identities and strict
prior-workload identity/status prerequisites. A 216-file manifest freezes the
complete pre-timing source/binary/evidence set. All other compute is terminal;
Windows guards and a pre-launch WSL inspection establish isolation.

Primary timing session 67199 completes all 36 invocations with exit 0 on the
Ryzen 7 5800X, six workers, ubatch 128, F32 KV, 215 prompt plus 32 forced decode
tokens. One outer triplet per model and each process's internal first iteration
are designated warmups; the remaining five triplets per model are retained in
full. Arm order rotates and reverses prospectively. Clocks cover all Model
prefill callback/topology/apply/restore costs and immediately following decode;
model loading is outside clocks. All 36 final vectors exactly match their
fresh-production witness controls. Normal timing has no activation counters,
so it measures the enabled policy including permitted fallback.

| Model / phase | Production mean tok/s | Disabled mean tok/s | Enabled mean tok/s | Mean vs production | Mean vs disabled | Median vs disabled |
|---|---:|---:|---:|---:|---:|---:|
| 0.6b / pp | 474.695274 | 472.415155 | 548.326819 | +15.511% | +16.069% | +17.878% |
| 0.6b / tg | 48.835786 | 50.004903 | 50.342125 | +3.084% | +0.674% | +0.222% |
| 8b / pp | 30.392742 | 30.891701 | 41.452967 | +36.391% | +34.188% | +34.854% |
| 8b / tg | 4.628661 | 4.649671 | 4.622986 | -0.123% | -0.574% | +0.004% |

The frozen screen requires enabled primary prefill mean and median >=5% higher
than BOTH controls, with at least 4/5 paired wins against each; decode mean and
median must be >=99.5% of BOTH controls. Both prefill cases pass with 5/5 wins.
The single failed rule is 8B decode mean against disabled prototype:
4.622985950 versus 4.649671131 tok/s, or -0.573915446%, beyond -0.5%.
Its median is +0.003835% against disabled; production comparison is -0.122610%
mean and +0.706814% median. These do not override the failed required rule.

Reject this integration and stop placement adoption. No outlier removal,
threshold change, repeat screen, wider API or map search follows. Conditional
one/nine-token follow-up timing is not run because primary fails; numerical
witnesses do not establish short-prompt performance. The unchanged-bound HF
validation plan is retained but not run for this rejected candidate. No fresh
HF/mx acceptance, main merge or GitHub publication is claimed. Prior external
decode requirements remain open, and no regression cause is established.

Independent timing/archive audit and the full 25-file Markdown review are
complete. Rebuild sources, recipes, logs, all timing results, six model/workload
full-vector groups with witness/timing aliases and independent reviews are preserved in
[`benchmarks/cpu-prefill-callback-20260920.json`](benchmarks/cpu-prefill-callback-20260920.json).


### Current decode caller attribution and native sampling capability (2026-09-20)

This scratch diagnostic starts at research checkpoint `4be2872`, with runtime
source unchanged from `bf122fd`. It follows the stopped placement studies.
The previous 2.010823 ms/token residual mixed model work, backend preparation
and observation; it was not an isolated serial cost or a recoverable saving.

The three fresh arms are unchanged production, archived worker-span observers,
and those same worker observers plus caller attribution. Legacy probes cover
measured prefill and decode in both probed arms. Added caller markers cover
only measured decode; warmup is untraced. No arithmetic, weight layout, worker
protocol, affinity, public API or production source is changed.

The fixed plan runs 12 processes: one discarded outer triplet, then three
rotated measured triplets. Each process has an internal warmup and measured
iteration. The pinned Qwen3-8B Q8_0 model uses six workers, ubatch 128, F32 KV,
215 prompt IDs and 32 forced decode IDs. All samples are retained. Model loading
and output-file writes are outside clocks. The original
`logits = model.step(...)` assignment remains inside the decode clock.

Session 21493 terminates with exit 0 and all 12 invocations passing. All 93
frozen source/binary/plan identities recheck unchanged. All 24 warmup/measured
full vectors are finite, 151,936 floats each, and byte-identical to the pinned
unchanged-runtime vector. Each of four attributed traces records exactly 32
step envelopes, 5,792 outer primitive/dispatch pairs and 37,152 caller events.
Per-step interleaved marker/primitive order, one observer per dispatch,
publication after probe setup and all nesting/worker boundaries are checked.

| 8B phase, mean elapsed ms | Plain | Legacy spans | Added caller attribution |
|---|---:|---:|---:|
| Prefill, 215 tokens | 6957.927567 | 6839.615500 | 6915.812233 |
| Decode, 32 tokens | 6884.288267 | 6903.010467 | 7031.672633 |

Decode elapsed changes are +0.271955% for legacy spans versus plain,
+2.140880% for added attribution versus plain and +1.863856% for added
attribution versus legacy spans. The last comparison's paired changes are
+4.207462%, +2.092283% and -0.676599%; all are retained. These are elapsed-time
ratios in one diagnostic session, not performance gains or an mx comparison.

| Added attribution, observed ms/token | Mean | Median |
|---|---:|---:|
| Gross dispatch intervals | 217.602291 | 216.097138 |
| Model-side work | 1.972685 | 1.897919 |
| Backend caller preparation/return | 0.080984 | 0.079369 |
| Harness remainder | 0.015842 | 0.003122 |
| Explicit caller-observer brackets | 0.067954 | 0.067009 |

The integer partition is exact before rounding. Outer timer fringe is reported
separately (mean 0.000014 ms/token); inner partition plus fringe equals the full
outer decode interval. Dispatch includes worker observation. Model-side means
outside selected dispatch-bearing primitive envelopes and includes serial
backend norms/RoPE. It is not exclusive C++ model-layer instruction time.
Within that model-side total, SwiGLU is 1.157311 ms/token mean and 1.140116
median; it is not an additional row to sum into total elapsed time.

Timestamp/store brackets cannot enclose all observer effects, including code
layout, register pressure, cache changes and scheduling. One prospective
10,000 clock-pair then 10,000 empty-marker calibration reports mean 18.15 ns
per clock pair and 36.32 ns outer elapsed per marker. Median adjacent clock
and bracket durations are zero at this timer's resolution. Calibration is
retained, never subtracted as a correction to production timing. In this
session total perturbation/variation is comparable to or larger than residual
regions. No recoverable production cost is proved: stop this outside-kernel
optimization direction without tuning the probe, repeating the screen or
selecting an implementation. No external HF/mx gate is closed.

Validation includes successful MSVC builds, unchanged backend-group checks
(540 cases, 141,750 outputs, 253,952 exact finite Q8 scale/weight cases and
1,824 ordered reductions), unchanged backend-error checks, and six active-probe
exception/reuse dispatches. Pure accounting checks cover three exact partitions,
14 malformed interval rejections and overflow; independent Python checks cover
250 interval trials and five adversarial malformed traces. Final-vector equality
is a diagnostic regression check, not an independent HF correctness claim.

Separately, installed Visual Studio CPU sampling is usable without elevating
this owned test process. The first collector call reports an error despite
exit code 0 and creates no session; the minimal Base-config call explicitly
reports Running then Stopped and produces a recording. A Python smoke yields
6,605 target samples. A native optimized smoke with local PDBs yields 6,221:
3,393 resolve to `sample_integer_mix`, 2,812 to `sample_rotate_mix`, and 16 are
other target-attributed locations. This proves function sampling capability,
not model hotspots, complete call stacks, memory-stall causes or model
elapsed-time costs. No model was sampled in this earlier capability study. Only owned-target sampled events are archived;
raw ETL/system dumps stay local. xperf text uses the local Windows ANSI code
page 1257; the extraction script decodes that format before retaining ASCII
owned-target rows.

Independent terminal audit rehashes all 93 identities, verifies all 12 raw
process results and 24 vectors, and reproduces all 148,608 caller events with
an independent interval sweep. Every category, region and timer fringe agrees.

Production source, tests, build/CI and root executable remain unchanged. Main
and GitHub remain unchanged. Rebuild sources, commands, raw diagnostics, shared
full-vector payload, target-only sampled events and reviews are preserved in
[`benchmarks/cpu-decode-caller-cost-20260920.json`](benchmarks/cpu-decode-caller-cost-20260920.json).

## Native CPU decode sampling (2026-09-20)

This diagnostic uses unchanged runtime source with optimized local PDBs and
one fixed executable for plain and sampled runs. Qwen3-8B Q8_0 uses six CPU
threads, ubatch 128, F32 KV, 215 prompt IDs and 32 forced decode IDs. Eight
invocations comprise one prospectively discarded pair and three measured
pairs; each process has its own internal warmup. Both arms use the same
10-second ready and 5-second done holds outside model clocks. Collector
attachment/stop and actual target holds must meet the frozen bounds.

The first session completes the discarded pair and saves a zero-loss trace,
then fails parsing unquoted commas inside xperf C++ symbol fields. Separate
recovery files preserve all 41 original identities and the complete failure.
The corrected reader requires an unambiguous split between two module symbols,
retains unknown locations and agrees with an independent raw-line parser.
A separately frozen 51-entry recovery manifest precedes exactly the remaining
six invocations; no model run is repeated. Continuation session 21434 exits 0.

| Mean elapsed ms | Plain | Sampled | Sampled/plain change |
|---|---:|---:|---:|
| Prefill, 215 tokens | 7060.177800 | 7022.348833 | -0.535808% |
| Decode, 32 tokens | 6970.996900 | 7034.000533 | +0.903797% |

Paired decode changes are +2.915900%, -0.611112% and +0.453332%; all samples
are retained. These differences combine profiler/handshake/state effects and
ordinary variation, not an isolated profiler-overhead measurement.

All 16 finite full vectors (151,936 floats each) equal the pinned historical
runtime vector byte-for-byte. Independent audit verifies both manifests,
records, phase clocks, hold bounds, owned collector Running/Stopped states,
and every target row against the raw export. Trace counts are 39,209 for the
discarded preflight, then 40,115, 39,733 and 39,571 measured. Every trace reports
zero lost events/buffers and resolves all application samples to named local
symbols. Unknown and system-module samples remain in the denominator.

Of 119,419 measured target samples, 116,184 (97.291051%) resolve to
`backend::CpuBackend::dot_row_impl`. These are sampled executing locations,
including gate bookkeeping around decode, not exclusive operation times,
complete call stacks or evidence of DRAM/cache stalls. No runtime optimization
or external HF/mx acceptance follows from this diagnostic. Production source,
tests, build/CI, main and GitHub remain unchanged.

Offline instruction mapping uses the exact executable, PDB and MAP, plus
recorded per-process image bases. All 345 instructions and 1,476 bytes in the
dot function match executable bytes; PE exception ranges bound the function
and exclude padding. Every retained dot sample maps to an instruction start.
The archive keeps per-run instruction/mnemonic/region counts and caller/other
thread groups, with the discarded pair separate and non-dot samples retained.
A high count at an instruction is not its exclusive latency: for example,
`mov rax,r11` accounts for 18,694 measured samples. Sampling delivery and
instruction overlap prevent interpreting this as time spent on that move.
No new optimization is selected and earlier rejected studies remain closed.

Full plan, failed parser and recovery, commands, exact source, byte-verified
assembly, target-only samples, vectors and reviews are preserved in
[`benchmarks/cpu-native-decode-sampling-20260920.json`](benchmarks/cpu-native-decode-sampling-20260920.json).
Raw system-containing traces/exports and binaries stay local with recorded
hashes; compressed target samples preserve their complete original bytes.
