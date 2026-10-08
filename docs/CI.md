# Continuous integration

`.github/workflows/ci.yml` runs on pull requests, pushes to `main`, pushes to `gate/<name>` branches, and manual dispatch.
A stack of branches about to merge is pushed as `gate/<name>` so these checks run before the merge, and that branch is deleted after it.
Branches are merged locally, so no pull request reaches the workflow, and the gates a branch passes before it merges are run locally and recorded in `docs/STATUS.md`.
A pull request's newer push cancels its older run; every other run has a concurrency group of its own, so each push to `main` keeps its run as the record of that merge and each push to a `gate/<name>` branch keeps its run as the check before it.
Builds use `--parallel 4`, the hosted runners' core count.
It contains seven independent checks:

| Check | Coverage |
|---|---|
| CPU (ubuntu-24.04) | GCC, CMake Release, synthetic tests and benchmark smoke; first, every file in `tests/` and `tools/` byte-compiled and every Python tool's `--help` run, so a tool broken by a change to the CLI or the tests fails here rather than when it is next run by hand |
| CPU (windows-2022) | MSVC, CMake Release and `build.bat`, synthetic tests and benchmark smoke on both binaries, and the `build.bat` binary reporting the same version as the CMake one |
| CPU (macos-15-intel) | Apple Clang, CMake Release, synthetic tests and benchmark smoke |
| CPU (Linux UBSan) | GCC 14 undefined-behavior checks, including mixed-tensor float alignment; GCC 14 is the compiler of `docker/Dockerfile`'s image, beside the runner's default GCC 13 in the ubuntu job |
| CPU (Linux TSan) | GCC 14 with ThreadSanitizer, address randomization off as its memory layout needs on the runner's kernel: the `http` test, whose listener is closed from another thread while its accept loop runs, and `server-passes-cpu`, the scheduler's passes in flight over CPU stages; the two binaries run without CTest, whose timeouts are set for uninstrumented builds, and a report fails the job |
| Vulkan backend (build, Linux) | The backend and every shader compiled with `-DLLMX_HAS_BACKEND_VULKAN=ON`, the headers and `glslc` from the LunarG repository, pinned there since the distribution's compiler is older than the shader extensions the kernels use and has not been retried; CTest with `backend-vulkan`, `mxfp4-vulkan`, `vulkan-lifetime` and `vulkan-quantization` skipping without a device, while `vulkan-buffer`, whose fake device supplies every Vulkan call, needs no loader and runs; then the Python suite with `--require-tools` on the CPU path (`--device cpu`) of the Vulkan-enabled binary, the build Linux GPU users make; then the dead-code check of the linked binaries, `tests/dead_code.py --linked`, on a build of its own at -O0 |
| HF reference (CPU) | Linux build plus the six pinned gate models, the four Qwen3-0.6B files and the two Qwen3.5-0.8B files: tokenizer, logits, continuous/chunked PPL, and the real-model server checks on the Qwen3-0.6B Q8_0 (limits, uncapped requests pausing, a prompt paused while prefilling, prefix reuse over a conversation, clients leaving, a chat turn, the tokenize routes against `llmx tokenize` and `llmx detokenize` on the tokenizer golden's texts and `messages` against the chat template goldens); the suite's HF chat and thread replies run here as in every CPU job. Then the `baseline` component again with f32 caches, `llmx-split-check` on the Q8_0 over two CPU backends, and many users through the server on the Q8_0. No CTest: the Ubuntu job runs it on the same build |

Every job that runs the Python suite starts it with the dead-code and stale-docs checks, the `dead-code` and `docs` components, and every build makes an unused function of one translation unit, or an unused local, a compile error (AGENTS.md, Dead code and stale docs).
The Vulkan job's extra step builds every target again at -O0 with every inline function emitted and links each executable with `--gc-sections`, so the functions only tests keep show; it adds about two minutes to the job, and it runs even when a step before it failed, since it builds a tree of its own.
A finding fails its job unless `tests/data/known_findings.txt` lists it, with how often it occurs and a reason, and a listed finding that no longer occurs, or occurs another number of times, fails it too.
Every CTest a CPU build registers runs in every job's "Backend tests" step but the HF job's, which builds what the Ubuntu job builds, so the KV cache, placement, HTTP layer, server UTF-8 repair, prefill-scope, log-probabilities, server-resume, server-passes, job-threads and server-passes-cpu checks are covered on all three platforms and under UBSan.
`http` and `server-passes-cpu` also run under ThreadSanitizer in the TSan job, which builds only those two.
The Vulkan-only CTests run in the Vulkan job alone, where `backend-vulkan`, `mxfp4-vulkan`, `vulkan-lifetime` and `vulkan-quantization` skip without a device.
That job then runs the Python suite on the CPU through the Vulkan-enabled binary, where the `cli` component finds no device and checks that a Vulkan device is refused rather than run on the CPU.
The Python suite's `server` component starts `llmx serve` on the synthetic dense and MoE models in every CPU job, the MoE model's prompts alone against four at a time and the dense model's tokenize routes against `llmx tokenize` and `llmx detokenize` on text beyond ASCII, special tokens' text and an empty text, and on the real Q8_0 fixture in the HF job.
What no hosted job establishes is device behaviour: the Vulkan job proves the tree compiles, and the kernel comparisons, the HF gate on the device and the matched floors are run on the Radeon VII and the Linux machine's MI50s by hand and recorded in `docs/STATUS.md`.
No self-hosted runner is planned: the Linux machine runs other work, and the gates on the cards stay by hand on both platforms.
On that machine, `docker/Dockerfile` carries the driver and the compiler and takes the cards through `/dev/dri`, and inside it the whole CTest suite, `backend-vulkan` included, passes on an MI50.

The layer split is covered on the CPU.
The `placement` CTest, in every job that runs CTest, splits a model over two and three CPU backends, among them a pipelined prompt of five chunks, which reuses pass slots and handoff buffers, its rollback when a backend on the last stage fails, and a history recomputed by class as a paused request's resume recomputes it.
It also runs passes of different sequences in flight through the pass API over two, three and four CPU backends at S, S + 1 and 2S pass slots for S stages, in random order and with a failed pass, against each sequence run alone.
The Python suite's `split` component, in every job that runs the suite, runs `llmx-split-check` 90 times against one CPU backend, at ubatch 1, 3 and 16 with f16 and f32 caches: on the tiny F32 models, tied and untied, over two CPU backends, the tiny MoE model over two and three, the three tiny MXFP4 models over two, the tiny qwen35 models of Hv = Hk and Hv = 3 Hk over two and four, a layer a stage, whose first and third stages keep only a recurrent state, and a synthetic Q8_0 model over two, its decode steps also recomputed by class, the Q8_0 model's histories long enough that each run also recomputes from a fork at a block, which the component requires; the tool keeps no checkpoint for a model that keeps a state, so it recomputes that model without a fork.
Its `decode-probe` component runs `llmx-decode-probe` on the tiny F32 model and on a model of four tokens: the greedy path and a step off it, malformed fixtures refused by entry, and a vocabulary smaller than the five ids the tool lists.
The `device-reference` component runs `llmx-model-logits` to capture full logits for the shared numerical criterion (Device versus CPU numerical checks, below).
CMake builds all three tools in every configuration with tests (the default), and those jobs pass `--require-tools`, so a tool missing beside the executable fails the job rather than skipping.
The Windows job's second run, on the `build.bat` binary, which has no tools beside it, leaves the flag off.
The HF job also runs `llmx-split-check <Q8_0> <excerpt> cpu cpu,cpu 8 64` on the real Qwen3-0.6B Q8_0 over the 247-token perplexity excerpt: every position through the prompt path, the prefill in four 64-token chunks pipelined over two CPU stages, 8 greedy steps, the prompt and steps recomputed by class, whole and from a fork at a block, a decoding sequence beside a fresh prompt, and passes in flight at two, three and four slots, bit for bit against one backend.
Splits over GPUs are run by hand on the Radeon VII and the MI50s.

The HF job ends with `tools/server_mix_check.py` on the Qwen3-0.6B Q8_0 on the CPU, `--requests 8 --cli 2`, prompts cut from `tests/data/wiki.test.raw`: eight requests of 120 to 12000 characters each give their ids alone, then all at once, then skewed, the long prompts landing while others decode and every fourth client leaving mid-stream, and the first two give the same text through `generate --temp 0`.
It is the only hosted check of long prompts landing on a real model's server while others decode, with clients leaving mid-stream among them.

The original four jobs passed in the [initial hosted run](https://github.com/mxxm-t/llmx/actions/runs/35440893448) at `ec74308`.
Local Windows MSVC and WSL Linux GCC CMake builds of `ec74308` also passed the suite with both HF fixtures it then had required.
Workflow lint and negative checks for corrupt downloads, missing fixtures and invalid throughput passed at that commit.

The CPU backend currently uses x86 intrinsics, and CMake enables AVX2/FMA/F16C.
The kernels use them with no runtime check, so that binary does not run on older CPUs.
Intel macOS is intentional; ARM and a portable scalar build are not covered.

The CPU jobs' limit is 15 minutes, 40 on macOS, 35 on Windows and 25 under the sanitizers. The Windows job ran 23 min 46 s and 24 min 5 s on main a0922bb62 and 07c21c6ee (runs 37778710464 and 37743681404), and was then ended at its 25 minutes twice on a branch that added no time to it, every step passing, the second time after its last step (runs 37778848475 and 37785016981); its build takes 6 minutes, the native tests 9 and the suite 6 and a half, and no test or check changes.
The macOS allowance is based on the retained [timeout comparison](#macos-intel-timeout-2026-10-03); individual test timeouts and checks are unchanged.
The TSan job's limit is 25 minutes: its thread tests took 591 s and 744 s on main 8278c6b7f and 5472fcf91 (runs 37841297903 and 37828241038) after a build of about a minute, and were then ended at 15 minutes after 818 s on a branch that adds no work to either test, every other job of the run passing (run 863 on its gate branch at 1dc3d1c5f); no test or check changes.
The UBSan job's limit is 25 minutes: its instrumented build and tests ran 14 min 36 s on main 235375a9, and the disk tier's tests then ended it at 15 minutes in the suite's last components, every step before passing, twice in a row (runs 37227214118 and 37235049535); the margin is the instrumented build's, and no test or check changes.
The Vulkan job's limit is 25 minutes too: it ran 14 min 17 s on main 235375a9 and 14 min 46 s on a2f32b8f, and the disk tier's read-back tests then ended it at 15 minutes in its linked dead-code step, every step before passing (run 37246868334); no test or check changes.
The Vulkan backend has its build job above; a job that runs its kernels
needs a device, which hosted runners do not have. Actual GPU numerical and
performance results require the corresponding hardware; compilation alone
does not establish backend correctness.

HF acquisition adds four offline native targets: Hub manifest/hash,
Hub cache/multi-stream assembly, curl child-process/response handling, and GGUF
shards. It also runs a sharded synthetic F32 consumer against committed HF
logits/NLL in the Python suite. These require no internet access or real curl
installation; the native transport test supplies a fake child executable.
Live downloads still require curl 8.4+ and separate network integration checks.

The HF job runs `tools/fetch_test_models.py`, a standard-library downloader that takes the entries of `tests/data/fixtures.json` marked `gate`, by their revisions and SHA-256 digests.
Downloads are verified before entering the HF snapshot cache.
The HF job caches those snapshots between runs, with a key `tools/fetch_test_models.py --key` derives from the pins of the models it downloads alone, so a change to a check or a bound in `tests/baseline.py`, or a model pinned ahead of the gate, keeps the cache.
The cache is restored before the fetch and, when the key missed, saved right after it, once every file is verified, so a run that fails later still keeps its downloads.
Restored files are still SHA-256 checked on every run.
Cold or invalid cache entries are downloaded from the pinned revision.

The downloader makes at most five attempts for HTTP 408/429/500/502/503/504
and transient connection/read failures. Backoff is 30/60/120/240 seconds;
valid `Retry-After` and HF `RateLimit` reset headers can extend each wait to
at most 300 seconds. A longer server wait fails with a clear diagnostic,
rather than retrying early. Permanent HTTP failures, local file errors and
SHA-256 mismatches fail immediately. Failed attempts remove temporary files;
only a complete verified download replaces the destination.

Every job except the Vulkan build also runs `python -X utf8 tests/fetch_models.py`: seventeen offline tests cover throttling, reset headers, retry exhaustion, interrupted reads, cache reuse/replacement, checksum rejection and permanent failures, and which pins a run downloads and hashes into the cache key.
These tests use tiny independent bytes and simulated network responses; they do not download models or replace the real HF reference checks.
At `b266650` all fifteen passed on Linux, including a real HTTP response parser test for premature EOF, and the downloader before it reproduced the single-request 429 failure.

`--require-baseline` makes
missing fixtures fatal, preventing a green numerical job made entirely of
skips. Test execution does not install torch, transformers or HF packages.
The ordinary CPU jobs can skip real-model checks because their fixtures are
absent; the separate HF job supplies that coverage.
The qwen35 pretokenizer is held to HF in every job all the same, with no model: the `tokenizer` component writes a file from `tests/data/baseline_tokenizer_qwen35.json` and requires HF's ids for its 37 texts and one id for each of the 7 control tokens only the GGUF files add.

For device coverage, `run_tests.py --require-device-types Q8_0,Q4_K` makes a selected backend's refusal of any named weight type fail its component instead of being skipped.
Names are the case-sensitive storage names in `tests/spec_decode.py`, comma separated; empty or unknown names are usage errors.
The shared `device_lacks_kernel` helper owns this policy, for both early model refusals and backend matrix/embedding refusals.
Without this option, unsupported device types still skip; types not named keep that behavior too.
Naming a type does not prove a component exercised it, require missing model files, change architecture or cache skips, or change native CTest's driverless skips.
Use it with `--device`, the components that exercise the required types and `--require-baseline` when real fixtures are required.

The Python suite also checks reference-generator argument safeguards and that the requested commit, float32 dtype and eager attention reach the HF loader.
The dtype calibration tests check that matrix inputs are rounded then widened, routers and norms are excluded, hooks are removed on failure, the torch/transformers pins are enforced, and an existing output is preserved. They use test doubles; reproducing the frozen budget uses the separate pinned HF environment described in `docs/PRECISION.md`.
It checks that the qwen35 tokenizer golden keeps every merge its texts reach and gives the added tokens the files' types, that a tokenizer file with another SHA-256 is refused, and that the committed golden holds the generator's texts, commit and digests.
Every qwen35 reference passes one check of its environment, whose doubles refuse another torch, transformers or tokenizers version and the packages HF would run in place of its torch functions.
For the tiny qwen35 goldens it checks that the generator loads in float32 with eager attention from local files and refuses unused keys other than `mtp.*` and `model.visual.*` and any missing key; and it maps the committed Qwen3.5-4B `dt_bias` values onto the 4B GGUF's through the tiled order `tests/qwen35.py` owns for the writer and both references, bit for bit.
For the layered qwen35 reference it checks the tool's arguments, that it runs in that environment check, its refusal of checkpoint keys it cannot map one to one onto the model, its float32 step count, that the committed Qwen3.5-0.8B whole-model and Qwen3.6-35B-A3B four-layer records show every input's logits and every parameter equal to HF's full forward at the same depth, and that the 9B, 27B and 35B-A3B goldens hold the generator's texts, windows, versions and routed experts implementation; `reference-consumer` runs their consumer over simulated passing, refused and failing outputs.
These use standard-library test doubles; CI does not generate new HF goldens or download larger models.
The `qwen35` component holds the tiny qwen35 and qwen35moe files to HF on the CPU in every job, and reports SKIP, not PASS, only on a device whose backend refuses the architecture's ops, or when `--cache-type` asks for a type other than f32.
The ordinary suite has 28 components, including `dead-code` and `docs`, the source and Markdown checks, `arch-boundary`, which holds the runtime to naming no architecture or tensor, `qwen35`, `raw-blocks`, the spec decoders' checks, `server-load`, the load tool's self-test, `tensor-split`, the tensor split against its HF fixtures, and `reference-consumer` rejection tests for 8B fixture tampering, malformed or out-of-bound numerical output, wrong model identity and failed launches, and a passing 8B run over simulated outputs that must have 41 checks with each NLL case scored in both modes.
These tests use small committed JSON fixtures and doubles, without 8B inference.
Default real-model downloads are the four pinned 0.6B GGUFs, Q8_0, Q4_0, Q5_K_M and Q4_K_M, and the two pinned Qwen3.5-0.8B files, Q8_0 and Q4_K_M.
Six more models are pinned there ahead of their tensor types, with `gate` false, and no job downloads them yet.
When their types join the gate, the three marked `hosted` (UD-Q8_K_XL, IQ4_XS and Q2_K, 1.51 GB) join the HF job's downloads and its cache key, and the other three (BF16, IQ4_NL and Q3_K_S) are checked by hand after `tools/fetch_test_models.py --all`.
The job downloads and requires every gate model, so `tests/baseline.py` refuses a gate model not marked `hosted`, and one of those three joins the gate only with a change that lets the job leave it out.
Three qwen35 files are pinned too, each naming the `qwen35` family: the two Qwen3.5-0.8B files marked `hosted` (Q8_0 and Q4_K_M, 1.34 GB), and the Qwen3.5-4B Q4_K_M, checked by hand with its bounds.
Both 0.8B files are in the gate with their bounds in `tests/baseline_qwen35.py`, so the HF job downloads and requires them and checks them at 512-token windows in both of its passes.
The Q4_K_M's own quantization moves HF's top-1 on two of its eight rankings, so it is held first to its committed file-exact goldens at the Q8_0 file-exact bounds and then to its own quality bounds against its model, which the user approved for its SHA-256 alone (`docs/STATUS-2026-09.md`).
The HF job's limit is 60 minutes, raised from 30 by the branch that added the two files: the job ran 19 minutes before them, and the suite's `baseline` component took 10.5 minutes with f16 caches and 8.6 with f32 over them on six loaded CPUs of the Linux machine, so the two passes add about 19 minutes to a job a hosted runner may run slower.
The `baseline` component checks the qwen35 files of the gate, reports one as skipped when it is absent and as one skip line on a device whose backend lacks the architecture's ops, and leaves the files outside the gate to be run by hand.
Every entry names its family; the key changed once when that field was added, and changes again with the two Qwen3.5-0.8B files joining the gate, so the HF job's next run downloads the six gate files afresh.

The separate 8B consumer requires an existing model and a new output directory:

```
python -X utf8 tests/baseline_8b.py --exe build/llmx --model path/to/Qwen3-8B-Q8_0.gguf --output-dir hf-8b-review
```

Use `--exe build/Release/llmx.exe` for MSVC.
It verifies model/fixture hashes, records executable identity, commands and failures in `report.json`, saves raw output beside it, and never downloads or skips a missing model.
Frozen bounds require exact token IDs, top-1 agreement and top-5 overlap 5/5, counting the boundary swaps AGENTS.md's HF baseline rule allows, with absolute NLL deltas <= 0.01 continuous and <= 0.02 windowed.
Each NLL case is scored twice, in batched passes and with `--per-token`, as in `tests/baseline.py`, so a run has 41 checks.
Top-10 output must be finite, sorted, unique-ID and within absolute magnitude 100; `tests/baseline.py` holds the 0.6B outputs to the same validators at its own bounds.
This optional run is outside default CI; see [ASSETS](ASSETS.md#optional-qwen3-8b-hf-consumer) for reference provenance, the verified Linux cache path and the limits of short-excerpt coverage.

Every job checks that `--version` and the usage banner agree with the release version, then runs the small F32 HF fixture without downloads.
Its deterministic weights are generated locally; committed HF float32 logits/NLL cover tied and untied embeddings, matrix tails, multiple physical batches and thread counts.
The same fixture checks that `logits --file` prints what the inline prompt does, and holds the rows `logits --last` and `--then-ids` print to its HF bound at their positions.
Those rows are printed once each, and the ones printed over passes of five tokens or after a head continued by `--then-ids` are the bytes the same positions print in one pass.
`bench --model` with `--seqs 2` runs on it and reports the token counts of its prompt and batched decode tests: two sequences need two cache blocks where the model's context fills one.
The `cli` component checks that the builds without the Vulkan backend, and the Vulkan job's build, which finds no device, refuse a Vulkan device rather than run on the CPU, and that `info` lists a synthetic model's architecture, layer count and tensors.
It also checks that the CLI's usage errors exit with status 2 and the command's page on stderr, before any model file is opened.
It shows every help page without a model, and checks that each command takes every flag its page lists, refuses a second value for it in any of its spellings, takes a switch given twice, and refuses the flags its page does not.
The UBSan job makes misaligned in-memory tensors a test failure.
The three CPU jobs and the UBSan job also run CTest for JSON syntax/Unicode/numeric boundaries and string escaping, GGUF structure, custom alignment and loading failures, Qwen model configuration and required tensor/storage layouts, grouped kernels, the qwen35 layers' CPU ops and the qwen35 module's refusals, plan and state rules (`arch-qwen35`), worker failures, chat rendering against transformers' own renderer, sampling, log-probabilities, KV storage, the automatic worker count's reading of the CPU quota and the available host memory's reading of the memory limits, plus the Python HF follow-up fixtures and CLI thread-control checks, which hold the automatic count to the one the test reads itself.
The Python suite's `roundtrip` component in these jobs checks the Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K and MXFP4 decoders bit for bit against the spec decoders of `tests/spec_decode.py`, each on raw blocks that reach every scale, min, high bit and nibble, Q8_0's and Q4_0's under the negative scales quantize never writes, so those readers are covered without a real model.
Its `raw-blocks` component checks the spec decoders themselves, those of F16, BF16, IQ4_NL, IQ4_XS, MXFP4, Q2_K and Q3_K included, against values computed from each format's fields and the sign each zero takes.
Every job that runs the suite with `--require-tools` installs numpy 2.4.3 first, so raw-blocks also holds the spec decoders' numpy form, which makes the file-exact references and the MXFP4 fixture, to the pure form, and fails rather than skips without numpy.
The suite itself needs only the standard library, and without numpy and that flag raw-blocks skips those checks and says so.
The combined five-job workflow first ran on published runtime `08351b0`.
[Run 35512421834](https://github.com/mxxm-t/llmx/actions/runs/35512421834)
passed ordinary Ubuntu and required HF, but exposed three portability issues:
Windows short-path spelling in a test, UBSan scalar-tail contraction in an
exact oracle, and macOS subnormal-number conversion in the JSON parser.
Repair `851d375` resolves those issues. Its [complete hosted run](https://github.com/mxxm-t/llmx/actions/runs/35512954742)
passes all five jobs, including the full required-HF suite and the actual macOS
and Windows runners. The initial four-job result above covers the older tree.
The historical reconciled release `08351b0` passes all ten native tests, all fifteen downloader cases
and all eleven required-HF Python components on Windows MSVC and WSL GCC 13.3.

Hosted jobs pass `--no-perf-floor`: `bench` must still run and report finite,
positive throughput, but the workstation-specific floors are disabled.
The synthetic regression test pins one CPU worker to preserve the workload
used to establish its local floors; automatic counts are tested separately.
Hosted timings are diagnostic. The performance gate against mx-llama.cpp
still requires matched hardware, model, quant and workload; see ROADMAP #8.

The first hosted run with the Vulkan job's suite and the HF job's added steps, at `73f4f78`, took 15 min 13 s for the HF job, 3 min 54 s for macOS, 2 min 58 s for Windows, 2 min 6 s for the Vulkan build and its suite, 1 min 54 s for UBSan and 1 min 5 s for Linux.
The HF job kept its 30-minute limit then, about twice its time; its limit is now 60 minutes, for the two Qwen3.5-0.8B files (above).

To reproduce locally:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --output-on-failure
python -X utf8 -m compileall -q tests tools
python -X utf8 tests/fetch_models.py
python -X utf8 tests/run_tests.py --exe build/llmx --no-perf-floor --require-tools
python -X utf8 tools/fetch_test_models.py
python -X utf8 tests/run_tests.py --exe build/llmx --no-perf-floor --require-tools --require-baseline
python -X utf8 tests/run_tests.py --exe build/llmx --require-baseline --cache-type f32 --only baseline
python -X utf8 tests/dead_code.py --linked build-linked
build/llmx-split-check <Q8_0 fixture> <excerpt> cpu cpu,cpu 8 64
python -X utf8 tools/server_mix_check.py --exe build/llmx --model <Q8_0 fixture> --text tests/data/wiki.test.raw --requests 8 --cli 2
```

The Ubuntu job also runs each `tools/*.py` with `--help`.
The Vulkan job runs the suite with `--device cpu` on its build, `-DLLMX_HAS_BACKEND_VULKAN=ON`, and the linked dead-code check, which needs Linux, GCC, GNU ld, binutils, CMake, the Vulkan headers and glslc.
The Q8_0 fixture is `baseline.find_fixture(baseline.BASELINE_MODELS[0])`, under `~/.cache/huggingface/hub`, and the excerpt is the `text` of `tests/data/baseline_perplexity.json` written to a file as it is, which the HF job's "Q8_0 fixture path and perplexity excerpt" step does.

For MSVC, use `--exe build/Release/llmx.exe` and `build/Release/llmx-split-check.exe`.
Omitting `--no-perf-floor` preserves the existing local timing floors.
`build.bat` still builds the root `llmx.exe`, which remains the Windows test runner's default.

Actions are pinned to commit SHAs, checkout credentials are not persisted,
and workflow permissions are read-only. Jobs run on hosted machines; this
workflow does not expose the GPU machines to pull-request jobs.

After the hosted runs succeed, the stable check names above can be
required for `main`. Branch protection is a separate repository setting;
adding this workflow does not enable it automatically.

CTest also covers synchronous text delivery before the next model step and split UTF-8 bytes, plus loader progress, truncated reads and consumer exceptions.
`cli-output` observes flushing through the actual CLI emitter with a controlled stream buffer, without wall-clock timing assertions.
It also runs the CLI's number readers and token id lists over every malformed form they refuse.
Python chat checks keep progress on stderr and compare follow-up replies to the HF goldens with progress enabled and disabled. Quiet chat must print exactly one dtype record on stderr, and server health must report the dtype requested by the test command, including suite overrides.

The reported [run at d6e00e0](https://github.com/mxxm-t/llmx/actions/runs/35498190148)
failed while fetching the Q8_0 fixture with HTTP 429, before HF tests ran.
Retry/cache handling addresses that download failure; persistent service
throttling can still exhaust the bounded retry policy and fail the job. The
[updated hosted run](https://github.com/mxxm-t/llmx/actions/runs/35510681421)
passed all four jobs, including fixture downloads and required HF checks.

The independent build-identification release at `9511a4a` also passed its
[four-job hosted run](https://github.com/mxxm-t/llmx/actions/runs/35511296680).
These public releases are merged into the published runtime without removing
its native, HF or UBSan checks. Every job except the Vulkan build runs the
offline downloader checks.

## Reading hosted failures

Use the run's `head_sha` to identify the tested commit, then list its jobs and read the failed job's log:

```
gh api repos/mxxm-t/llmx/actions/runs/RUN/jobs
gh api --allow-escape-sequences repos/mxxm-t/llmx/actions/jobs/JOB/logs
```

Capture the second command's output and strip terminal control sequences before displaying excerpts. Without that flag, `gh` can refuse a readable log because it contains escape sequences; that refusal does not establish a GitHub permission failure. Check annotations can identify a failed step but may contain only its exit code, so they do not replace the log.

Use an existing authenticated `gh` session, or supply an existing Git credential as `GH_TOKEN` only in the child process environment. Do not print the credential, put it in command arguments or save it in the evidence. A successful read of one run does not establish that a different commit passed.

## macOS Intel timeout (2026-10-03)

The [half-weight job at 325479c9](https://github.com/mxxm-t/llmx/actions/runs/37112245232/job/111172258256) was cancelled during the Python suite. Its check annotation says "The job has exceeded the maximum execution time of 25m0s"; the build and all 38 native tests had passed, with no observed test assertion failure. The [preceding main job at d9c37b07](https://github.com/mxxm-t/llmx/actions/runs/37109015725/job/111163128251) passed in 14m15s.

| Stage | Previous main | Half-weight head |
|---|---:|---:|
| Build | 3m56s | 15m08s |
| Native checks | 3m21s, 37 passed | 6m28s, 38 passed |
| Python suite | 6m13s, passed | Cancelled after 2m42s |

Both jobs used macos-15 image 20260824.0482.1 and AppleClang 17.0.0.17000013 on different hosted runners. The added half-weight native test took 5.32s. Common native tests rose from 200.69s to 382.12s, while dead-code rose from 18.8s to 35.4s and docs from 15.0s to 22.4s. Broad runner slowness is plausible, but no machine telemetry establishes its cause or excludes additional build cost.

The finite 40-minute macOS limit allows for the observed 21m36s build/native work plus about 11m50s if the earlier complete Python suite takes the common-native 1.90x time. Setup and cleanup bring that estimate to about 35 minutes, leaving roughly five minutes of margin. This is a scheduling estimate, not a measured completed run. Windows remains at 20 minutes and ordinary Linux CPU at 15; no test, assertion, individual test timeout or retry policy changes. The cancelled attempt remains incomplete. The timeout correction at [59d7e14b](https://github.com/mxxm-t/llmx/commit/59d7e14b75364dd11cd520d21dfcc49ea801484e) passed all seven jobs in [CI 37115070878](https://github.com/mxxm-t/llmx/actions/runs/37115070878) and is published on both main refs; that run validates the workflow correction, not the cancelled half-weight integration.

The full logs, check annotations and all native durations remain in `.tmp-half-release-20261003/ci/macos-cancelled-111172258256/`. The annotation JSON has SHA-256 `3a05e83d61338ae9fd58df8f5e814dcb5ffa89488591351d500909b901ba0794`; the full stage/native comparison JSON has SHA-256 `ab8a4014b35eec691b8cc03279fb973a844bca88fc7cc38d9e77df590fd72c06`.

## Exact reduction test compilation

The `backend-group` test disables implicit floating-point contraction on
GCC/Clang, including AppleClang. Its explicit SIMD FMA and `std::fma` calls
remain fused, and nothing else is.
Its bitwise oracle for the prompt's three-, two- and one-column float dots takes every product, the tail's included, as an explicit FMA, so under this option a kernel matches it only if the kernel's tail is written as FMAs too.
That is the rule the kernels keep (`docs/src/backends-cpu.md`): a tail left as `v += a * b` is fused or not at the compiler's choice, and GCC 14 under UBSan fused it in one column kernel and not in another, which gave a prompt row different bits by its place in the batch.
The option applies only to this test target, not the CLI or performance tools; the CLI HF checks separately exercise the normal runtime flags.

The reference-generator path test checks an absolute path and filesystem
identity, so Windows short names such as `RUNNER~1` and their long names are
accepted as aliases. A real short-name regression checks this when available.
JSON parsing accepts a fully consumed finite nonzero result with absolute magnitude at or below
minimum normal even when the standard library sets failbit for underflow.
Malformed input, overflow and underflow to zero still fail; boundary cases run
in the native JSON test on every platform.

## Prefill scope coverage

CTest includes `prefill-scope` on all platforms. It checks synchronous caller
ownership, stable pool participants, error draining/reuse, and allocation plus
microbatch boundaries with generated tied/untied F32 fixtures. Windows also
runs `prefill-placement`: active real topology when available, real fallback
otherwise, and synthetic topology/failure cases even on small hosted runners.
These checks do not require a real model or establish performance.

CMake registers the native tests; `ctest --test-dir build -N` lists the configured build's set. Windows additionally registers `prefill-placement`, and a build with `LLMX_HAS_BACKEND_VULKAN=ON` adds `backend-vulkan`, `vulkan-buffer`, `vulkan-lifetime` and `vulkan-quantization`.
The buffer test runs on a fake device that supplies every Vulkan call, so it needs no loader; the lifetime test opens a device and intercepts transfers for ownership checks, including failed padded-cache invalidation.
It also substitutes five kernel creation failures to check cleanup/retry, and runs two real diagnostic-query cases for failed creation and idle-before-destruction.
Query cases skip on a device without diagnostic timestamps.
None replaces the kernel or HF gate.
At placement release `3c5d4b9`, Windows 12/12 and Linux 11/11 passed locally.
That release also passed all five hosted jobs in [run 35516912422](https://github.com/mxxm-t/llmx/actions/runs/35516912422), including the new native targets on Windows, macOS Intel, Linux and Linux UBSan, plus the required HF job.
The earlier `851d375` pass predates these added tests.

## Device versus CPU numerical checks

`tools/check_device.py` applies the quantization plan's shared CPU/device criterion, implemented beside `top5_overlap` in `tests/common.py`.
It supplements the independent HF gate; it does not measure quantization loss against original weights.
Choose an existing weight type on the same model or architecture as the control before measuring the candidate:

```
python tools/check_device.py --exe build/llmx --model candidate.gguf --control existing-type.gguf --device vulkan:0 --output device-check
```

The new output directory retains model, binary and fixture hashes, command lines, exit codes, complete raw logits and `report.json`, including failures.
The control and candidate must be different model files of the same architecture; each is run on CPU and the selected device with the same cache type and ubatch.
Their metadata includes the actual storage types, so the control's existing-type provenance can be checked against its pinned file.
The control runs first and supplies the maximum full-logit gap separately for the batched and per-token paths; the candidate may not exceed those limits.
The control measures those limits: its ranking, NLL and greedy disagreements are retained in the report, but are not extra candidate acceptance requirements. Both captures must still be complete, finite and internally valid.
Every acceptance check below applies to the candidate; calibration is an internal caller role, not a user-selectable waiver.
Both paths compare every position of the pinned HF excerpt: top-1 agrees wherever the CPU top-two gap exceeds the existing 0.1 margin, top-5 agrees with that margin, and mean NLL differs by at most 0.01.
NLL scores each next token, leaving the final row unscored; that final row still participates in ranking and full-logit checks.
Both models also take 64 argmax steps after one prefill, without stopping at EOS; CPU and device IDs must agree before the first CPU near-tie.
After that point their histories may differ, so later logits are retained and checked for finite values and correct argmax IDs, but not compared numerically across histories.

`--cache-type` chooses f16 (default) or f32 for both sides; `--ubatch` defaults to 512.
`--n-cpu-moe` offloads that many expert layers beside the device, or all with -1; CPU reference calls always use 0.
The capture uses six CPU threads and accepts a device list through the normal placement owner.
There are no unsupported-device skips and no automatic downloads.
Naming a file as the control is not independent evidence of its prior HF validation; the gate record must retain that evidence too.

The caller uses `llmx-model-logits` beside the CLI, built by CMake with tests.
Its arguments are `MODEL IDS OUTPUT_PREFIX DEVICE CACHE UBATCH CPU_EXPERTS [SHARES]`; the optional final comma-separated whole-number proportions pin the layer split through the loader, for example `1,1` for two equal shares. Omitting them keeps automatic placement.
It writes native binary32 rows in token-ID order for the batched excerpt, the per-token excerpt and the greedy continuation, plus metadata on stdout.
It calls the existing loader and model APIs; it implements no loading, placement or numerical policy.
The `device-reference` suite component checks the shared criterion with planted numerical faults and malformed rows, then checks the capture against the independent tiny F32 HF fixture, Unicode paths, malformed IDs, cache/ubatch/share refusals, explicit-share capture equivalence, incomplete captures, a full comparison and a damaged logit outside the top ten.
A missing tool fails with `--require-tools`, and otherwise the component reports SKIP after its criterion tests.
These tests use the CPU and tiny models; passing them does not establish large-model or GPU correctness.

## CPU MXFP4 coverage

Hosted CPU jobs run the native `mxfp4` test and the Python `mxfp4` component against committed independent HF fixtures, including its original-F32 activation control. `roundtrip` checks raw MXFP4 decoding against the spec decoder, `split` adds dense tied/untied and routed fixtures, and `server` checks their first-token HF log-probabilities and concurrent/streamed consistency. Real MXFP4 files are manual checks; hosted jobs do not download them. Vulkan builds add native F32/BF16 and direct F16 row/tile checks and can run the independent HF fixtures when the device has the required optional properties. The MXFP4 component also checks committed 39/40/41- and 63/64/65-token prompt references across microbatches, so its dense cases cross both measured narrow prompt-tile thresholds. Unsupported devices still refuse at load and skip explicitly unless the type is required. STATUS records the released dtype qualification, its hosted run and the separate device evidence; hosted runners provide no GPU execution gate.

### Dtype dispatch witnesses

The raw-logit tool now writes `dtype` and `matrix_paths` beside its captures. Each of the batched, decode and greedy phases records one activation-path list per participating backend, consumed after that phase completes. The current capture test uses those lists through `tests/common.py`'s `hf_bounds`; F32-only work keeps the original logit and NLL bounds even when the selected policy is narrower. Missing evidence, an unknown path, block-int8, or a path incompatible with the selected policy is refused. The real-model acceptance bounds are unchanged. The capture reader accepts old metadata for retained historical comparisons, but such metadata cannot select a calibrated tiny-fixture bound.

The native dtype test checks evidence against its distinguishing F32/BF16/block-int16 arithmetic and over K-quant widths 3840, 4096 and 4352 in dense, grouped and routed calls. The Vulkan test checks all six matrix operations against a unit-weight row with inputs whose float, 8-bit and 16-bit results differ. Each measurement clears the evidence before the next. The shared tiny-fixture consumers now use the completed CLI witness, including dense, MoE, MXFP4, sharded and Qwen3.5 checks. Server MXFP4 checks compare the served values with the witnessed CLI row before applying its HF allowance; health alone supplies no execution witness. The legacy MXFP4 bound is removed. The dtype implementation is released; model/depth qualification and measured performance tradeoffs are recorded in STATUS.
