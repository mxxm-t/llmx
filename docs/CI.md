# Continuous integration

`.github/workflows/ci.yml` runs on pull requests, pushes to `main`, pushes to `gate/<name>` branches, and manual dispatch.
A stack of branches about to merge is pushed as `gate/<name>` so these checks run before the merge, and that branch is deleted after it.
Branches are merged locally, so no pull request reaches the workflow, and the gates a branch passes before it merges are run locally and recorded in `docs/STATUS.md`.
A pull request's newer push cancels its older run; every other run has a concurrency group of its own, so each push to `main` keeps its run as the record of that merge and each push to a `gate/<name>` branch keeps its run as the check before it.
Builds use `--parallel 4`, the hosted runners' core count.
It contains six independent checks:

| Check | Coverage |
|---|---|
| CPU (ubuntu-24.04) | GCC, CMake Release, synthetic tests and benchmark smoke; first, every file in `tests/` and `tools/` byte-compiled and every Python tool's `--help` run, so a tool broken by a change to the CLI or the tests fails here rather than when it is next run by hand |
| CPU (windows-2022) | MSVC, CMake Release and `build.bat`, synthetic tests and benchmark smoke on both binaries, and the `build.bat` binary reporting the same version as the CMake one |
| CPU (macos-15-intel) | Apple Clang, CMake Release, synthetic tests and benchmark smoke |
| CPU (Linux UBSan) | GCC undefined-behavior checks, including mixed-tensor float alignment |
| Vulkan backend (build, Linux) | The backend and every shader compiled with `-DLLMX_HAS_BACKEND_VULKAN=ON`, the headers and `glslc` from the LunarG repository, pinned there since the distribution's compiler is older than the shader extensions the kernels use and has not been retried; CTest with `backend-vulkan` and `vulkan-lifetime` skipping without a driver, while `vulkan-buffer`, whose fake device supplies every Vulkan call, needs no loader and runs; then the Python suite with `--require-tools` on the CPU path (`--device cpu`) of the Vulkan-enabled binary, the build Linux GPU users make |
| HF reference (CPU) | Linux build plus all four pinned real models: tokenizer, logits, continuous/chunked PPL, and the real-model server checks on the Q8_0 (limits, uncapped requests pausing, a prompt paused while prefilling, prefix reuse over a conversation, clients leaving, a chat turn); the suite's HF chat and thread replies run here as in every CPU job. Then the `baseline` component again with f32 caches, `llmx-split-check` on the Q8_0 over two CPU backends, and many users through the server on the Q8_0. No CTest: the Ubuntu job runs it on the same build |

Every CTest a CPU build registers runs in every job's "Backend tests" step but the HF job's, which builds what the Ubuntu job builds, so the KV cache, placement, HTTP layer, server UTF-8 repair and prefill-scope checks are covered on all three platforms and under UBSan.
The three Vulkan-only CTests run in the Vulkan job alone, where `backend-vulkan` and `vulkan-lifetime` skip without a device.
That job then runs the Python suite on the CPU through the Vulkan-enabled binary, where the `cli` component finds no device and checks that a Vulkan device is refused rather than run on the CPU.
The Python suite's `server` component starts `llmx serve` on the synthetic dense and MoE models in every CPU job, the MoE model's prompts alone against four at a time, and on the real Q8_0 fixture in the HF job.
What no hosted job establishes is device behaviour: the Vulkan job proves the tree compiles, and the kernel comparisons, the HF gate on the device and the matched floors are run on the Radeon VII and the Linux machine's MI50s by hand and recorded in `docs/STATUS.md`.
No self-hosted runner is planned: the Linux machine runs other work, and the gates on the cards stay by hand on both platforms.
On that machine, `docker/Dockerfile` carries the driver and the compiler and takes the cards through `/dev/dri`, and inside it the whole CTest suite, `backend-vulkan` included, passes on an MI50.

The layer split is covered on the CPU.
The `placement` CTest, in every job, splits a model over two and three CPU backends, among them a pipelined prompt of five chunks, which reuses pass slots and handoff buffers, and its rollback when a backend on the last stage fails.
The Python suite's `split` component, in every job that runs the suite, runs `llmx-split-check` on the tiny F32 and MoE models over two and three CPU backends against one, with f16 and f32 caches.
CMake builds that tool in every configuration with tests (the default), and those jobs pass `--require-tools`, so a tool missing beside the executable fails the job rather than skipping.
The Windows job's second run, on the `build.bat` binary, which has no tools beside it, leaves the flag off.
The HF job also runs `llmx-split-check <Q8_0> <excerpt> cpu cpu,cpu 8 64` on the real Q8_0 over the 247-token perplexity excerpt: every position through the prompt path, the prefill in four 64-token chunks pipelined over two CPU stages, 8 greedy steps and a decoding sequence beside a fresh prompt, bit for bit against one backend.
Splits over GPUs are run by hand on the Radeon VII and the MI50s.

The HF job ends with `tools/server_mix_check.py` on the Q8_0 on the CPU, `--requests 8 --cli 2`, prompts cut from `tests/data/wiki.test.raw`: eight requests of 120 to 12000 characters each give their ids alone, then all at once, then skewed, the long prompts landing while others decode and every fourth client leaving mid-stream, and the first two give the same text through `generate --temp 0`.
It is the only hosted check of long prompts landing while others decode and of clients leaving a real model's server.

The original four jobs passed in the [initial hosted run](https://github.com/mxxm-t/llmx/actions/runs/35440893448) at `ec74308`.
Local Windows MSVC and WSL Linux GCC CMake builds of `ec74308` also passed the suite with both HF fixtures it then had required.
Workflow lint and negative checks for corrupt downloads, missing fixtures and invalid throughput passed at that commit.

The CPU backend currently uses x86 intrinsics, and CMake enables AVX2/FMA/F16C.
Runtime checks inside some kernels do not make that binary safe on older CPUs.
Intel macOS is intentional; ARM and a portable scalar build are not covered.
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

The HF job runs `tools/fetch_test_models.py`, a standard-library downloader using the revisions and SHA-256 digests in `tests/data/fixtures.json`.
Downloads are verified before entering the HF snapshot cache.
The HF job caches those snapshots between runs, with a key derived from `tests/data/fixtures.json` alone, which holds only the pinned model specs, so a change to a check or a bound in `tests/baseline.py` keeps the cache.
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

Every job except the Vulkan build also runs `python -X utf8 tests/fetch_models.py`: fifteen offline tests cover throttling, reset headers, retry exhaustion, interrupted reads, cache reuse/replacement, checksum rejection and permanent failures.
These tests use tiny independent bytes and simulated network responses; they do not download models or replace the real HF reference checks.
At `b266650` all fifteen passed on Linux, including a real HTTP response parser test for premature EOF, and the downloader before it reproduced the single-request 429 failure.

`--require-baseline` makes
missing fixtures fatal, preventing a green numerical job made entirely of
skips. Test execution does not install torch, transformers or HF packages.
The ordinary CPU jobs can skip real-model checks because their fixtures are
absent; the separate HF job supplies that coverage.
The qwen35 pretokenizer is held to HF in every job all the same, with no model: the `tokenizer` component writes a file from `tests/data/baseline_tokenizer_qwen35.json` and requires HF's ids for its 37 texts and one id for each of the 7 control tokens only the GGUF files add.

The Python suite also checks reference-generator argument safeguards and that the requested commit, float32 dtype and eager attention reach the HF loader.
It checks that the qwen35 tokenizer golden keeps every merge its texts reach and gives the added tokens the files' types, that a tokenizer file with another SHA-256 is refused, and that the committed golden holds the generator's texts, commit and digests.
These use standard-library test doubles; CI does not generate new HF goldens or download larger models.
The ordinary suite now has 17 components, including `server-load`, the load tool's self-test, and `reference-consumer` rejection tests for 8B fixture tampering, malformed or out-of-bound numerical output, wrong model identity and failed launches, and a passing 8B run over simulated outputs that must have 41 checks with each NLL case scored in both modes.
These tests use small committed JSON fixtures and doubles, without 8B inference.
Default real-model downloads are the four pinned 0.6B GGUFs: Q8_0, Q4_0, Q5_K_M and Q4_K_M.

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
At `dacf18c` local Windows and Linux runs each passed the 37 checks the consumer had before the per-token half, with identical printed NLLs and HF deltas.
The Linux ordinary suite passed its 11 components with `--no-perf-floor` at that commit.
These local results do not establish hosted 8B coverage; the optional consumer is not run by the workflow.

Every job checks that `--version` and the usage banner agree with the release version, then runs the small F32 HF fixture without downloads.
Its deterministic weights are generated locally; committed HF float32 logits/NLL cover tied and untied embeddings, matrix tails, multiple physical batches and thread counts.
The same fixture checks that `logits --file` prints what the inline prompt does, and holds the rows `logits --last` and `--then-ids` print to its HF bound at their positions.
Those rows are printed once each, and the ones printed over passes of five tokens or after a head continued by `--then-ids` are the bytes the same positions print in one pass.
`bench --model` with `--seqs 2` runs on it and reports the token counts of its prompt and batched decode tests: two sequences need two cache blocks where the model's context fills one.
The `cli` component checks that the builds without the Vulkan backend, and the Vulkan job's build, which finds no device, refuse a Vulkan device rather than run on the CPU, and that `info` lists a synthetic model's architecture, layer count and tensors.
It also checks that the CLI's usage errors exit with status 2 and the command's page on stderr, before any model file is opened.
It shows every help page without a model, and checks that each command takes every flag its page lists and refuses the flags its page does not.
The UBSan job makes misaligned in-memory tensors a test failure.
The three CPU jobs and the UBSan job also run CTest for JSON syntax/Unicode/numeric boundaries and string escaping, GGUF structure, custom alignment and loading failures, Qwen model configuration and required tensor/storage layouts, grouped kernels, worker failures, chat rendering, sampling and KV storage, plus the Python HF/Jinja2 follow-up fixtures and CLI thread-control checks.
The Python suite's `roundtrip` component in these jobs checks the Q8_0, Q4_0, Q4_1 and Q4_K decoders bit for bit against a decode written from the format description, the last two on raw blocks that reach every scale, min and nibble bit, so those readers are covered without a real model.
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
The HF job keeps its 30-minute limit, about twice its time.

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
build/llmx-split-check <Q8_0 fixture> <excerpt> cpu cpu,cpu 8 64
python -X utf8 tools/server_mix_check.py --exe build/llmx --model <Q8_0 fixture> --text tests/data/wiki.test.raw --requests 8 --cli 2
```

The Ubuntu job also runs each `tools/*.py` with `--help`.
The Vulkan job runs the suite with `--device cpu` on its build, `-DLLMX_HAS_BACKEND_VULKAN=ON`.
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
Python chat checks keep progress on stderr and compare follow-up replies to the HF goldens with progress enabled and disabled.

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

## Exact reduction test compilation

The `backend-group` test disables implicit floating-point contraction on
GCC/Clang, including AppleClang. Its explicit SIMD FMA and `std::fma` calls
remain fused. This fixes scalar-tail rounding for the bitwise ordered oracle:
UBSan can otherwise make the compiler fuse one multiply/add expression and
leave the identical expression in the other path unfused. Equality and
numerical bounds are unchanged. The option applies only to this test target,
not the CLI or performance tools. It proves ordered reduction under controlled
contraction; the CLI HF checks separately exercise the normal runtime flags.

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

Native counts are 23 on Windows and 22 on Linux/macOS; the Windows-only `prefill-placement` target accounts for the difference, and a build with `LLMX_HAS_BACKEND_VULKAN=ON` adds `backend-vulkan`, `vulkan-buffer` and `vulkan-lifetime`: 26 native tests on Windows and 25 on Linux/macOS.
The buffer test runs on a fake device that supplies every Vulkan call, so it needs no loader; the lifetime test opens a device and intercepts transfers for ownership checks, including failed padded-cache invalidation.
It also substitutes five kernel creation failures to check cleanup/retry, and runs two real diagnostic-query cases for failed creation and idle-before-destruction.
Query cases skip on a device without diagnostic timestamps.
None replaces the kernel or HF gate.
At placement release `3c5d4b9`, Windows 12/12 and Linux 11/11 passed locally.
That release also passed all five hosted jobs in [run 35516912422](https://github.com/mxxm-t/llmx/actions/runs/35516912422), including the new native targets on Windows, macOS Intel, Linux and Linux UBSan, plus the required HF job.
The earlier `851d375` pass predates these added tests.
