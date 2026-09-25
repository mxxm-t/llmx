# Continuous integration

`.github/workflows/ci.yml` runs on pull requests, pushes to `main`, and manual
dispatch. It contains six independent checks:

| Check | Coverage |
|---|---|
| CPU (ubuntu-24.04) | GCC, CMake Release, synthetic tests and benchmark smoke |
| CPU (windows-2022) | MSVC, CMake Release and `build.bat`, synthetic tests and benchmark smoke on both binaries, and the `build.bat` binary reporting the same version as the CMake one |
| CPU (macos-15-intel) | Apple Clang, CMake Release, synthetic tests and benchmark smoke |
| CPU (Linux UBSan) | GCC undefined-behavior checks, including mixed-tensor float alignment |
| Vulkan backend (build, Linux) | The backend and every shader compiled with `-DLLMX_HAS_BACKEND_VULKAN=ON`, the headers and `glslc` from the LunarG repository, pinned there since the distribution's compiler is older than the shader extensions the kernels use and has not been retried; CTest with `backend-vulkan` and `vulkan-lifetime` skipping without a driver, while `vulkan-buffer` exercises fake API cleanup with only the loader |
| HF reference (CPU) | Linux build plus all three pinned real models: tokenizer, logits, continuous/chunked PPL |

Every CTest in `CMakeLists.txt` runs in every job's "Backend tests" step,
so the KV cache, placement, HTTP layer and prefill-scope checks are covered
on all three platforms and under UBSan, and the Python suite's `server`
component starts `llmx serve` on the synthetic model in every CPU job and
on the real Q8_0 fixture in the HF job. What no hosted job establishes is
device behaviour: the Vulkan job proves the tree compiles, and the kernel
comparisons, the HF gate on the device and the matched floors are run on
the Radeon VII and the Linux machine's MI50s by hand and recorded in `docs/STATUS.md`. A self-hosted
runner on that machine would close that. It needs no packages of its own
for it: `docker/Dockerfile` carries the driver and the compiler and takes
the cards through `/dev/dri`, and inside it the whole CTest suite,
`backend-vulkan` included, passes on an MI50.

The layer split is covered on the CPU.
The `placement` CTest, in every job, splits a model over two and three CPU backends, among them a pipelined prompt of five chunks, which reuses pass slots and handoff buffers, and its rollback when a backend on the last stage fails.
Splits over GPUs are run by hand on the Radeon VII and the MI50s.

The original four jobs passed in the [initial hosted run](https://github.com/mxxm-t/llmx/actions/runs/35440893448)
at `ec74308`. Local Windows MSVC and WSL Linux GCC CMake builds also passed
the suite with both HF fixtures required. Workflow lint and negative checks
for corrupt downloads, missing fixtures and invalid throughput passed.

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

The HF job runs `tools/fetch_test_models.py`, a standard-library downloader
using the revisions and SHA-256 digests in `tests/baseline.py`. Downloads are
verified before entering the HF snapshot cache. The HF job caches those
snapshots between runs, with a key derived from `tests/baseline.py`; restored
files are still SHA-256 checked on every run. Cold or invalid cache entries
are downloaded from the pinned revision.

The downloader makes at most five attempts for HTTP 408/429/500/502/503/504
and transient connection/read failures. Backoff is 30/60/120/240 seconds;
valid `Retry-After` and HF `RateLimit` reset headers can extend each wait to
at most 300 seconds. A longer server wait fails with a clear diagnostic,
rather than retrying early. Permanent HTTP failures, local file errors and
SHA-256 mismatches fail immediately. Failed attempts remove temporary files;
only a complete verified download replaces the destination.

Every job except the Vulkan build also runs
`python -X utf8 tests/fetch_models.py`: fifteen offline tests cover
throttling, reset headers, retry exhaustion, interrupted reads,
cache reuse/replacement, checksum rejection and permanent failures. These
tests use tiny independent bytes and simulated network responses; they do
not download models or replace the real HF reference checks. All fifteen pass
on Linux, including a real HTTP response parser test for premature EOF; the
previous downloader reproduces the single-request 429 failure.

`--require-baseline` makes
missing fixtures fatal, preventing a green numerical job made entirely of
skips. Test execution does not install torch, transformers or HF packages.
The ordinary CPU jobs can skip real-model checks because their fixtures are
absent; the separate HF job supplies that coverage.

The Python suite also checks reference-generator argument safeguards and that the requested commit, float32 dtype and eager attention reach the HF loader.
These use standard-library test doubles; CI does not generate new HF goldens or download larger models.
The ordinary suite now has 15 components, including `reference-consumer` rejection tests for 8B fixture tampering, malformed or out-of-bound numerical output, wrong model identity and failed launches, and a passing 8B run over simulated outputs that must have 41 checks with each NLL case scored in both modes.
These tests use small committed JSON fixtures and doubles, without 8B inference.
Default real-model downloads are the three pinned 0.6B GGUFs: Q8_0, Q4_0 and Q5_K_M.

The separate 8B consumer requires an existing model and a new output directory:

```
python -X utf8 tests/baseline_8b.py --exe build/llmx --model path/to/Qwen3-8B-Q8_0.gguf --output-dir hf-8b-review
```

Use `--exe build/Release/llmx.exe` for MSVC.
It verifies model/fixture hashes, records executable identity, commands and failures in `report.json`, saves raw output beside it, and never downloads or skips a missing model.
Frozen bounds require exact token IDs, top-1 agreement and top-5 overlap 5/5, with absolute NLL deltas <= 0.01 continuous and <= 0.02 windowed.
Each NLL case is scored twice, in batched passes and with `--per-token`, as in `tests/baseline.py`, so a run has 41 checks.
Top-10 output must be finite, sorted, unique-ID and within absolute magnitude 100; `tests/baseline.py` holds the 0.6B outputs to the same validators at its own bounds.
This optional run is outside default CI; see [ASSETS](ASSETS.md#optional-qwen3-8b-hf-consumer) for reference provenance, the verified Linux cache path and the limits of short-excerpt coverage.
Local Windows and Linux runs each passed the 37 checks the consumer had before the per-token half, with identical printed NLLs and HF deltas.
The Linux ordinary suite passed 11/11 with `--no-perf-floor` at the time.
These local results do not establish hosted 8B coverage; the optional consumer is not run by the workflow.

Every job except the Vulkan build checks that `--version` and the usage
banner agree with the release version, then runs the small F32 HF fixture
without downloads. Its deterministic weights are generated locally;
committed HF float32 logits/NLL cover tied and untied embeddings, matrix
tails, multiple physical batches and thread counts.
The same fixture holds the rows `logits --last` and `--then-ids` print to its bound at their positions, and `logits --file` to the inline prompt's output.
The `cli` component checks that these builds, which have no Vulkan backend, refuse a Vulkan device rather than run on the CPU, and that `info` lists a synthetic model's architecture, layer count and tensors.
The UBSan job makes misaligned in-memory tensors a test failure. These jobs also
run CTest for JSON syntax/Unicode/numeric boundaries and string escaping,
GGUF structure, custom alignment and loading failures, Qwen model configuration
and required tensor/storage layouts,
grouped kernels, worker
failures, chat rendering, sampling and KV storage,
plus the Python HF/Jinja2 follow-up fixtures and CLI thread-control checks.
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

To reproduce locally:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel 2
ctest --test-dir build -C Release --output-on-failure
python -X utf8 tests/fetch_models.py
python -X utf8 tests/run_tests.py --exe build/llmx --no-perf-floor
python -X utf8 tools/fetch_test_models.py
python -X utf8 tests/run_tests.py --exe build/llmx --no-perf-floor --require-baseline
```

For MSVC, use `--exe build/Release/llmx.exe`. Omitting `--no-perf-floor`
preserves the existing local timing floors. `build.bat` still builds the root
`llmx.exe`, which remains the Windows test runner's default.

Actions are pinned to commit SHAs, checkout credentials are not persisted,
and workflow permissions are read-only. Jobs run on hosted machines; this
workflow does not expose the GPU machines to pull-request jobs.

After the hosted runs succeed, the stable check names above can be
required for `main`. Branch protection is a separate repository setting;
adding this workflow does not enable it automatically.

CTest also covers synchronous text delivery before the next model step, legacy
filtering and split UTF-8 bytes, plus loader progress, truncated reads and
consumer exceptions. `cli-output` observes flushing through the actual CLI
emitter with a controlled stream buffer, without wall-clock timing assertions.
Python chat checks keep progress on stderr and compare
follow-up replies to the HF goldens with progress enabled and disabled.

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

Native counts are 22 on Windows and 21 on Linux/macOS; the Windows-only
`prefill-placement` target accounts for the difference, and a build with
`LLMX_HAS_BACKEND_VULKAN=ON` adds `backend-vulkan`, `vulkan-buffer` and
`vulkan-lifetime`: 25 native tests on Windows and 24 on Linux/macOS.
The buffer test substitutes Vulkan allocation calls and needs only a loader;
the lifetime test opens a device and intercepts transfers for ownership checks,
including failed padded-cache invalidation. It also substitutes five kernel
creation failures to check cleanup/retry, and runs two real diagnostic-query
cases for failed creation and idle-before-destruction. Query cases skip on a
device without diagnostic timestamps. None replaces the kernel or HF gate.
At placement release `3c5d4b9`, Windows 12/12 and Linux 11/11 passed locally.
That release also passed all five hosted jobs in
[run 35516912422](https://github.com/mxxm-t/llmx/actions/runs/35516912422), including
the new native targets on Windows, macOS Intel, Linux and Linux UBSan, plus the
required HF job. The earlier `851d375` pass predates these added tests.
