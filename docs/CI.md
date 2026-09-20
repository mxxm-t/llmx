# Continuous integration

`.github/workflows/ci.yml` runs on pull requests, pushes to `main`, and manual
dispatch. It contains four independent checks:

| Check | Coverage |
|---|---|
| CPU (ubuntu-24.04) | GCC, CMake Release, synthetic tests and benchmark smoke |
| CPU (windows-2022) | MSVC, CMake Release, synthetic tests and benchmark smoke |
| CPU (macos-15-intel) | Apple Clang, CMake Release, synthetic tests and benchmark smoke |
| HF reference (CPU) | Linux build plus both pinned real models: tokenizer, logits, continuous/chunked PPL |

All four jobs passed in the [initial hosted run](https://github.com/mxxm-t/llmx/actions/runs/35440893448)
at `ec74308`. Local Windows MSVC and WSL Linux GCC CMake builds also passed
the suite with both HF fixtures required. Workflow lint and negative checks
for corrupt downloads, missing fixtures and invalid throughput passed.

The ordinary test suite also checks `--version` format and agreement with the
usage banner. Build metadata does not change the inference kernels. Local
Windows plain/CMake builds and Linux CMake builds pass the version checks.
The Windows required-HF suite passes; the local Linux synthetic suite passes
with real-model checks skipped. Lifecycle checks cover tracked changes,
commit refresh without reconfiguration, no-op rebuilds and no-Git/archive
fallback. Windows build-script failure checks reject failed toolchain setup,
blocked output directories and unwritable version headers before compilation.

The CPU backend currently uses x86 intrinsics, and CMake enables AVX2/FMA/F16C.
Runtime checks inside some kernels do not make that binary safe on older CPUs.
Intel macOS is intentional; ARM and a portable scalar build are not covered.
GPU jobs should be added when the device execution model and each backend
exist. Actual GPU numerical/performance results require the corresponding
hardware; compilation alone does not establish backend correctness.

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

Every CI job also runs `python -X utf8 tests/fetch_models.py`: fifteen offline
tests cover throttling, reset headers, retry exhaustion, interrupted reads,
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

Hosted jobs pass `--no-perf-floor`: `bench` must still run and report finite,
positive throughput, but the workstation-specific floors are disabled.
Hosted timings are diagnostic. The performance gate against mx-llama.cpp
still requires matched hardware, model, quant and workload; see ROADMAP #8.

To reproduce locally:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel 2
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
workflow does not expose the GPU rig to pull-request jobs.

After the updated hosted run succeeds, the four stable check names above can be
required for `main`. Branch protection is a separate repository setting;
adding this workflow does not enable it automatically.

The reported [run at d6e00e0](https://github.com/mxxm-t/llmx/actions/runs/35498190148)
failed while fetching the Q8_0 fixture with HTTP 429, before HF tests ran.
Retry/cache handling addresses that download failure; persistent service
throttling can still exhaust the bounded retry policy and fail the job. The
[updated hosted run](https://github.com/mxxm-t/llmx/actions/runs/35510681421)
passed all four jobs, including fixture downloads and required HF checks.
