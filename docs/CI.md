# Continuous integration

`.github/workflows/ci.yml` runs on pull requests, pushes to `main`, and manual
dispatch. It contains five independent checks:

| Check | Coverage |
|---|---|
| CPU (ubuntu-24.04) | GCC, CMake Release, synthetic tests and benchmark smoke |
| CPU (windows-2022) | MSVC, CMake Release, synthetic tests and benchmark smoke |
| CPU (macos-15-intel) | Apple Clang, CMake Release, synthetic tests and benchmark smoke |
| CPU (Linux UBSan) | GCC undefined-behavior checks, including mixed-tensor float alignment |
| HF reference (CPU) | Linux build plus both pinned real models: tokenizer, logits, continuous/chunked PPL |

The original four jobs passed in the [initial hosted run](https://github.com/mxxm-t/llmx/actions/runs/35440893448)
at `ec74308`. Local Windows MSVC and WSL Linux GCC CMake builds also passed
the suite with both HF fixtures required. Workflow lint and negative checks
for corrupt downloads, missing fixtures and invalid throughput passed.

The CPU backend currently uses x86 intrinsics, and CMake enables AVX2/FMA/F16C.
Runtime checks inside some kernels do not make that binary safe on older CPUs.
Intel macOS is intentional; ARM and a portable scalar build are not covered.
GPU jobs should be added when the device execution model and each backend
exist. Actual GPU numerical/performance results require the corresponding
hardware; compilation alone does not establish backend correctness.

The HF job runs `tools/fetch_test_models.py`, a standard-library downloader
using the revisions and SHA-256 digests in `tests/baseline.py`. Downloads are
verified before entering the HF snapshot cache. `--require-baseline` makes
missing fixtures fatal, preventing a green numerical job made entirely of
skips. Test execution does not install torch, transformers or HF packages.
The ordinary CPU jobs can skip real-model checks because their fixtures are
absent; the separate HF job supplies that coverage.

Every job also runs the small F32 HF fixture without downloads. Its deterministic
weights are generated locally; committed HF float32 logits/NLL cover tied and
untied embeddings, matrix tails, multiple physical batches and thread counts.
The UBSan job makes misaligned in-memory tensors a test failure. Every job also
runs CTest for grouped kernels, worker failures, chat rendering and KV storage,
plus the Python HF/Jinja2 follow-up fixtures and CLI thread-control checks.
The five-job workflow and these
new native checks still await a hosted run for the current unmerged stack;
the initial four-job result above does not validate this branch.

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

After the hosted runs succeed, the stable check names above can be
required for `main`. Branch protection is a separate repository setting;
adding this workflow does not enable it automatically.
