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

The Python suite also checks reference-generator argument safeguards and that
the requested commit, float32 dtype and eager attention reach the HF loader.
These use standard-library test doubles; CI does not generate new HF goldens
or download larger models. The ordinary suite now has 11 components, including
`reference-consumer` rejection tests for 8B fixture tampering, malformed or
out-of-bound numerical output, wrong model identity and failed launches. These
tests use small committed JSON fixtures and doubles, without 8B inference.
Default real-model downloads remain the two pinned 0.6B GGUFs.

The separate 8B consumer requires an existing model and a new output directory:

```
python -X utf8 tests/baseline_8b.py --exe build/llmx --model path/to/Qwen3-8B-Q8_0.gguf --output-dir hf-8b-review
```

Use `--exe build/Release/llmx.exe` for MSVC. It verifies model/fixture hashes,
records executable identity, commands and failures in `report.json`, saves raw
output beside it, and never downloads or skips a missing model. Frozen bounds
require exact token IDs, top-1 agreement and top-5 overlap 5/5, with absolute NLL deltas <= 0.01
continuous and <= 0.02 windowed. Top-10 output must be finite, sorted, unique-ID
and within absolute magnitude 100. This optional run is outside default CI;
see [ASSETS](ASSETS.md#optional-qwen3-8b-hf-consumer) for reference provenance,
the verified Linux cache path and the limits of short-excerpt coverage. Local
Windows and Linux runs each pass 37/37 checks with identical printed NLLs and
HF deltas. The Linux ordinary suite passes 11/11 with `--no-perf-floor`.
These local results do not establish hosted 8B coverage; the optional consumer
is not run by the workflow.

Every job checks that `--version` and the usage banner agree with the release
version, then runs the small F32 HF fixture without downloads. Its deterministic
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

CTest also covers synchronous text delivery before the next model step, legacy
filtering and split UTF-8 bytes, plus loader progress, truncated reads and
consumer exceptions. `cli-output` observes flushing through the actual CLI
emitter with a controlled stream buffer, without wall-clock timing assertions.
Python chat checks keep progress on stderr and compare
follow-up replies to the HF goldens with progress enabled and disabled.
