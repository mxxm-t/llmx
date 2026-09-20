# llmx

[![CI](https://github.com/mxxm-t/llmx/actions/workflows/ci.yml/badge.svg)](https://github.com/mxxm-t/llmx/actions/workflows/ci.yml)

A ground-up C++ LLM inference runtime, built to control the full stack from
model files and tokenization to compute kernels and serving.

The goal is a dependency-free core with hand-written CPU and GPU kernels,
multiple model architectures, and execution across devices and machines.
Hugging Face integration is part of that direction: downloading pinned models,
reading native Hub formats, and validating inference against HF references.
The project is early; most of that broader execution and serving work is planned.

## Works today

- Dense **Qwen3** inference on x86 CPU with **AVX2/FMA/F16C**.
- GGUF v3 reading/writing, including mixed Q8_0, Q4_0, Q4_1, Q4_K, Q5_K,
  Q6_K and F32 tensors. The CLI quantizes to Q8_0 or Q4_0; K-quants are read-only.
- Native `llmx pull` downloads pinned GGUF models with multiple streams,
  verified caching and gated-repo credentials. Sharded GGUF loads through the
  same model commands. Downloads use curl 8.4+; Python is not required.
- Byte-level BPE with Qwen2/Qwen3 pretokenization, generation and interactive
  follow-up chat using a Jinja2-subset template renderer. Text streams as tokens
  arrive; legacy reasoning filters retain buffering. Loading and processing
  status appears on stderr in a terminal or with `--verbose`.
- Batched prompt processing, a growing CPU KV cache, sampling and windowed
  perplexity. One model instance currently handles one sequence at a time.

See [usage](docs/USAGE.md#llmx-pull-ownerrepoquant) for the download/cache
interface. ARM, GPU execution, additional model architectures and a multi-user
server are not implemented yet. Some development checkpoints
remain on feature branches while their performance gates are open; see
[development status](docs/STATUS.md) for the current state.

## Direction

| Area | Planned work |
|---|---|
| Model coverage | Llama, Mistral, Gemma and Phi; additional quantizations |
| Hugging Face | Safetensors, BF16/F16 tensors, HF tokenizer/config files |
| Device execution | Backend-owned buffers, resident activations and asynchronous submission |
| GPU backends | ROCm as a first-class target; CUDA and SYCL; Vulkan for portability |
| Multiple devices/nodes | Model splitting across devices and cluster nodes |
| Serving | Shared read-only weights, independent request/KV state, continuous batching and streaming |
| Hub kernels | Optional later work: port suitable kernel source or distribute llmx kernels through the Hub |

Device execution must be refactored before useful GPU backends can be added.
The existing synchronous host-pointer backend interface is a starting point.
CPU worker parallelism does not make a model instance safe for concurrent users.

The runtime has no external libraries today. Planned GPU SDKs are a deliberate
build dependency; vendor math libraries are outside the design. HF download
support uses system curl HTTPS tooling. Direct loading of PyTorch Hub
kernel extensions would introduce PyTorch/Python dependencies and is not planned.
See the [roadmap](docs/ROADMAP.md) for dependencies, priorities and scope.

## Quality gates

**Correctness is measured against independent Hugging Face references.** Tests
include tokenizer IDs, logits, NLL and chat replies. Equality with an earlier
llmx build supplements those checks; it cannot replace them.

**Performance must meet mx-llama.cpp** on the same model, quantization, prompt
and hardware, for both prefill and decode. This is the project target, not an
achieved universal parity claim. Current gaps and comparison tables are in
[STATUS](docs/STATUS.md); scopes, limitations and reproducible evidence are in
[ASSETS](docs/ASSETS.md).

## Build

Windows with Visual Studio's C++ tools:

```bat
build.bat
```

This produces `llmx.exe` in the repository root. For Windows, Linux and Intel
macOS through CMake:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Current builds require x86 AVX2/FMA/F16C. Git-enabled builds identify themselves
as `0.1.0+g<commit>` (at least 12 revision characters), with `.dirty` for tracked
changes. The identifier refreshes on rebuild. Source archives without
Git metadata report `0.1.0+unknown`; no timestamp or automatic release bump is used.

## Use

```sh
llmx --version
llmx generate model.gguf "The capital of France is" -n 64
llmx chat model.gguf --system "You are a terse assistant."
llmx perplexity model.gguf --file corpus.txt --ctx-size 512
```

Use `llmx.exe` on Windows. See [USAGE](docs/USAGE.md) for all commands and flags.

## Test

```sh
ctest --test-dir build -C Release --output-on-failure
python tests/run_tests.py --exe build/llmx
```

For MSVC CMake builds, use `--exe build/Release/llmx.exe`; the plain Windows
build is selected by default when `--exe` is omitted. Synthetic fixtures and
small committed HF goldens run without external Python packages. Real-model
HF checks skip when models are absent; fetch the pinned fixtures with
`python tools/fetch_test_models.py` and add `--require-baseline` to require them.

Native tests cover JSON parsing/string escaping, GGUF structure/custom alignment,
Qwen model configuration and tensor layouts, grouped kernels, worker
failures, chat templates, KV
storage, early text delivery, CLI flushing and loader progress/error handling.
The Python suite covers conversion, tokenization, F32 logits,
perplexity, follow-up chat, thread controls and performance guardrails.
[CI](docs/CI.md) describes the configured platform jobs and their limits;
shared-runner timings do not establish the external performance floor.

## Project guide

- [Architecture](docs/ARCHITECTURE.md): layer boundaries and execution design.
- [Roadmap](docs/ROADMAP.md): planned features and prerequisites.
- [Status](docs/STATUS.md): current checkpoints, validation and remaining work.
- [Source documentation](docs/src/): responsibilities of individual files.
- [Contributor guidance](AGENTS.md): build, test and checkpoint rules.

The intended dependency direction is
`cli > inference > model > backends > tokenizer > format > quant > core`.
Some current code still couples directly to GGUF, including quant type constants;
further formats and device execution need integration work rather than just a
registry entry. Keep changes small, complete and supported by measurements.
