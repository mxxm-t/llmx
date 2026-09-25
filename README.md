# llmx

[![CI](https://github.com/mxxm-t/llmx/actions/workflows/ci.yml/badge.svg)](https://github.com/mxxm-t/llmx/actions/workflows/ci.yml)

A ground-up C++ LLM inference runtime, built to control the full stack from
model files and tokenization to compute kernels and serving.

The goal is a dependency-free core with hand-written CPU and GPU kernels,
multiple model architectures, and execution across devices and machines.
Hugging Face integration is part of that direction: downloading pinned models,
reading native Hub formats, and validating inference against HF references.
The CPU and Vulkan backends, the batched execution model and the server are
in the tree, including contiguous layer splits across local devices. Tensor
groups, execution across machines, further vendor backends and further
architectures are planned.

## Works today

- **Qwen3** inference, dense and mixture of experts (`qwen3moe`, such as
  Qwen3-30B-A3B), on x86 CPU with **AVX2/FMA/F16C** and on Vulkan devices,
  with a model's experts optionally on the CPU beside a device (`--n-cpu-moe`).
- GGUF v3 reading/writing, including mixed Q8_0, Q4_0, Q4_1, Q4_K, Q5_K,
  Q6_K and F32 tensors. The CLI quantizes to Q8_0 or Q4_0; K-quants are read-only.
- Native `llmx pull` downloads pinned GGUF models with multiple streams,
  verified caching and gated-repo credentials. Sharded GGUF loads through the
  same model commands. Downloads use curl 8.4+; Python is not required.
- Byte-level BPE with Qwen2/Qwen3 pretokenization, generation and interactive
  follow-up chat using a Jinja2-subset template renderer. Text streams as tokens
  arrive; legacy reasoning filters retain buffering. Loading and processing
  status appears on stderr in a terminal or with `--verbose`.
- Batched prompt processing, a paged KV cache, sampling and windowed
  perplexity. A model runs a batch of sequences per pass, each at its own
  positions over its own history, and can be placed across several
  backends; the CLI drives one sequence on the CPU, a Vulkan device or a
  layer split selected with `--device` and optional `--layer-shares`.
- A **Vulkan backend** (`-DLLMX_HAS_BACKEND_VULKAN=ON`, `--device vulkan:N`)
  with kernels for every type the CPU reads, f16 or f32 KV caches, integer
  activations in decode and, where the device's 8-bit integer dot is
  native, 8-bit decode activations and an integer-dot prefill tile, a
  prompt computing the same however it is batched, checked against the CPU backend and the HF
  references and measured against the reference runtime's Vulkan build on a
  Radeon VII and one MI50 (`docs/VULKAN.md`).
- **`llmx serve`**: one model, a sequence per request, continuous batching
  with chunked prefill, streaming, prompt-prefix reuse through KV forks, and
  the OpenAI-compatible routes beside native ones, so existing clients
  connect unchanged (`docs/SERVER.md`).

See [usage](docs/USAGE.md#llmx-pull-ownerrepoquant) for the download/cache
interface. Local layer splitting is implemented; its recorded coverage is in
[multi-device status](docs/STATUS.md), where the pinned Qwen3-8B HF check,
once open on the MI50, now passes 41 of 41 on one card. ARM,
tensor groups, execution across machines, other vendor backends and additional
model architectures are not implemented yet; see
[development status](docs/STATUS.md) for the current state.

## Direction

| Area | State and plan |
|---|---|
| Model coverage | Qwen3, dense and mixture of experts, today; Llama, Mistral, Gemma and Phi and additional quantizations planned |
| Hugging Face | `llmx pull` today; safetensors, BF16/F16 tensors and the HF tokenizer/config files planned |
| Execution model | Done: tickets, batched sequence views and device placement (`docs/EXECUTION.md`) |
| Server | Done: `llmx serve` with continuous batching, streaming HTTP without dependencies, prefix reuse and the OpenAI-compatible routes (`docs/SERVER.md`, `docs/USAGE.md`) |
| GPU backends | Vulkan done, running on a Radeon VII under Windows and on MI50s under Linux. Against the reference's own Vulkan build on the same card, over the dense-model device gate (five Qwen3 files, prompts of 64 to 512 tokens and 32 tokens decoded after a short prompt), decode is 102 to 115 percent and prefill 109 to 455 percent on the Radeon VII, and decode 102 to 115 percent and prefill 102 to 267 percent on one MI50; decode after a 16k-token history, behind the reference on the MI50, is recorded in `docs/STATUS.md`; ROCm first-class on Linux, CUDA and SYCL planned |
| Multiple devices/nodes | Local layer splits with memory fitting, manual layer shares and a prompt pipelined over the stages implemented; server passes in flight, tensor groups and cluster nodes planned |
| Hub kernels | Optional later work: port suitable kernel source or distribute llmx kernels through the Hub |

The device execution model is complete and the Vulkan backend is written
over it, so a further vendor backend implements the same `Backend`
interface. CPU worker parallelism does not make a model instance safe for
concurrent users; the server's scheduler is what does.

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
and hardware, for both prefill and decode, and the server must beat the
reference runtime's server by a wide margin at concurrency on time to first
token, inter-token latency and throughput. These are the project targets, not
achieved universal claims. Current gaps and comparison tables are in
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

Run `llmx --help` for the command overview or `llmx chat --help` for focused
options and examples. See the [command reference](docs/USAGE.md) for details.

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
failures, chat templates, the paged KV cache and its forks, placement across
backends, the HTTP layer, early text delivery, CLI flushing and loader
progress/error handling, and every Vulkan kernel against the CPU backend
when a device is present. The Python suite covers conversion, tokenization,
F32 logits, perplexity, follow-up chat, thread controls, the server's routes
and performance guardrails, on the CPU or with `--device vulkan:N`.
[CI](docs/CI.md) describes the configured platform jobs, including the Linux
Vulkan build, and their limits; shared-runner timings do not establish the
external performance floor.

## Project guide

- [Architecture](docs/ARCHITECTURE.md): layer boundaries and execution design.
- [Roadmap](docs/ROADMAP.md): planned features and prerequisites.
- [Status](docs/STATUS.md): current checkpoints, validation and remaining work.
- [Source documentation](docs/src/): responsibilities of individual files.
- [Contributor guidance](AGENTS.md): build, test and checkpoint rules.

The intended dependency direction is
`cli > server > inference > model > backends > tokenizer > format > quant > core`.
Some current code still couples directly to GGUF, including quant type constants;
further formats and device execution need integration work rather than just a
registry entry. Keep changes small, complete and supported by measurements.
