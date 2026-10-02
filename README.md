# llmx

[![CI](https://github.com/mxxm-t/llmx/actions/workflows/ci.yml/badge.svg)](https://github.com/mxxm-t/llmx/actions/workflows/ci.yml)

llmx is a dependency-free LLM inference runtime in C++17 for CPUs and GPUs, run from the command line or as an OpenAI-compatible server, and able to run one model over several devices of one machine.
The tables below list the supported models, file formats, quantization types, backends and multi-device modes, each marked **Supported** (works today) or **Planned** (on the [ROADMAP](docs/ROADMAP.md), not yet qualified for release).

## Supported models

| Architecture | Models tested | CPU | Vulkan | `llmx serve` |
|---|---|---|---|---|
| `qwen3` (dense) | Qwen3-0.6B, 8B, 14B, 32B | Supported | Supported | Supported |
| `qwen3moe` (mixture of experts) | Qwen3-30B-A3B, Qwen3-235B-A22B | Supported | Supported | Supported |
| `qwen35` (hybrid linear and full attention) | Qwen3.5-0.8B, 4B, 9B, Qwen3.6-27B, Qwen3.8-27B | Supported | Supported | Supported, prefix reuse for follow-up turns |
| `qwen35moe` (Qwen 3.5 mixture of experts) | Qwen3.6-35B-A3B | Supported | Supported | Supported, prefix reuse for follow-up turns |
| Llama, Mistral, Gemma, Phi, DeepSeek V4 | | Planned | Planned | Planned |

Outputs are checked against Hugging Face reference outputs; [ASSETS](docs/ASSETS.md) lists what is covered.

## Quantization types

| Tensor type | CPU | Vulkan | Written by `llmx quantize` |
|---|---|---|---|
| F32, Q8_0, Q4_0 | Supported | Supported | Q8_0, Q4_0 |
| Q4_1, Q4_K, Q5_K, Q6_K (so Q4_K_M and Q5_K_M files) | Supported | Supported | No |
| MXFP4 | Supported | Supported on devices with the required float preservation and double arithmetic | No |
| F16, BF16, IQ4_NL, IQ4_XS, Q3_K, Q2_K | Planned | Planned | No |

A file loads when every tensor type in it is supported on its device; a device never falls back to the CPU on its own.
Activation precision is selected with `--dtype auto|f16|bf16|f32`; `auto` selects F16 on the supported AVX2 CPU, MI50 and Radeon VII paths. Explicit BF16 is emulated on those tested devices. This does not add F16 or BF16 weight-file support; see [precision](docs/USAGE.md#precision) for the execution policy and reported fallbacks.
The KV cache is stored as `f16` (default) or `f32` on every backend, independently of activation precision.

## File formats

| Format | Status |
|---|---|
| GGUF v3, one file or a shard set | Supported: read by every model command, written by `quantize` |
| Hugging Face download | Supported: `llmx pull <owner/repo>:<quant>` |
| Raw F32 tensors (`.json` + `.bin`) | Supported: input of `quantize`, output of `dequantize` |
| safetensors, with `config.json` and `tokenizer.json` | Planned |

## Backends and devices

| Backend | Hardware | Systems | Status |
|---|---|---|---|
| CPU | x86-64 with AVX2, FMA and F16C | Windows, Linux, Intel macOS | Supported |
| Vulkan | Vulkan 1.2 GPUs; tested on AMD Radeon VII and AMD MI50 | Windows, Linux | Supported |
| ROCm (HIP) | AMD GPUs | Linux | Planned |
| CUDA | NVIDIA GPUs | | Planned |
| SYCL | Intel GPUs | | Planned |

ARM CPUs, Apple Silicon included, are not supported; the Vulkan kernels are tuned for 64-wide subgroups, and other GPUs are untested.
Pick a device with `--device cpu` (the default) or `--device vulkan:N`; see [USAGE](docs/USAGE.md#device-selection---device).

## Multi-device modes

| Mode | What it does | Flags | Status |
|---|---|---|---|
| Layer split | Consecutive layers on different devices, GPUs and the CPU mixed, fitted to free memory, for every supported model in every command and `llmx serve` | `--device A,B,...`, `--layer-shares` | Supported |
| Expert offload | A mixture-of-experts model's experts run on the CPU beside one GPU | `--n-cpu-moe N`, `--cpu-moe`, `--moe-stream-from N` | Supported |
| Tensor split | Every layer on a group of 2 to 4 devices at once | | Planned |
| Staged tensor split | A layer split whose stages are tensor splits | | Planned |
| Replicas | Several copies of a model behind one scheduler | | Planned |
| Multi-node | One model over several machines | | Planned |

[USAGE](docs/USAGE.md#several-devices---device-ab---layer-shares) covers the flags and [MULTI-DEVICE](docs/MULTI-DEVICE.md) the design.

## Server and API

| Routes | API |
|---|---|
| `/v1/chat/completions`, `/v1/completions`, `/v1/models` | OpenAI-compatible, whole or streamed, with log-probabilities; a reasoning model's thinking comes in `reasoning_content` apart from the answer, and `chat_template_kwargs` sets template options such as `enable_thinking` |
| `/v1/generate`, `/v1/chat`, `/v1/tokenize`, `/v1/detokenize`, `/v1/health` | Native, with token ids |

Tool calls, embeddings and more than one choice per request are not supported.
There is no TLS or authentication, so put a reverse proxy in front of a server that faces a network.
[USAGE](docs/USAGE.md) lists the request fields and server flags, and [SERVER](docs/SERVER.md) the scheduler's design.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

Add `-DLLMX_HAS_BACKEND_VULKAN=ON` to the first command for the Vulkan backend.
The binary is `build/llmx`, or `build\Release\llmx.exe` with Visual Studio; on Windows, `build.bat` builds a CPU-only `llmx.exe` without CMake.
[BUILD](docs/BUILD.md) lists the compilers and packages for each platform and a Docker image.

## Run a model

```sh
model=$(./build/llmx pull Qwen/Qwen3-0.6B-GGUF:Q8_0)
./build/llmx generate "$model" "The capital of France is" --temp 0 -n 32
./build/llmx chat "$model" -n 512 --device vulkan:0
./build/llmx serve "$model"
```

`pull` needs curl 8.4 or later and reads `HF_TOKEN` for gated repositories; the server listens on `127.0.0.1:8080`.
`llmx bench --model "$model"` measures prefill and decode speed on your hardware.
`llmx <command> --help` shows a command's options, and [USAGE](docs/USAGE.md) covers every command and flag.

## Documentation

- [USAGE](docs/USAGE.md): every command and flag.
- [BUILD](docs/BUILD.md): building on each platform, and running the tests.
- [ROADMAP](docs/ROADMAP.md): what is planned.
- [ARCHITECTURE](docs/ARCHITECTURE.md), [SERVER](docs/SERVER.md), [VULKAN](docs/VULKAN.md) and [MULTI-DEVICE](docs/MULTI-DEVICE.md): how the runtime, the server, the Vulkan backend and the multi-device modes work.
- [AGENTS](AGENTS.md): for contributors.
