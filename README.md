# llmx

[![CI](https://github.com/mxxm-t/llmx/actions/workflows/ci.yml/badge.svg)](https://github.com/mxxm-t/llmx/actions/workflows/ci.yml)

llmx is a dependency-free C++17 inference runtime for supported x86-64 CPUs and Vulkan devices. Run it from the command line or as an OpenAI-compatible server, and split one model across devices in one machine.

The tables summarize current model, weight-format and backend support. **Supported** means the path is implemented and has applicable checks. **In development** means the implementation exists but qualification remains open. **Planned** means roadmap work not available in this tree; see the [ROADMAP](docs/ROADMAP.md).

## Supported models

| Architecture | Models tested | CPU | Vulkan | `llmx serve` |
|---|---|---|---|---|
| `qwen3` (dense) | Qwen3-0.6B, 8B, 14B, 32B | Supported | Supported | Supported |
| `qwen3moe` (mixture of experts) | Qwen3-30B-A3B, Qwen3-235B-A22B | Supported | Supported | Supported |
| `qwen35` (hybrid linear and full attention) | Qwen3.5-0.8B, 4B, 9B, Qwen3.6-27B, Qwen3.8-27B | Supported | Supported | Supported, prefix reuse for follow-up turns |
| `qwen35moe` (Qwen 3.5 mixture of experts) | Qwen3.6-35B-A3B | Supported | Supported | Supported, prefix reuse for follow-up turns |
| `qwen4exp` (hybrid with per-layer n-gram embeddings, [details](docs/QWEN38-FLASH.md)) | Qwen3.8-Flash-Next Q8_0 (released-file HF) and UD-Q4_K_XL (bounded serving smoke with Q8_0 MTP sidecar) | Released-file CPU qualification pending | Supported | Supported, prefix reuse for follow-up turns |
| Llama, Mistral, Gemma, Phi, DeepSeek V4 | | Planned | Planned | Planned |

Selected outputs are checked against Hugging Face references; [ASSETS](docs/ASSETS.md) lists the models and checks covered.

## Quantization types

This is the current model-weight support matrix. The [storage metadata page](docs/src/quant-types.md) lists recognized GGUF layouts, including layouts that have no inference decoder.

| Weight format | CPU inference | Vulkan inference | Written by `llmx quantize` |
|---|---|---|---|
| `F32`, `F16`, `BF16` | Supported | Supported | No |
| `MXFP4` | Supported | Supported | No |
| `Q2_K` | Supported | Supported | No |
| `Q3_K` | Supported | Supported | No |
| `Q4_0`, `Q4_1`, `Q4_K` | Supported | Supported | `Q4_0` |
| `Q5_0`, `Q5_1`, `Q5_K` | Supported | Supported | No |
| `Q6_K` | Supported | Supported | No |
| `Q8_0` | Supported | Supported | `Q8_0` |
| `IQ2_XXS`, `IQ2_XS`, `IQ2_S` | Supported | Supported | No |
| `IQ3_XXS`, `IQ3_S` | Supported | Supported | No |
| `IQ4_NL`, `IQ4_XS` | Supported | Supported | No |
| `TQ1_0`, `TQ2_0` | Qualification pending | Qualification pending | No |

Vulkan `MXFP4` inference requires float-preservation and double-arithmetic support from the device. Each model tensor must be supported by its assigned backend; devices do not fall back to the CPU automatically.

GGUF metadata can recognize layouts without an inference decoder; `info`, `tokenize` and `detokenize` accept those known layouts. `Q8_1` and `Q8_K` are auxiliary formats. See the [storage type list](docs/src/quant-types.md); metadata recognition alone does not make a weight executable.

Activation `--dtype` is independent of weight storage and KV-cache type. `auto` selects a backend policy; requested F16, BF16 or int8 paths may use a wider implementation or input rounding depending on the device. See [precision and dtype reporting](docs/USAGE.md#precision). F16 is the KV-cache default; F32 is also available.

## File formats

| Format | Status |
|---|---|
| GGUF v3, one file or a shard set | Supported: read by every model command, written by `quantize` |
| Hugging Face download | Supported: `llmx pull <owner/repo>:<quant>` |
| Raw F32 tensors (`.json` + `.bin`) | Supported: input of `quantize`, output of `dequantize` |
| safetensors, with `config.json` and `tokenizer.json` | Planned |

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
| Tensor split | Every layer on a group of 2 to 4 devices at once; dense `qwen3`, hybrid `qwen35` and `qwen4exp` and mixture-of-experts `qwen3moe` and `qwen35moe` models, with their embedded drafter, on Vulkan devices on Linux | `--tensor-width N` | Supported; ahead of or below the layer split by width, precision and user count ([USAGE](docs/USAGE.md#tensor-split---tensor-width-n)) |
| Staged tensor split | A layer split whose stages are tensor groups | `--tensor-width N` with several groups listed in `--device`; `--layer-shares` gives each group its share | Supported, as above |
| Replicas | Several copies of a model behind one scheduler | | Planned |
| Multi-node | One model over several machines | | Planned |

[USAGE](docs/USAGE.md#several-devices---device-ab---layer-shares) covers the flags, [MULTI-DEVICE](docs/MULTI-DEVICE.md) the design and [TENSOR-SPLIT](docs/TENSOR-SPLIT.md) the tensor split.

## Server and API

| Routes | API |
|---|---|
| `/v1/chat/completions`, `/v1/completions`, `/v1/models` | OpenAI-compatible, whole or streamed, with log-probabilities; a reasoning model's thinking comes in `reasoning_content` apart from the answer, and `chat_template_kwargs` sets template options such as `enable_thinking` |
| `/v1/generate`, `/v1/chat`, `/v1/tokenize`, `/v1/detokenize`, `/v1/health` | Native, with token ids |

Tool calls, embeddings and more than one choice per request are not supported.
There is no TLS or authentication, so put a reverse proxy in front of a server that faces a network.
Finished conversations are kept for prefix reuse in device memory, then host memory (`--host-cache-bytes`) and optionally on disk (`--disk-cache-bytes`, `--disk-cache-dir`, `--disk-cache-keep`, `--disk-cache-max-age`; [DISK-TIER](docs/DISK-TIER.md)).
[USAGE](docs/USAGE.md) lists the request fields and server flags, [OPERATING](docs/OPERATING.md) covers running a server (setups, reading `/v1/health`, troubleshooting) and [SERVER](docs/SERVER.md) the scheduler's design.

## Speed options

| Option | What it does |
|---|---|
| `--drafter lookup\|embedded\|PATH`, `--draft-max N` | Speculative decoding in `generate`, `chat` and `serve`: drafts from the prompt and reply (`lookup`), from the MTP block a `qwen35` file carries (`embedded`) or from a drafter file, sources tried in the order listed (`lookup,embedded`), a file of its MTP blocks or in `generate` and `chat` a draft model; the text is identical with it on or off ([SPECULATIVE](docs/SPECULATIVE.md)) |
| `--cache-type-k`, `--cache-type-v` | KV cache storage, `f16` (default) or `f32` |
| `--load-mode auto\|mapped\|direct` | How the weights are read from disk |

## Documentation

- [USAGE](docs/USAGE.md): every command and flag.
- [OPERATING](docs/OPERATING.md): running a server.
- [BUILD](docs/BUILD.md): building on each platform, and running the tests.
- [ROADMAP](docs/ROADMAP.md): what is planned.
- [ARCHITECTURE](docs/ARCHITECTURE.md), [SERVER](docs/SERVER.md), [VULKAN](docs/VULKAN.md) and [MULTI-DEVICE](docs/MULTI-DEVICE.md): how the runtime, the server, the Vulkan backend and the multi-device modes work.
- [AGENTS](AGENTS.md): for contributors.
