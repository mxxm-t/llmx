# llmx

[![CI](https://github.com/mxxm-t/llmx/actions/workflows/ci.yml/badge.svg)](https://github.com/mxxm-t/llmx/actions/workflows/ci.yml)

llmx is an LLM inference runtime written in C++17 that links no third-party libraries.
It runs Qwen3, Qwen3-MoE and Qwen 3.5 hybrid models from GGUF files on x86-64 CPUs and on GPUs through Vulkan, from the command line or as an HTTP server with OpenAI-compatible routes.
It downloads models from Hugging Face, and it can run one model over several GPUs and the CPU of one machine as a layer split, or keep a mixture-of-experts model's experts on the CPU with expert offload.

Every item below is marked **Supported**, meaning it is on main and covered by tests, or **Planned**, meaning [ROADMAP](docs/ROADMAP.md) plans it and it is not implemented yet.

## Supported models

An architecture is chosen from the file's `general.architecture` key.

| Architecture | Models tested | CPU | Vulkan | `llmx serve` |
|---|---|---|---|---|
| `qwen3` (dense) | Qwen3-0.6B, 8B, 14B, 32B | Supported | Supported | Supported |
| `qwen3moe` (mixture of experts) | Qwen3-30B-A3B, Qwen3-235B-A22B | Supported | Supported | Supported |
| `qwen35` (dense hybrid: linear attention with a recurrent state, and full attention) | Qwen3.5-0.8B, 4B, 9B, Qwen3.6-27B | Supported | Supported | Planned: refused before the server listens until its scheduler holds a recurrent state |
| `qwen35moe` (Qwen 3.5 mixture of experts) | | Planned | Planned | Planned |
| Llama, Mistral, Gemma, Phi, DeepSeek V4 | | Planned | Planned | Planned |

Correctness is checked against Hugging Face reference outputs: token ids, logits, perplexity and chat replies on pinned Qwen3-0.6B, Qwen3-8B and Qwen3.5-0.8B files and on tiny random-weight models of each architecture, and on Qwen3.5-4B, Qwen3.5-9B and Qwen3.6-27B by hand.
[ASSETS](docs/ASSETS.md) lists the pinned files and what each check covers.

## Quantization types

The type of each tensor in a GGUF file, and where it runs.
Real files mix types: a Q4_K_M file holds Q4_K, Q6_K and F32 tensors, and llmx reads it when it reads every type in it.

| Type | CPU | Vulkan | Written by `llmx quantize` |
|---|---|---|---|
| F32 | Supported | Supported | (input) |
| Q8_0 | Supported | Supported | Yes |
| Q4_0 | Supported | Supported | Yes |
| Q4_1 | Supported | Supported | No |
| Q4_K, Q5_K, Q6_K (and so Q4_K_M and Q5_K_M files) | Supported | Supported | No |
| MXFP4 | Supported | Planned: refused at load | No |
| F16, BF16 | Planned | Planned | No |
| IQ4_NL, IQ4_XS | Planned | Planned | No |
| Q3_K, Q2_K | Planned | Planned | No |

A device that cannot run a type refuses the file as it loads, before any weight is uploaded, and never falls back to the CPU on its own.
The KV cache is stored as `f16` (the default) or `f32` on every backend, chosen per side with `--cache-type-k` and `--cache-type-v`.

## File types and formats

| Format | Status |
|---|---|
| GGUF v3, one file | Supported: every model command reads it, `quantize` writes it, `info` lists its metadata and tensors |
| GGUF shard sets (`-0000N-of-0000M.gguf`) | Supported: pass the first file |
| Hugging Face download | Supported: `llmx pull <owner/repo>:<quant>` fetches a file or a whole shard set, verified, into a local cache |
| Raw F32 tensors (`model.json` + `model.bin`) | Supported: input of `quantize`, output of `dequantize` |
| safetensors directories, with `config.json` and `tokenizer.json` | Planned |
| ONNX export | Planned |

The tokenizer is the byte-level BPE a GGUF file embeds, and chat templates are rendered from the file's Jinja template as the Hugging Face reference renders them.

## Backends and devices

| Backend | Hardware | Systems | Status |
|---|---|---|---|
| CPU | x86-64 with AVX2, FMA and F16C | Windows, Linux, Intel macOS | Supported |
| Vulkan | Vulkan 1.2 GPUs; tested on an AMD Radeon VII on Windows and AMD MI50 cards on Linux | Windows, Linux | Supported |
| ROCm (HIP) | AMD GPUs | Linux | Planned |
| CUDA | NVIDIA GPUs | | Planned |
| SYCL | Intel GPUs | | Planned |

ARM CPUs, Apple Silicon included, and Vulkan on macOS are not supported.
The Vulkan kernels are tuned for the 64-wide subgroups of the Radeon VII and MI50, and other GPUs are untested.
`--device cpu` is the default, `--device vulkan:N` picks Vulkan device N in the order the loader lists them (`vulkaninfo --summary`), and every command that runs a model takes `--device`.

## Multi-device modes

| Mode | What it does | Flags | Status |
|---|---|---|---|
| Layer split | Consecutive layers on different devices, GPUs and the CPU mixed, fitted to each device's free memory or set by proportions; bit-identical to one device over identical devices; a prompt is pipelined over the stages, and the server keeps a pass in flight per stage | `--device A,B,...`, `--layer-shares A,B,...` | Supported |
| Expert offload | A mixture-of-experts model on one GPU keeps the experts of some or all routed layers in host memory and runs them on the CPU; long prompts can stream those experts to the GPU instead | `--n-cpu-moe N`, `--cpu-moe`, `--moe-stream-from N` | Supported for `qwen3moe` on one device, not combined with a layer split |
| Tensor split | Every layer on a tensor group of 2 to 4 devices at once, for faster decode of one request | | Planned |
| Staged tensor split | A layer split whose stages are tensor groups | | Planned |
| Replicas | Several copies of a model behind one scheduler | | Planned |
| Multi-node | One model over several machines | | Planned |

For example, Qwen3-32B Q8_0 runs as a layer split over two MI50s, Qwen3-235B-A22B Q4_K_M over six, and Qwen3-30B-A3B Q4_K_M fits a 16 GB Radeon VII with expert offload of twelve layers (`--n-cpu-moe 12`).
[USAGE](docs/USAGE.md#several-devices---device-ab---layer-shares) covers the flags and [MULTI-DEVICE](docs/MULTI-DEVICE.md) the design of every mode.

## Server and API

`llmx serve` runs one model for many users: requests are batched together in each pass, replies stream as they are sampled, a new prompt reuses the KV blocks of a finished request that shares its prefix, and requests that outgrow the KV pool are paused and resumed with the reply they would have given unpaused.
A greedy or seeded request gives the same tokens as `llmx generate` with the same settings, alone or beside other requests.

| Route | API | Status |
|---|---|---|
| `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models` | OpenAI-compatible, whole or streamed, with log-probabilities | Supported |
| `POST /v1/generate`, `POST /v1/chat` | Native, with token ids, whole or streamed, with log-probabilities | Supported |
| `POST /v1/tokenize`, `POST /v1/detokenize`, `GET /v1/health` | Native | Supported |

`qwen35` models are refused by the server until its serving step lands; the other commands run them.
Tool calls, embeddings and more than one choice per request are not supported.
There is no TLS or authentication, so put a reverse proxy in front of the server before it faces a network.
The `llmx serve` section of [USAGE](docs/USAGE.md) lists the request fields, limits and server flags, and [SERVER](docs/SERVER.md) the scheduler's design.

## Build

There are no prebuilt binaries.
Build llmx with CMake and a C++17 compiler: Visual Studio 2022 or later on Windows, GCC or Clang on Linux, or Apple Clang on macOS.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

The binary is `build/llmx`, or `build\Release\llmx.exe` with Visual Studio.
Add `-DLLMX_HAS_BACKEND_VULKAN=ON` to the first command for the Vulkan backend on Windows or Linux; it needs the Vulkan headers and `glslc` to build, and the Vulkan loader and a GPU driver to run.
On Windows, `build.bat` builds a CPU-only `llmx.exe` in the repository root without CMake.
[BUILD](docs/BUILD.md) lists the packages for each platform, the Vulkan setup and a Docker image for the Vulkan build.

## Run a model

```sh
model=$(./build/llmx pull Qwen/Qwen3-0.6B-GGUF:Q8_0)
./build/llmx generate "$model" "The capital of France is" --temp 0 -n 32
./build/llmx chat "$model" -n 512 --device vulkan:0
./build/llmx serve "$model"
curl http://127.0.0.1:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":512}'
```

`pull` downloads into `.cache/llmx` in your home directory and prints the local path; it needs curl 8.4 or later and reads `HF_TOKEN` for gated repositories.
`chat` reads one message per line, and `-n 512` raises the reply limit from the default of 64 tokens, which Qwen3's reasoning alone can use up.
The server listens on `127.0.0.1:8080`; run `curl` from a second terminal in bash or a similar shell.
In PowerShell, call `.\build\Release\llmx.exe` in place of `./build/llmx` and write the first line as `$model = .\build\Release\llmx.exe pull Qwen/Qwen3-0.6B-GGUF:Q8_0`.
`llmx --help` lists the commands and `llmx <command> --help` one command's options; [USAGE](docs/USAGE.md) covers every command and flag.

## Correctness and speed

Correctness is the Hugging Face reference, as above.
The performance target is to be at least as fast as [mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp), a llama.cpp fork tuned for gfx906 GPUs such as the Radeon VII and MI50, on the same model, quantization, prompt and hardware, and for the server to beat that fork's server by a wide margin under concurrent load.
[STATUS](docs/STATUS.md) records the measured comparisons.
`./build/llmx bench --model "$model" --p 512 --n 128 --r 3` measures prefill and decode speed on your own hardware, and `python3 tools/server_load.py --url http://127.0.0.1:8080` measures a running server.

## Tests

```sh
ctest --test-dir build -C Release --output-on-failure
python3 tests/run_tests.py --exe build/llmx --no-perf-floor
```

On Windows, run `python` with `--exe build/Release/llmx.exe`; after `build.bat`, which builds no native tests, run only the Python suite without `--exe`.
Real-model checks skip until `python3 tools/fetch_test_models.py` downloads the pinned models.
[BUILD](docs/BUILD.md#after-building-the-tests) has the test commands for each platform, [AGENTS](AGENTS.md#tests) describes each test, and [CI](docs/CI.md) lists what each hosted job runs.

## Documentation

- [BUILD](docs/BUILD.md): requirements and build commands for each platform, the Vulkan backend, Docker and the tests.
- [USAGE](docs/USAGE.md): every command and flag, the server routes, devices, the layer split and expert offload.
- [ARCHITECTURE](docs/ARCHITECTURE.md): how the code is layered.
- [SERVER](docs/SERVER.md): the server's scheduler, batching and prefix reuse.
- [VULKAN](docs/VULKAN.md): the Vulkan backend's design and kernels.
- [MULTI-DEVICE](docs/MULTI-DEVICE.md): the layer split, the tensor split and the other multi-device modes.
- [EXECUTION](docs/EXECUTION.md), [DEVICE-EXECUTION](docs/DEVICE-EXECUTION.md) and [KV-CACHE](docs/KV-CACHE.md): batched execution, device placement and the paged KV cache.
- [QWEN35](docs/QWEN35.md): the Qwen 3.5 architectures (`qwen35` and `qwen35moe`).
- [ROADMAP](docs/ROADMAP.md): planned models, formats, backends and features.
- [STATUS](docs/STATUS.md): the dated development log, newest first, with measurements.
- [ASSETS](docs/ASSETS.md): the models, corpora and reference fixtures behind the recorded results.
- [CI](docs/CI.md): what each hosted CI job builds and runs.
- [Source notes](docs/src/): what each source file is responsible for.
- [AGENTS](AGENTS.md): rules for contributors, and what each test covers.
