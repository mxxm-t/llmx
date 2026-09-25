# llmx

[![CI](https://github.com/mxxm-t/llmx/actions/workflows/ci.yml/badge.svg)](https://github.com/mxxm-t/llmx/actions/workflows/ci.yml)

llmx is an LLM inference runtime written in C++17 that links no third-party libraries.
It runs Qwen3 dense and mixture-of-experts models from GGUF files on x86-64 CPUs and on GPUs through Vulkan, from the command line or as an HTTP server with OpenAI-compatible routes.
It downloads models from Hugging Face and can split a model by layers across the GPUs and CPU of one machine.

## What it runs

| | Supported |
|---|---|
| Models | Qwen3 dense (`qwen3`) and mixture of experts (`qwen3moe`), such as Qwen3-0.6B, 8B, 32B, 30B-A3B and 235B-A22B |
| Files | GGUF v3, one file or a shard set, with Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K and F32 tensors, such as Q8_0, Q4_0, Q4_K_M and Q5_K_M files |
| CPU | x86-64 with AVX2, FMA and F16C |
| GPU | Vulkan 1.2 GPUs, tested on an AMD Radeon VII on Windows and AMD MI50 cards on Linux |
| Several devices | A model split by layers over the GPUs and CPU of one machine, or a mixture-of-experts model whose experts run on the CPU beside a GPU |
| Systems | Windows and Linux on x86-64, and Intel macOS with the CPU backend only |

Not supported:

- ARM CPUs, Apple Silicon included, and the Vulkan backend on macOS.
- GPU backends other than Vulkan, such as ROCm or CUDA.
- Model architectures other than Qwen3, safetensors files, and tensor types outside the list above, F16 and BF16 included.
- Splitting a single layer across devices, and running one model across several machines.

[ROADMAP](docs/ROADMAP.md) lists what is planned.

## Build

There are no prebuilt binaries.
Build llmx with CMake and a C++17 compiler: Visual Studio 2022 or later on Windows, GCC or Clang on Linux, or Apple Clang on macOS.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

The binary is `build/llmx`, or `build\Release\llmx.exe` with Visual Studio.
To build the Vulkan backend on Windows or Linux, add `-DLLMX_HAS_BACKEND_VULKAN=ON` to the first command; it needs the Vulkan headers and `glslc` to build, and the Vulkan loader and a GPU driver to run.
On Windows, `build.bat` builds a CPU-only `llmx.exe` in the repository root without CMake.
[BUILD](docs/BUILD.md) lists the packages for each platform, the Vulkan setup on [Windows](docs/BUILD.md#the-vulkan-backend-on-windows) and [Linux](docs/BUILD.md#the-vulkan-backend-on-linux), and a Docker image for the Vulkan build.

## Run a model

```sh
model=$(./build/llmx pull Qwen/Qwen3-0.6B-GGUF:Q8_0)
./build/llmx generate "$model" "The capital of France is" --temp 0 -n 32
./build/llmx chat "$model" -n 512
```

`pull` downloads a GGUF file, or a whole shard set, from Hugging Face into `.cache/llmx` in your home directory, verifies it and prints its local path.
It needs curl 8.4 or later, and it reads `HF_TOKEN` for gated repositories.
Every model command also takes a local GGUF file, or the first file of a shard set.
`generate` continues a raw prompt, and `--temp 0` makes it greedy.
`chat` applies the model's chat template, reads one message per line and exits on Ctrl+C.
Its `-n 512` raises the reply limit from the default of 64 tokens, which Qwen3's reasoning alone can use up.
In PowerShell, call `.\build\Release\llmx.exe`, or `.\llmx.exe` after `build.bat`, in place of `./build/llmx`, and write the first line as `$model = .\llmx.exe pull Qwen/Qwen3-0.6B-GGUF:Q8_0`.
`llmx --help` lists the commands, `llmx <command> --help` shows one command's options, and [USAGE](docs/USAGE.md) covers every command and flag.

## Serve

```sh
./build/llmx serve "$model"
# in a second terminal:
curl http://127.0.0.1:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":512}'
```

The server listens on `127.0.0.1:8080` by default, batches concurrent requests together, and streams the reply when a request sets `"stream": true`.
`/v1/chat/completions`, `/v1/completions` and `/v1/models` follow the OpenAI API, so OpenAI client libraries connect with the base URL `http://127.0.0.1:8080/v1`, and the native routes beside them also return token ids.
Tool calls, embeddings and more than one choice per request are not supported.
There is no TLS or authentication, so put a reverse proxy in front of the server before it faces a network.
The `curl` line is for bash or a similar shell, since `curl` in Windows PowerShell is a different command.
The `llmx serve` section of [USAGE](docs/USAGE.md) lists the routes, request fields, limits and server flags.

## Use a GPU

```sh
./build/llmx bench --device vulkan:0
./build/llmx chat "$model" -n 512 --device vulkan:0
```

In a build with the Vulkan backend, `--device vulkan:N` runs on Vulkan device N, counted from 0 in the order the Vulkan loader lists them, which `vulkaninfo --summary` shows.
`--device cpu` is the default, and every command that runs a model takes `--device`, `serve` included.
The first line above runs a synthetic benchmark as a check that the device and the build work.
A list such as `--device vulkan:0,vulkan:1` splits the model by layers over those devices to fit their free memory, the CPU can be one of them, and `--verbose` prints what each device holds.
For a mixture-of-experts model larger than the GPU, `--n-cpu-moe N` runs the experts of the first N layers on the CPU from host memory, and `--cpu-moe` runs all of them there.
With `--n-cpu-moe 12`, Qwen3-30B-A3B Q4_K_M fits a 16 GB card.
The kernels are tuned for the 64-wide subgroups of the Radeon VII and MI50, and other GPUs are untested.
[USAGE](docs/USAGE.md) covers devices, splits and experts on the CPU, and [VULKAN](docs/VULKAN.md) describes the backend's design.

## Correctness and speed

The tests compare token ids, logits, perplexity and chat replies with Hugging Face reference outputs for pinned models.
The performance target is to be at least as fast as [mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp), a llama.cpp fork tuned for gfx906 GPUs such as the Radeon VII and MI50, in prefill and decode on the same model, quantization, prompt and hardware, and for the llmx server to beat that fork's server by a wide margin in time to first token, inter-token latency and throughput under concurrent load.
[STATUS](docs/STATUS.md) records the measured comparisons.
`./build/llmx bench --model "$model" --p 512 --n 128 --r 3` measures prefill and decode speed on your own hardware, and `python3 tools/server_load.py --url http://127.0.0.1:8080` measures a running server.

## Documentation

- [BUILD](docs/BUILD.md): requirements and build commands for each platform, the Vulkan backend, Docker and the tests.
- [USAGE](docs/USAGE.md): every command and flag, the server routes, device selection and splits.
- [ARCHITECTURE](docs/ARCHITECTURE.md): how the code is layered.
- [SERVER](docs/SERVER.md): design of the server's scheduler, batching and prefix reuse.
- [VULKAN](docs/VULKAN.md): design and kernels of the Vulkan backend.
- [MULTI-DEVICE](docs/MULTI-DEVICE.md): design of the layer split and the splits planned after it.
- [EXECUTION](docs/EXECUTION.md), [DEVICE-EXECUTION](docs/DEVICE-EXECUTION.md) and [KV-CACHE](docs/KV-CACHE.md): design of batched execution, device placement and the paged KV cache.
- [ROADMAP](docs/ROADMAP.md): planned models, formats, backends and features.
- [STATUS](docs/STATUS.md): the dated development log, newest first, with measurements.
- [ASSETS](docs/ASSETS.md): the models, corpora and reference fixtures behind the recorded results.
- [CI](docs/CI.md): what each hosted CI job builds and runs.
- [Source notes](docs/src/): what each source file is responsible for.
- [AGENTS](AGENTS.md): rules for contributors, and what each test covers.

## Tests

```sh
ctest --test-dir build -C Release --output-on-failure
python3 tests/run_tests.py --exe build/llmx --no-perf-floor
```

On Windows, run `python` with `--exe build/Release/llmx.exe`.
After `build.bat`, which builds no native tests, run only the Python suite and leave out `--exe`.
The Python suite needs only the standard library, and its real-model checks skip until `python3 tools/fetch_test_models.py` downloads the pinned models.
[BUILD](docs/BUILD.md#after-building-the-tests) has the test commands for each platform, [AGENTS](AGENTS.md#tests) describes each test, and [CI](docs/CI.md) lists what each hosted job runs.
