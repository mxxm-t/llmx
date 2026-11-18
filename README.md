# llmx

[![CI](https://github.com/mxxm-t/llmx/actions/workflows/ci.yml/badge.svg)](https://github.com/mxxm-t/llmx/actions/workflows/ci.yml)

llmx is a dependency-free C++17 inference runtime for supported x86-64 CPUs, Vulkan devices (AMD and NVIDIA GPUs) and, on Linux, AMD GPUs through ROCm and NVIDIA GPUs through CUDA (in development).
Run it from the command line or as an OpenAI-compatible server, read images with a vision encoder file, and split one model across devices in one machine.

The optional [chat companion](tools/chat_ui/README.md) brings conversations,
monitoring, code previews and a managed file workspace into one owner interface.
It includes branches, MCP tools, request statistics and optional isolated execution.
It uses Python's standard library and static browser files, with no inference
runtime dependencies. Run `python tools/chat_ui/server.py --upstream http://127.0.0.1:8090`
and open `http://127.0.0.1:8183` after starting the llmx server.

The tables summarize current model, weight-format and backend support. **Supported** means the path is implemented and has applicable checks. **In development** means the implementation exists but qualification remains open. **Planned** means roadmap work not available in this tree; see the [ROADMAP](docs/ROADMAP.md).

## Supported models

| Architecture | Models tested | CPU | Vulkan | ROCm | `llmx serve` |
|---|---|---|---|---|---|
| `qwen3` (dense) | Qwen3-0.6B, 8B, 14B, 32B | Supported | Supported | Supported | Supported |
| `qwen3moe` (mixture of experts) | Qwen3-30B-A3B, Qwen3-235B-A22B | Supported | Supported | Supported | Supported |
| `qwen35` (hybrid linear and full attention) | Qwen3.5-0.8B, 4B, 9B, Qwen3.6-27B, Qwen3.8-27B | Supported | Supported | Supported | Supported, prefix reuse for follow-up turns |
| `qwen35moe` (Qwen 3.5 mixture of experts) | Qwen3.6-35B-A3B | Supported | Supported | Supported | Supported, prefix reuse for follow-up turns |
| `qwen4exp` (hybrid with per-layer n-gram embeddings, [details](docs/QWEN38-FLASH.md)) | Qwen3.8-Flash-Next Q8_0 (released-file HF) and UD-Q4_K_XL (bounded serving smoke with Q8_0 MTP sidecar) | Released-file CPU qualification pending | Supported | Supported | Supported, prefix reuse for follow-up turns |
| `deepseek41` (hybrid compressed sparse attention, mixture of experts, lookup-table memory, [details](docs/DEEPSEEK4.md)) | DeepSeek-V4.1-Flash-1017 MXFP4 | Supported | Supported | Supported | Supported |
| Llama, Mistral, Gemma, Phi, DeepSeek-V4-Flash | | Planned | Planned | Planned | Planned |

On ROCm, `qwen3` is checked with Qwen3-0.6B and 8B Q8_0, `qwen35` with Qwen3.5-0.8B Q4_K_M, `qwen4exp` with Qwen3.8-Flash-Next UD-Q4_K_XL and `deepseek41` with DeepSeek-V4.1-Flash-1017 MXFP4 over ten devices.
`qwen3moe` and `qwen35moe` pass their tiny Hugging Face fixtures on ROCm and run Qwen3-30B-A3B and Qwen3.6-35B-A3B Q4_K_M there, though those two files are not yet checked against Hugging Face on ROCm.
A mixture-of-experts file whose expert gate and up weights differ in type is refused on ROCm.
On CUDA, which is in development, `qwen3`, `qwen3moe`, `qwen35`, `qwen35moe` and `deepseek41` run on one device, checked on an RTX 3060 and an RTX 3090: Qwen3-0.6B Q8_0 passes the Hugging Face checks at `--dtype f16`, including 16k-token sequences, Qwen3.5-0.8B Q8_0 passes the `qwen35` Hugging Face checks at `--dtype f16`, Qwen3.6-27B Q4_K_M runs with drafts on and off, Qwen3-30B-A3B Q4_K_M runs at `f16` and `int8`, and the synthetic models pass at `--dtype f32`.
`int8` runs and holds the frozen int8 budget on the synthetic models, while the Qwen3-0.6B and Qwen3.5-0.8B Q8_0 baselines each miss one top-5 tie at `int8`.
Layer splits and tensor groups on `cuda:N` have run once, on an RTX 3090 pair without peer access, so with host writes and pulls only; see [CUDA](docs/CUDA.md).
A model runs on ROCm only when every weight format it holds is one the ROCm backend reads (below); image input (`--mmproj`) runs there as on Vulkan.

Selected outputs are checked against Hugging Face references; [ASSETS](docs/ASSETS.md) lists the models and checks covered.

## Image input

| What | Status |
|---|---|
| Models | `qwen35`, `qwen35moe` and `qwen4exp` files, and DeepSeek-V4.1-Flash's `-vl` `deepseek41` files, with the `mmproj` vision encoder file that ships beside them |
| Commands | `generate --chat` and `chat`, with `--mmproj FILE --image A,B` and `--image-max-tokens N`; PNG and JPEG |
| Server | `llmx serve --mmproj FILE`; `/v1/chat` and `/v1/chat/completions` read `image_url` parts as `data:` URLs |
| Devices | The encoder runs on the CPU, a Vulkan device or a ROCm device, the one that holds the embedding |

See [VISION](docs/VISION.md) and [USAGE](docs/USAGE.md).

## Quantization types

This is the current model-weight support matrix. The [storage metadata page](docs/src/quant-types.md) lists recognized GGUF layouts, including layouts that have no inference decoder.

| Weight format | CPU inference | Vulkan inference | ROCm inference | Written by `llmx quantize` |
|---|---|---|---|---|
| `F32`, `F16`, `BF16` | Supported | Supported | Supported | No |
| `MXFP4` | Supported | Supported | Supported | No |
| `NVFP4` | Supported | Supported | Supported | No |
| `Q1_0` | Supported | Supported | Planned | No |
| `Q2_0` | Supported | Supported | Planned | No |
| `Q2_K`, `Q3_K` | Supported | Supported | Supported | No |
| `Q4_0`, `Q4_1`, `Q4_K` | Supported | Supported | Supported | `Q4_0` |
| `Q5_0`, `Q5_1`, `Q5_K` | Supported | Supported | Supported | No |
| `Q6_K` | Supported | Supported | Supported | No |
| `Q8_0` | Supported | Supported | Supported | `Q8_0` |
| `IQ1_S`, `IQ1_M` | Supported | Supported | Planned | No |
| `IQ2_XXS`, `IQ2_XS`, `IQ2_S` | Supported | Supported | Supported | No |
| `IQ3_XXS`, `IQ3_S` | Supported | Supported | Supported | No |
| `IQ4_NL`, `IQ4_XS` | Supported | Supported | Supported | No |
| `TQ1_0`, `TQ2_0` | Supported | Supported | Planned | No |
| `F8_E4M3` | Supported | Supported | Planned | No |

The current Vulkan `MXFP4` kernels require support for preserving small and special floating-point values and for double-precision intermediate calculations. A tensor assigned to a device that cannot process its format, such as a format the ROCm column does not list, is rejected; llmx does not move it to the CPU automatically.

`NVFP4` is read-only and uses software decoding. Vulkan may use F16-rounded tile inputs on capable devices and falls back to F32 inputs otherwise. ROCm runs `NVFP4`, `Q2_K`, `Q3_K`, `IQ4_XS` and the `IQ2` and `IQ3` types at exact F32 for every `--dtype`.

GGUF metadata can recognize layouts without an inference decoder; `info`, `tokenize` and `detokenize` accept those known layouts. `Q8_1` and `Q8_K` are auxiliary formats. See the [storage type list](docs/src/quant-types.md); metadata recognition alone does not make a weight executable.

TQ read-only support is qualified with a user-approved baseline exception for one position in the two pinned released files' default Vulkan chat captures; original miss counts and the generic 0.1 rule remain unchanged. See [precision policy](docs/PRECISION.md#independent-hf-depth-acceptance-agreed-2026-10-01).

Activation `--dtype` is independent of weight storage and KV-cache type.
It sets the form of the inputs to the weight matrix products, never the weights, which are decoded exactly.
`auto` is the default, and `int8` is never chosen by `auto`.
A device that lacks the requested dtype runs the nearest wider form the table gives and prints a warning in the `dtype:` line on stderr.

| `--dtype` | CPU | Vulkan | ROCm | CUDA (in development) |
|---|---|---|---|---|
| `auto` | `f16` | `f16` | `f16` | `f16` |
| `f32` | Native | Native | Native | Native |
| `f16` | Native: 16-bit integer blocks where a kernel has them, F32 products elsewhere | Native: 16-bit integer blocks on most types, F32 products for the `Q1_0`, `Q2_0`, `Q2_K` and `Q3_K` rows of devices without the integer dot and for some `IQ` and `TQ` types | Native: 16-bit integer blocks for `Q8_0`, `Q4_0`, `Q4_1`, `Q4_K`, `Q5_K`, `Q6_K`, `Q5_0`, `Q5_1`, `IQ4_NL` and `MXFP4`, binary16 inputs for `F16` weights' prompt tiles, F32 products for `F32` and `BF16` weights and `F16` weights' generated rows | Native, in ROCm's forms |
| `bf16` | Emulated: inputs rounded to BF16, F32 arithmetic | Emulated on devices that keep F32 denormals, signed zeros, infinities and NaNs; others run `f32` with a warning | Runs `f32` with a warning | Runs `f32` with a warning |
| `int8` | Runs `f16` with a warning | Native on devices that prefer the integer dot product, today the MI50 under Mesa; others run `f16` with a warning | Native: 8-bit integer blocks for the types with 16-bit integer blocks above, `F16` weights as under `f16`, F32 products for `F32` and `BF16` weights | Native: 8-bit integer rows on the four-way dot and prompt tiles on the integer tensor cores; routed tiles on the four-way dot |

Not every weight type has every form, and `--dtype` prints the forms each device uses; see [USAGE](docs/USAGE.md#precision).

Matrix products take 16-bit integer inputs under the default and 8-bit under `int8`, and every sum is F32.
Attention, norms, RoPE, residuals and routers stay F32, and the KV cache is `f16` unless `--cache-type-k` and `--cache-type-v` say `f32`, or `native` on a model that declares its own cache format, where that is the default.
The full table of what each part computes in is in [PRECISION](docs/PRECISION.md#what-each-part-computes-in).

A model that declares YaRN rope scaling runs it as declared, and `--yarn F` (with `--yarn-original-ctx N`) turns it on for any model, the context becoming F times the trained length. See [rope scaling](docs/USAGE.md#rope-scaling).

## File formats

| Format | Status |
|---|---|
| GGUF v3, one file or a shard set | Supported: read by every model command, written by `quantize` |
| Hugging Face download | Supported: `llmx pull <owner/repo>:<quant>` for GGUF, `llmx pull <owner/repo>` for a safetensors directory |
| Raw F32 tensors (`.json` + `.bin`) | Supported: input of `quantize`, output of `dequantize` |
| safetensors, with `config.json` and `tokenizer.json` | Supported: dense `qwen3` directories in `F32`, `F16` or `BF16`; AWQ 4-bit directories read as exact `Q5_0`; GPTQ and compressed-tensors int directories read as exact `Q4_0` (4-bit symmetric), `Q5_0` (other 4-bit) or `Q8_0` (8-bit symmetric); FP8 directories (`quant_method` fp8 or compressed-tensors `float-quantized`, by tensor, channel or 128 x 128 block) read as `F8_E4M3`, which runs on the CPU and Vulkan; quantize, dequantize, drafter and mmproj files stay GGUF |

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

Add `-DLLMX_HAS_BACKEND_VULKAN=ON` to the first command for the Vulkan backend, and on Linux `-DLLMX_HAS_BACKEND_ROCM=ON` for the ROCm backend, which needs the ROCm SDK to build and the HIP runtime to run on a device, or `-DLLMX_HAS_BACKEND_CUDA=ON` for the CUDA backend, which needs the CUDA toolkit to build and the NVIDIA driver to run on a device.
The binary is `build/llmx`, or `build\Release\llmx.exe` with Visual Studio; on Windows, `build.bat` builds a CPU-only `llmx.exe` without CMake.
[BUILD](docs/BUILD.md) lists the compilers and packages for each platform and a Docker image.

## Run a model

```sh
model=$(./build/llmx pull Qwen/Qwen3-0.6B-GGUF:Q8_0)
./build/llmx generate "$model" "The capital of France is" --temp 0 -n 32
./build/llmx chat "$model" -n 512 --device vulkan:0
./build/llmx serve "$model" --no-auth
```

`pull` needs curl 8.4 or later and reads `HF_TOKEN` for gated repositories; the server listens on `127.0.0.1:8080` and needs a key file (`--api-key-file`) or `--no-auth`; requests send a key as `Authorization: Bearer KEY`.
`llmx bench --model "$model"` measures prefill and decode speed on your hardware.
`llmx <command> --help` shows a command's options, and [USAGE](docs/USAGE.md) covers every command and flag.

## Backends and devices

| Backend | Hardware | Systems | Status |
|---|---|---|---|
| CPU | x86-64 with AVX2, FMA and F16C | Windows, Linux, Intel macOS | Supported |
| Vulkan | Vulkan 1.2 GPUs; tested on AMD Radeon VII, AMD MI50 and NVIDIA GeForce RTX 3060 | Windows, Linux | Supported |
| ROCm (HIP) | AMD GPUs; tested on AMD MI50 (gfx906), built also for AMD MI300X (gfx942) | Linux | Supported |
| CUDA | NVIDIA GPUs; tested on NVIDIA GeForce RTX 3060, RTX 3090 and RTX A6000 | Linux | In development: `qwen3`, `qwen3moe`, `qwen35`, `qwen35moe` and `deepseek41` with `F32`, `F16` and `int8` activations on one device, and layer splits and tensor groups once on an RTX 3090 pair; see [CUDA](docs/CUDA.md) |
| SYCL | Intel GPUs | | Planned |

ARM CPUs, Apple Silicon included, are not supported.
The Vulkan kernels are tuned for AMD GPUs' 64-wide subgroups; they run on NVIDIA GPUs, tested on the one card named above, and other GPUs are untested.
The ROCm kernels are built for gfx906, gfx942 (MI300X) and gfx1151 (Strix Halo), and only gfx906 has run them: MI300X and Strix Halo are untested and not checked against Hugging Face.
Pick a device with `--device cpu` (the default), `--device vulkan:N`, `--device rocm:N` or, in a CUDA build, `--device cuda:N`; see [USAGE](docs/USAGE.md#device-selection---device).

## Multi-device modes

| Mode | What it does | Flags | Status |
|---|---|---|---|
| Layer split | Consecutive layers on different devices, GPUs and the CPU mixed, fitted to free memory, for every supported model in every command and `llmx serve` | `--device A,B,...`, `--layer-shares` | Supported |
| Expert offload | A mixture-of-experts model's experts are held in host memory and run on the CPU beside one GPU or a layer split over several GPUs; a faster CPU path for prompts and decode steps | `--n-cpu-moe N`, `--cpu-moe`, `--moe-stream-from N` | Supported |
| Tensor split | Every layer on a group of 2 to 16 devices at once, with ROCm devices paired into groups by timing; dense `qwen3`, hybrid `qwen35` and `qwen4exp`, mixture-of-experts `qwen3moe` and `qwen35moe` and `deepseek41` models, with their embedded drafter, on Vulkan and ROCm devices on Linux; groups wider than 4 are not yet measured on devices | `--tensor-width N` | Supported; ahead of or below the layer split by width, precision and user count ([USAGE](docs/USAGE.md#tensor-split---tensor-width-n)) |
| Staged tensor split | A layer split whose stages are tensor groups | `--tensor-width N` with several groups listed in `--device`; `--layer-shares` gives each group its share | Supported, as above |
| Replicas | Several copies of a model behind one scheduler | | Planned |
| Multi-node | One model over several machines | | Planned |

A split prices a long prompt's slices from its own measured stage times, and from the third prefill after a load on, however cold the first one ran.

[USAGE](docs/USAGE.md#several-devices---device-ab---layer-shares) covers the flags, [MULTI-DEVICE](docs/MULTI-DEVICE.md) the design and [TENSOR-SPLIT](docs/TENSOR-SPLIT.md) the tensor split.

## Server and API

| Routes | API |
|---|---|
| `/v1/chat/completions`, `/v1/completions`, `/v1/models` | OpenAI-compatible, whole or streamed, with log-probabilities; a reasoning model's thinking comes in `reasoning_content` apart from the answer, and `chat_template_kwargs` sets template options such as `enable_thinking` |
| `/v1/generate`, `/v1/chat`, `/v1/tokenize`, `/v1/detokenize`, `/v1/health` | Native, with token ids |

Chat requests take `tools` and `tool_choice` and return `tool_calls`, and `/v1/live` answers while a server starts.
Embeddings and more than one choice per request are not supported.
There is no TLS, so put a reverse proxy in front of a server that faces a network; every route but `/v1/live` needs a key ([USAGE](docs/USAGE.md)).
Finished conversations are kept for prefix reuse in device memory, then host memory (`--host-cache-bytes`) and optionally on disk (`--disk-cache-bytes`, `--disk-cache-dir`, `--disk-cache-keep`, `--disk-cache-max-age`; [DISK-TIER](docs/DISK-TIER.md)).
[USAGE](docs/USAGE.md) lists the request fields and server flags, [OPERATING](docs/OPERATING.md) covers running a server (setups, reading `/v1/health`, troubleshooting) and [SERVER](docs/SERVER.md) the scheduler's design.

## Speed options

| Option | What it does |
|---|---|
| `--drafter lookup\|embedded\|PATH`, `--draft-max N` | Speculative decoding in `generate`, `chat` and `serve`: drafts from the prompt and reply (`lookup`), from the MTP block a `qwen35` file carries (`embedded`), from a DeepSeek-V4.1 DSpark sidecar file or from a drafter file, sources tried in the order listed (`lookup,embedded`), a file of its MTP blocks or in `generate` and `chat` a draft model; the text is identical with it on or off ([SPECULATIVE](docs/SPECULATIVE.md)) |
| `--dtype auto\|f16\|bf16\|f32\|int8` | Activation precision, independent of weight storage; see Quantization types above |
| `--cache-type-k`, `--cache-type-v` | KV cache storage, `f16` (default) or `f32`, or `native` where the model declares a cache format (the default there) |
| `--load-mode auto\|mapped\|direct` | How the weights are read from disk |

## Documentation

- [USAGE](docs/USAGE.md): every command and flag.
- [OPERATING](docs/OPERATING.md): running a server.
- [BUILD](docs/BUILD.md): building on each platform, and running the tests.
- [ROADMAP](docs/ROADMAP.md): what is planned.
- [ARCHITECTURE](docs/ARCHITECTURE.md), [SERVER](docs/SERVER.md), [VULKAN](docs/VULKAN.md), [ROCM](docs/ROCM.md) and [MULTI-DEVICE](docs/MULTI-DEVICE.md): how the runtime, the server, the Vulkan and ROCm backends and the multi-device modes work.
- [CHAT-LIVE-METRICS](docs/CHAT-LIVE-METRICS.md): the chat companion's live request metrics.
- [AGENTS](AGENTS.md): for contributors, and [TESTS](docs/TESTS.md): what each test checks.
