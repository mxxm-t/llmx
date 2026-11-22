# llmx

[![CI](https://github.com/mxxm-t/llmx/actions/workflows/ci.yml/badge.svg)](https://github.com/mxxm-t/llmx/actions/workflows/ci.yml)

llmx runs large language models on your own machine: on the CPU, on AMD and NVIDIA GPUs through Vulkan, and on AMD GPUs through ROCm on Linux; NVIDIA GPUs through CUDA are in development.
It reads GGUF files and dense Qwen3 and Qwen 3.5 safetensors directories, plain or quantized by AWQ, GPTQ, compressed-tensors or FP8, downloads them from Hugging Face, chats in the terminal and serves an OpenAI-compatible API, and it splits one model over several GPUs when it does not fit on one.
It is one program: `llmx pull` needs curl 8.4 or later, and a GPU needs its driver.

## Quick start

Download the archive for your system from the [releases page](https://github.com/mxxm-t/llmx/releases): `llmx-X.Y.Z-linux-x86_64.tar.gz` (CPU, Vulkan and ROCm) or `llmx-X.Y.Z-windows-x64.zip` (CPU and Vulkan).
The CPU must be x86-64 with AVX2, FMA and F16C: Intel Haswell (2013), AMD Excavator (2015) or later.

On Linux:

```sh
tar -xzf llmx-X.Y.Z-linux-x86_64.tar.gz && cd llmx-X.Y.Z-linux-x86_64
model=$(./llmx pull Qwen/Qwen3-0.6B-GGUF:Q8_0)   # download a model
./llmx chat "$model" -n 512                      # chat in the terminal, on the CPU
./llmx chat "$model" -n 512 --device vulkan:0    # the same on the first GPU
./llmx serve "$model" --no-auth                  # an API at http://127.0.0.1:8080/v1
```

On Windows, in PowerShell:

```powershell
Expand-Archive llmx-X.Y.Z-windows-x64.zip -DestinationPath .
cd llmx-X.Y.Z-windows-x64
$model = .\llmx.exe pull Qwen/Qwen3-0.6B-GGUF:Q8_0
.\llmx.exe chat $model -n 512
.\llmx.exe chat $model -n 512 --device vulkan:0
.\llmx.exe serve $model --no-auth
```

Without `--device` llmx runs on the CPU; `--device vulkan:0` or `--device rocm:0` picks a GPU.
Terminal chat defaults to 64 tokens per reply. Use `-n` to change the limit.
The native API also defaults to 64. On the OpenAI-compatible endpoints, set `max_tokens` to limit a reply or omit it for an uncapped reply.
Without curl 8.4, download a GGUF file yourself and pass its path.

## Models

| Architecture | Models | CPU | Vulkan | ROCm | `llmx serve` |
|---|---|---|---|---|---|
| `qwen3` | Qwen3 0.6B to 32B | Supported | Supported | Supported | Supported |
| `qwen3moe` | Qwen3-30B-A3B | Supported | Supported | Supported | Supported |
| `qwen35` | Qwen3.5 0.8B to 9B, Qwen3.6-27B, Qwen3.8-27B | Supported | Supported | Supported | Supported |
| `qwen35moe` | Qwen3.6-35B-A3B | Supported | Supported | Supported | Supported |
| `qwen4exp` ([details](docs/QWEN38-FLASH.md)) | Qwen3.8-Flash-Next | In development | Supported | Supported | Supported |
| `deepseek41` ([details](docs/DEEPSEEK4.md)) | DeepSeek-V4.1-Flash-1017 | In development | Supported | Supported | Supported |
| Llama, Mistral, Gemma, Phi, DeepSeek-V4-Flash | | Planned | Planned | Planned | Planned |

- `llmx serve` reuses the shared beginning of follow-up turns on every architecture.
- Image input (`--mmproj`) works with `qwen35`, `qwen35moe` and `qwen4exp` on the CPU, Vulkan and ROCm, and with DeepSeek-V4.1-Flash's `-vl` files on Vulkan and ROCm ([VISION](docs/VISION.md)).
- On ROCm a model runs when every weight format it holds is in the ROCm column below.

## How fast is it

The speed figures are being measured again on the released code and will be published here.
`llmx bench --model FILE` measures your own machine.

## Running models bigger than your GPU

- List several GPUs, `--device rocm:0,rocm:1`, and llmx splits the layers over them to fit; add `cpu` to the list to keep the last layers in system RAM ([MULTI-DEVICE](docs/MULTI-DEVICE.md)).
- `--cpu-moe` keeps a mixture-of-experts model's experts in system RAM and the rest on the GPU, and `--n-cpu-moe N` only the first N layers' experts; not with `--tensor-width`.
- `--load-mode direct` reads the weights around the system's file cache, which loads a large model faster from some file systems; `auto`, the default, picks for you ([USAGE](docs/USAGE.md#reading-the-weights---load-mode)).

## Install

- **Download** the Linux or Windows archive from the [releases page](https://github.com/mxxm-t/llmx/releases) and check it against `SHA256SUMS`.
  The [nightly pre-release](https://github.com/mxxm-t/llmx/releases/tag/nightly) holds the same files built from the newest `main`, unchecked by a release's smoke runs ([RELEASE](docs/RELEASE.md#the-nightly-channel)).
  The Linux binary needs glibc 2.35 or later; GPUs need the Vulkan driver or the ROCm 7 runtime the system already has.
- **Docker:** the `ghcr.io/mxxm-t/llmx:X.Y.Z-vulkan` and `ghcr.io/mxxm-t/llmx:X.Y.Z-rocm7` images run `llmx`, and `ghcr.io/mxxm-t/llmx:chat-X.Y.Z` the chat companion ([RELEASE](docs/RELEASE.md#the-images)). These examples make the API available locally at `http://127.0.0.1:8080/v1`. For clients on other machines, see the [key-file example](docs/RELEASE.md#what-a-release-holds).

  ```sh
  docker run --rm --device /dev/dri --group-add "$(getent group render | cut -d: -f3)" -v /path/to/models:/models -p 127.0.0.1:8080:8080 \
    ghcr.io/mxxm-t/llmx:X.Y.Z-vulkan serve /models/model.gguf --device vulkan:0 --host 0.0.0.0 --no-auth
  docker run --rm --device /dev/kfd --device /dev/dri --group-add "$(getent group render | cut -d: -f3)" -v /path/to/models:/models -p 127.0.0.1:8080:8080 \
    ghcr.io/mxxm-t/llmx:X.Y.Z-rocm7 serve /models/model.gguf --device rocm:0 --host 0.0.0.0 --no-auth
  ```

### Server and chat companion in containers

On Linux, [docker/compose.yaml](docker/compose.yaml) runs `llmx serve` from the release image and the [chat companion](tools/chat_ui/README.md) built from this checkout, both on the machine's own address `127.0.0.1`.
The server takes a key from `docker/api-key`, which the companion sends for you, and the companion keeps its conversations in a Docker volume:

```sh
head -c 32 /dev/urandom | base64 | tr -d '/+=\n' > docker/api-key && chmod 600 docker/api-key
LLMX_MODELS=/path/to/models LLMX_MODEL=model.gguf docker compose -f docker/compose.yaml --profile vulkan up -d
```

Open `http://127.0.0.1:8183`.
The profile `rocm` in place of `vulkan` serves through ROCm, and the file's comments list the other settings, such as `LLMX_DEVICE` and the ports.
From another computer, forward the port over SSH, `ssh -L 8183:127.0.0.1:8183 HOST`, and open the same address there.
`docker compose -f docker/compose.yaml --profile vulkan down` stops both and keeps the conversations.

### Build

From source, with CMake and a C++ compiler; add `-DLLMX_HAS_BACKEND_VULKAN=ON` or `-DLLMX_HAS_BACKEND_ROCM=ON` to the first line for a GPU ([BUILD](docs/BUILD.md)):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel                   # Linux and Intel macOS: build/llmx
cmake --build build --parallel --config Release  # Windows: build\Release\llmx.exe
```

On Windows `build.bat` builds a CPU-only `llmx.exe` without CMake.
The ROCm and CUDA kernels are compiled for the main GPU architectures from AMD MI50 (gfx906) and NVIDIA V100 (sm_70; sm_75 with CUDA 13) on; `-DLLMX_ROCM_ARCHS="gfx1031"` or `-DLLMX_CUDA_ARCHS="87"` replaces that list ([BUILD](docs/BUILD.md#gpu-architectures)).
Intel macOS builds for the CPU only and has no download.

## Using it

```sh
./llmx generate "$model" "The capital of France is" -n 32   # one completion
./llmx chat "$model" -n 512                                   # a conversation
./llmx serve "$model" --device rocm:0 --api-key-file keys.txt # a server on 127.0.0.1:8080
curl http://127.0.0.1:8080/v1/chat/completions -H "Authorization: Bearer $KEY" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":512}'
```

- The server speaks the OpenAI API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`), streamed or whole, with tool calls, so editors and agents that take an OpenAI endpoint work with it ([OPERATING](docs/OPERATING.md)).
  It needs a key file, `--api-key-file`, whose keys requests send as `Authorization: Bearer KEY`, or `--no-auth` to serve every client; it has no TLS, so put a reverse proxy in front of one that faces a network.
- The [chat companion](tools/chat_ui/README.md) is a browser interface for a running server, with conversations, file workspaces and live request metrics: `python tools/chat_ui/server.py --upstream http://127.0.0.1:8080`, then open `http://127.0.0.1:8183`.
  It needs only Python 3 and comes in the release archives, so the command runs from the extracted directory as from a checkout; add `--api-key-file FILE` with one of the server's keys.
  The [health dashboard](tools/health_dashboard/README.md) alone charts a server's `/v1/health` on `http://127.0.0.1:8182`: `python tools/health_dashboard/server.py --upstream http://127.0.0.1:8080/v1/health --api-key-file FILE`.
- Faster answers: `--drafter` drafts tokens ahead, with the same text as without ([SPECULATIVE](docs/SPECULATIVE.md)).
  `lookup` drafts from the prompt and reply, `embedded` from the MTP block a Qwen 3.5 file carries, and a file path from a DeepSeek DSpark file or another drafter file.
  Lookup combines with the others, `--drafter lookup,embedded` or `--drafter lookup,FILE`: it drafts where the text repeats and the model drafter covers the rest.
- `llmx <command> --help` lists a command's options, and [USAGE](docs/USAGE.md) every command and flag.

### Conversation cache

The server keeps finished conversations, so a follow-up message processes only its new text instead of the whole history again.
On a GPU this pays off once a conversation passes about 512 tokens: a shorter prompt is computed again whole, which keeps the results the same however the prompt arrives.
They stay in GPU memory first, then in system RAM (`--host-cache-bytes`, half the free RAM by default), and on disk if you give it room (`--disk-cache-bytes`, `--disk-cache-dir`).
`--disk-cache-keep` keeps the disk cache across server restarts ([DISK-TIER](docs/DISK-TIER.md)).
`--context-overflow shift` drops the oldest turns when a conversation outgrows the context instead of refusing it.

### Fastest setup on four MI50s

Qwen3.8-Flash-Next on four MI50s with ROCm, as two groups of two GPUs, 8-bit activations and drafting:

```sh
./llmx serve Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --device rocm:0,rocm:1,rocm:2,rocm:3 --tensor-width 2 --dtype int8 \
  --drafter lookup,MTP_HEAD.gguf --draft-max 63 \
  --lookup-storage file --max-seqs 16 --context-overflow shift \
  --host-cache-bytes 12g \
  --host 0.0.0.0 --port 8090 --api-key-file keys.txt
```

`--lookup-storage file` reads the model's large lookup table from its file and keeps the rows in use in RAM, leaving the GPU memory for the layers and the cache.
`MTP_HEAD.gguf` is a Flash-Next MTP drafter file; with `--drafter lookup` alone it drafts only where the text repeats.

## Something went wrong

- **`Illegal instruction` at start:** the CPU lacks AVX2, FMA or F16C, which llmx needs.
- **`cannot open file: PATH`:** check the path; `llmx pull` prints where it saved a model.
- **`inference: unsupported metadata general.architecture`:** the file is a model family llmx does not run yet (Models, above).
- **`--device vulkan: this build has no Vulkan backend`** (or `rocm`): the binary was built without it; use the release download or rebuild with the backend on.
- **`vulkan: libvulkan.so.1 is not installed` or `vulkan: no device`:** install the GPU's Vulkan driver (on Linux, `libvulkan1` and `mesa-vulkan-drivers`) and add your user to the `render` group.
- **`rocm: no HIP runtime (libamdhip64.so) could be opened` or `rocm: the HIP runtime sees no device`:** install the ROCm 7 runtime and add your user to the `render` group ([ROCM](docs/ROCM.md)).
- **`placement: the model does not fit the devices' free memory`:** add GPUs to `--device`, lower `--ctx-size`, or use `--cpu-moe` on a mixture-of-experts model.
- **`... of type X, which the backend of device N does not support`:** that backend lacks the weight format (Quantization types, below); pick another file of the model.
- **`serve needs --api-key-file PATH, or --no-auth to serve every client`:** give the server a key file, or `--no-auth` on a machine only you reach.
- **`http: bind to 127.0.0.1:8080 failed`:** the port is in use; `--port 8081`.
- Anything else: open an issue with the command, its full output and the first line of `llmx --help`, which names the version.

## Support

### Quantization types

| Weight format | CPU | Vulkan | ROCm | CUDA (in development) |
|---|---|---|---|---|
| `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q4_K`, `Q5_0`, `Q5_1`, `Q5_K`, `Q6_K`, `Q8_0`, `IQ4_NL`, `MXFP4` | Supported | Supported | Supported | Supported |
| `NVFP4`, `Q1_0`, `Q2_0`, `Q2_K`, `Q3_K`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_XS` | Supported | Supported | Supported | Supported |
| `TQ1_0`, `TQ2_0` | Supported | Supported | Supported | Supported |
| `F8_E4M3` (FP8 directories and files) | Supported | Supported | Supported | Supported |

`llmx quantize` writes Q8_0 and Q4_0.
A weight a device cannot run is refused, never moved to the CPU by itself ([storage types](docs/src/quant-types.md)); on Vulkan, MXFP4 needs a device with double-precision support ([VULKAN](docs/VULKAN.md)).
ROCm runs the `TQ` types at exact F32 for every `--dtype`.

### Precision (`--dtype`)

| `--dtype` | CPU | Vulkan | ROCm | CUDA (in development) |
|---|---|---|---|---|
| `auto` (default) | `f16` | `f16` | `f16` | `f16` |
| `f32`, `f16` | Supported | Supported | Supported | Supported |
| `bf16` | Supported | Supported on most devices | Runs `f32` | Runs `f32` |
| `int8` | Runs `f16` | Supported on devices that prefer the integer dot product, such as the MI50 | Supported | Supported |

Weights are never changed; `--dtype` sets the inputs of the matrix products, and every sum is F32.
A device without the asked form runs the next wider one and says so ([PRECISION](docs/PRECISION.md)).

### Backends

| Backend | Hardware | Systems | Status |
|---|---|---|---|
| CPU | x86-64 with AVX2, FMA and F16C | Windows, Linux, Intel macOS | Supported |
| Vulkan | AMD and NVIDIA GPUs with Vulkan 1.2 | Windows, Linux | Supported ([VULKAN](docs/VULKAN.md)) |
| ROCm | AMD GPUs from MI50 (gfx906) on; only gfx906 (MI50, Radeon VII) has run the kernels | Linux | Supported ([ROCM](docs/ROCM.md)) |
| CUDA | NVIDIA GPUs | Linux | In development: `qwen3`, `qwen3moe`, `qwen35`, `qwen35moe` and `deepseek41` with `f32`, `f16` and `int8` on one device, layer splits and tensor groups tried once on a GPU pair ([CUDA](docs/CUDA.md)) |
| SYCL | Intel GPUs | | Planned |

### Several devices

| Mode | Flags | Status |
|---|---|---|
| Layer split: consecutive layers on different devices, GPUs and the CPU mixed | `--device A,B,...`, `--layer-shares` | Supported ([MULTI-DEVICE](docs/MULTI-DEVICE.md)) |
| Expert offload: a mixture-of-experts model's experts on the CPU, not with `--tensor-width` | `--n-cpu-moe N`, `--cpu-moe` | Supported ([USAGE](docs/USAGE.md#expert-offload---n-cpu-moe-n---cpu-moe)) |
| Tensor split: every layer over a group of devices, Vulkan and ROCm on Linux | `--tensor-width N` | Supported ([TENSOR-SPLIT](docs/TENSOR-SPLIT.md)) |
| Several machines | | Planned |

## Documentation

For users:
- [USAGE](docs/USAGE.md): every command and flag.
- [OPERATING](docs/OPERATING.md): running a server, reading its health, fixing problems.
- [SERVER](docs/SERVER.md): how the server schedules users, and its cache.
- [BUILD](docs/BUILD.md): building on each platform; [RELEASE](docs/RELEASE.md): what the downloads and images hold.
- [MULTI-DEVICE](docs/MULTI-DEVICE.md) and [TENSOR-SPLIT](docs/TENSOR-SPLIT.md): running one model on several devices.
- [PRECISION](docs/PRECISION.md): what each part computes in, and `--dtype`.
- [SPECULATIVE](docs/SPECULATIVE.md): drafting and checkpoints.
- [VISION](docs/VISION.md): image input.
- [CHAT-LIVE-METRICS](docs/CHAT-LIVE-METRICS.md): the chat companion's live request metrics.
- [VULKAN](docs/VULKAN.md) and [ROCM](docs/ROCM.md): the GPU backends.
- [QWEN35](docs/QWEN35.md) and [QWEN38-FLASH](docs/QWEN38-FLASH.md): the model families; [Q1](docs/Q1.md): Q1_0 files.
- [ASSETS](docs/ASSETS.md): the models llmx is checked with.
- [ROADMAP](docs/ROADMAP.md): what is planned.

For contributors: [AGENTS](AGENTS.md), [ARCHITECTURE](docs/ARCHITECTURE.md), [ADDING-AN-ARCHITECTURE](docs/ADDING-AN-ARCHITECTURE.md), [EXECUTION](docs/EXECUTION.md), [DEVICE-EXECUTION](docs/DEVICE-EXECUTION.md), [KV-CACHE](docs/KV-CACHE.md), [CACHE-PLAN](docs/CACHE-PLAN.md), [TESTS](docs/TESTS.md), [CI](docs/CI.md) and [STATUS](docs/STATUS.md); the design plans [DEEPSEEK4](docs/DEEPSEEK4.md), [CUDA](docs/CUDA.md) and [DISK-TIER](docs/DISK-TIER.md).

## License

MIT, see [LICENSE](LICENSE).
