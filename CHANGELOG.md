# Changelog

What changed in each release of llmx, newest first, in the style of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Unreleased lists what is on `main` and not yet in a release; the nightly pre-release's notes show it.
Each release's notes on the [releases page](https://github.com/mxxm-t/llmx/releases) are its section here.

Each entry is one short line that links the doc explaining it; reasons, flags and figures belong there ([RELEASE](docs/RELEASE.md#the-changelog)).

## [Unreleased]

### Added

- A `rocm10` release image runs llmx on the ROCm 10.2 runtime ([ROCm](docs/ROCM.md)).
- A DSpine drafter (`--drafter FILE`) drafts a block of tokens beside a Qwen 3.5-family model ([speculative decoding](docs/SPECULATIVE.md)).
- The server reads tool calls in the Inkling-2-Mini chat format and streams reasoning apart from the answer ([tool calls](docs/SERVER.md#tool-calls)).

### Changed

- Several sequences of BF16, F16 and F32 weights decode faster on CUDA, and 2 to 4 on MI50s ([GPU backends](docs/src/backends-gpu.md)).
- The Docker Hub page is set from the tree at each release, and its quick start serves with a key file ([releases](docs/RELEASE.md)).
- Qwen3.6-27B decodes faster with ROCm on four MI50s as one tensor group at `--dtype int8` ([tensor split](docs/TENSOR-SPLIT.md)).
- Qwen3.5 and Qwen3.6 decode each full attention layer in one ROCm launch ([kernel notes](docs/src/backends-gpu.md)).

### Fixed

- Chat templates that use Jinja's `float` filter, such as Inkling-2-Mini's, render ([chat templates](docs/src/inference-chat.md)).
- DeepSeek-V4.1-Flash with its DSpark drafter fits the ten-card layer split ([speculative decoding](docs/SPECULATIVE.md)).

## [0.2.1] - 2026-11-23

### Added

- The chat companion shows web and video search results in its tool activity ([chat companion](tools/chat_ui/README.md#tools)).
- A changelog, `CHANGELOG.md`, whose sections are the release notes ([RELEASE](docs/RELEASE.md#the-changelog)).
- `docker/compose.yaml` runs the server and the chat companion together ([README](README.md#server-and-chat-companion-in-containers)).
- Release archives hold the chat companion and health dashboard, and a `chat-X.Y.Z` image runs them ([RELEASE](docs/RELEASE.md#what-a-release-holds)).
- A nightly pre-release is rebuilt from every push to `main` ([RELEASE](docs/RELEASE.md#the-nightly-channel)).

### Changed

- The README drops its speed table until it is measured again; `llmx bench --model FILE` times your machine ([README](README.md#how-fast-is-it)).
- The default reply length is stated: 64 tokens in terminal chat and the native API, uncapped on OpenAI routes ([README](README.md#quick-start)).

### Fixed

- A follow-up turn that sends back the reply's reasoning now reuses the kept conversation ([README](README.md#conversation-cache)).
- The Docker and server examples now pass `--api-key-file` or `--no-auth`, which the server requires ([USAGE](docs/USAGE.md#llmx-serve)).
- The health dashboard and chat Monitor mark capacity as the last sample once the server stops answering ([dashboard](tools/health_dashboard/README.md)).
- The dashboard's model details say whether the server is starting, loading, failed or silent ([dashboard](tools/health_dashboard/README.md)).

## [0.2.0] - 2026-11-22

The first release.

### Added

- Downloads: a Linux tarball, a Windows zip and signed `SHA256SUMS` ([RELEASE](docs/RELEASE.md#what-a-release-holds)).
- Docker images `0.2.0-vulkan` and `0.2.0-rocm7`, signed with cosign ([RELEASE](docs/RELEASE.md#the-images)).
- CPU, Vulkan and ROCm backends; CUDA builds from source only ([README](README.md#backends)).
- Qwen3, Qwen3.5, Qwen3.6, Qwen3.8, Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash models ([README](README.md#models)).
- GGUF and safetensors files, and `llmx pull` to download from Hugging Face ([USAGE](docs/USAGE.md#llmx-pull)).
- Weight formats from `F32` to the `IQ` and `TQ` families ([README](README.md#quantization-types)).
- `--dtype` picks the matrix-product precision: `f16`, `f32`, `bf16` or `int8` ([README](README.md#precision---dtype)).
- Commands `generate`, `chat`, `serve`, `pull`, `quantize` and `bench` ([USAGE](docs/USAGE.md)).
- A server for the OpenAI API with tool calls, started with `--api-key-file` or `--no-auth` ([USAGE](docs/USAGE.md#llmx-serve)).
- A conversation cache in GPU memory, system RAM and on disk, so follow-up turns skip old text ([README](README.md#conversation-cache)).
- One model over several devices, with expert offload and tensor split ([MULTI-DEVICE](docs/MULTI-DEVICE.md)).
- Drafting with `--drafter` for faster replies ([SPECULATIVE](docs/SPECULATIVE.md)).
- Image input with `--mmproj` ([VISION](docs/VISION.md)).
- `--load-mode direct` reads weights around the file cache ([USAGE](docs/USAGE.md#reading-the-weights---load-mode)).
- The chat companion and the health dashboard are in the source tree ([chat companion](tools/chat_ui/README.md)).
- llmx is MIT licensed ([README](README.md#license)).
