# Changelog

What changed in each release of llmx, newest first, in the style of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Unreleased lists what is on `main` and not yet in a release; the nightly pre-release's notes show it.
Each release's notes on the [releases page](https://github.com/mxxm-t/llmx/releases) are its section here.

## [Unreleased]

## [0.2.1] - 2026-11-23

### Added

- The chat companion shows web and video search results in a conversation's tool activity as linked titles with their sites and, where the search gives them, descriptions or durations; an empty result says so.
- A changelog, `CHANGELOG.md`.
- `docker/compose.yaml` runs the server from a release image beside the chat companion on one machine, with a `vulkan` and a `rocm` profile and one key file for both; the README says how.
- The release archives hold the chat companion and the health dashboard, which need only Python 3, and the image `ghcr.io/mxxm-t/llmx:chat-X.Y.Z` runs the companion, signed with cosign like the other images.
- A nightly pre-release, rebuilt from every push to `main` with a release's downloads and images, for trying changes before they are released; its `--version` names the day and the commit.

### Changed

- The README no longer shows a speed table: several of its MI50 figures, measured before the release, did not reproduce on the released code, and they return once measured again. `llmx bench --model FILE` measures your own machine.
- The quick start and the usage guide say how long a reply runs when no limit is given: 64 tokens in terminal chat (`-n` changes it) and on the server's native API, while on the OpenAI-compatible endpoints a request without `max_tokens` may use all the room its prompt leaves.

### Fixed

- A follow-up turn whose client sends the previous reply's reasoning back, as the chat companion does with Qwen 3.8 models, reused none of the conversation the server had kept and processed the whole prompt again; it now reuses the kept conversation.
- The Docker examples in the README and the release guide, and the server and load-test examples in the usage guide, failed because the server refuses to start without an authentication choice; they now pass `--api-key-file` or `--no-auth`.
- The health dashboard, and the chat companion's Monitor, kept showing request capacity as the current state after the server stopped answering. It now marks those counts as the last sample, and tells a paused view apart from paused scheduler work.
- The health dashboard's model details (placement, lookup storage, RoPE) showed a warning while the server was still starting. They now say whether the server is starting, loading the model, failed to start, or did not report the detail.

## [0.2.0] - 2026-11-22

The first release.

### Added

- Downloads: a Linux x86-64 tarball with the CPU, Vulkan and ROCm backends in one binary (glibc 2.35 or later), a Windows x64 zip with the CPU and Vulkan backends, and `SHA256SUMS`, signed with minisign.
- Docker images `ghcr.io/mxxm-t/llmx:0.2.0-vulkan` and `ghcr.io/mxxm-t/llmx:0.2.0-rocm7`, signed with cosign.
- Backends: the CPU (x86-64 with AVX2, FMA and F16C) on Windows, Linux and Intel macOS; Vulkan on AMD and NVIDIA GPUs on Windows and Linux; ROCm on AMD GPUs from the MI50 on, on Linux. CUDA is in development and builds from source only.
- Models: Qwen3 (0.6B to 32B), Qwen3-30B-A3B, Qwen3.5, Qwen3.6 and Qwen3.8 (dense, and the 35B-A3B mixture of experts) on every backend, and Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash on Vulkan and ROCm.
- Files: GGUF, and dense Qwen3 and Qwen 3.5 safetensors directories, plain or quantized by AWQ, GPTQ, compressed-tensors or FP8. `llmx pull` downloads a model from Hugging Face (curl 8.4 or later).
- Weight formats `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q4_K`, `Q5_0`, `Q5_1`, `Q5_K`, `Q6_K`, `Q8_0`, `IQ4_NL`, `MXFP4`, `NVFP4`, `Q1_0`, `Q2_0`, `Q2_K`, `Q3_K`, the `IQ1` to `IQ4_XS` family, `TQ1_0`, `TQ2_0` and `F8_E4M3` on the CPU, Vulkan and ROCm. `llmx quantize` writes Q8_0 and Q4_0.
- `--dtype` picks the precision of the matrix products: `f16` by default, `f32`, `bf16` or `int8`; every sum is F32, and a device without the asked form runs the next wider one and says so.
- Commands for a completion (`generate`), a conversation in the terminal (`chat`), a server (`serve`), downloading (`pull`), quantizing (`quantize`) and measuring speed (`bench`); `llmx --help` lists them all.
- The server speaks the OpenAI API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`), streamed or whole, with tool calls. It needs a key file (`--api-key-file`) or `--no-auth`.
- The server keeps finished conversations, so a follow-up turn processes only its new text: in GPU memory, then in system RAM (`--host-cache-bytes`), and on disk when given room (`--disk-cache-bytes`, `--disk-cache-dir`, `--disk-cache-keep`). `--context-overflow shift` drops the oldest turns of a conversation that outgrows the context.
- One model over several devices: a layer split over GPUs and the CPU (`--device A,B,...`, `--layer-shares`), a mixture-of-experts model's experts in system RAM (`--cpu-moe`, `--n-cpu-moe N`), and a tensor split over groups of GPUs on Vulkan and ROCm on Linux (`--tensor-width N`).
- Drafting for faster replies with the same text (`--drafter`): from the prompt and reply (`lookup`), from a Qwen 3.5 file's own MTP block (`embedded`), or from a drafter file such as DeepSeek's DSpark.
- Image input (`--mmproj`) for the Qwen 3.5 family and Qwen3.8-Flash-Next on every backend, and for DeepSeek-V4.1-Flash's vision files on Vulkan and ROCm.
- `--load-mode direct` reads the weights around the system's file cache, which loads a large model faster from some file systems.
- In the source tree: the chat companion, a browser interface for a running server with conversations, file workspaces and live request metrics, and the health dashboard.
- llmx is MIT licensed.
