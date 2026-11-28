# Changelog

What changed in each release of llmx, newest first, in the style of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Unreleased lists what is on `main` and not yet in a release; the nightly pre-release's notes show it.
Each release's notes on the [releases page](https://github.com/mxxm-t/llmx/releases) are its section here.

Each entry is one short line that links the doc explaining it; reasons, flags and figures belong there ([RELEASE](docs/RELEASE.md#the-changelog)).

## [Unreleased]

### Added

- An experimental Windows ROCm zip runs llmx on the AMD driver's HIP runtime ([build](docs/BUILD.md#the-rocm-backend-on-windows)).
- Inkling-2-Mini runs on the CPU, Vulkan and ROCm ([Inkling-2](docs/INKLING2.md)).
- Inkling-2-Mini reads images with `--mmproj` on the CPU, Vulkan and ROCm ([vision](docs/VISION.md#12-inkling-2-inkling2v)).
- A `rocm10` release image runs llmx on the ROCm 10.2 runtime ([ROCm](docs/ROCM.md)).
- A DFly2 drafter (`--drafter FILE`) drafts beside DeepSeek-V4.1-Flash, not yet faster than plain decoding ([speculative decoding](docs/SPECULATIVE.md)).
- A DSpine drafter (`--drafter FILE`) drafts a block of tokens beside a Qwen 3.5-family model ([speculative decoding](docs/SPECULATIVE.md)).
- A DFlash2 drafter (`--drafter FILE`) drafts beside a Qwen 3.5-family model on the CPU, ROCm and Vulkan ([speculative decoding](docs/SPECULATIVE.md)).
- A DFlash drafter (`--drafter FILE`), such as z-lab's for Qwen3.6-35B-A3B, drafts beside a Qwen 3.5-family model ([speculative decoding](docs/SPECULATIVE.md)).
- The server reads tool calls in the Inkling-2-Mini chat format and streams reasoning apart from the answer ([tool calls](docs/SERVER.md#tool-calls)).
- MiniMax-M3.1 runs whole on ROCm and Vulkan devices, its sigmoid router included ([MiniMax](docs/MINIMAX.md)).
- `--expert-cache SIZE` keeps hot CPU-held MoE experts on the device ([usage](docs/USAGE.md#expert-offload---n-cpu-moe-n---cpu-moe)).

### Changed

- Decode sums at tensor width 2 on ROCm take beside their push, about 0.7% faster on two MI50s ([DEVICE-TUNING](docs/DEVICE-TUNING.md)).
- Faster int8 decode at four MI50s: the held Q8_0 build takes the tensor group's sum itself ([DEVICE-TUNING](docs/DEVICE-TUNING.md)).
- ROCm tensor groups on MI50 sum prompt-sized parts up to a quarter faster by reading their peers' rows ([GPU backends](docs/src/backends-gpu.md)).
- A tensor width that does not divide a model's vocabulary splits the head's rows unevenly instead of refusing ([TENSOR-SPLIT](docs/TENSOR-SPLIT.md)).
- Greedy server requests, `ignore_eos` ones too, take their tokens from the device's argmax ([server](docs/SERVER.md#device-argmax)).
- Greedy `generate` and a server on a tensor group queue the next decode step before reading the last one ([generate](docs/src/inference-generate.md)).
- Greedy `ignore_eos` requests whose top token is the end token no longer read their logits back from the device ([server](docs/SERVER.md#device-argmax)).
- Experts on the CPU decode in the three-dispatch MoE step whatever their weight types ([CPU backend](docs/src/backends-cpu.md)).
- A server round mixing a prompt with decoding spends no attention launches on the prompt's rows on CUDA and ROCm ([GPU backends](docs/src/backends-gpu.md)).
- A server on one device queues the next greedy decode step on the device before the host reads the last one ([server](docs/SERVER.md#fed-decode)).
- `generate` with a drafter on one device feeds each verify its drafts on the device ([speculative decoding](docs/SPECULATIVE.md)).
- Vulkan fuses the MoE decode step per product, so Q8_0, Q5_K and BF16-shared Qwen3.5/3.6 MoE files take it ([Vulkan backend](docs/src/backends-vulkan.md)).
- Several sequences of BF16, F16 and F32 weights decode faster on CUDA, and 2 to 4 on MI50s ([GPU backends](docs/src/backends-gpu.md)).
- Batched CUDA decode of BF16, F16 and F32 weights takes a faster column geometry, q/k/v and gate/up one launch each ([GPU backends](docs/src/backends-gpu.md)).
- The Docker Hub page is set from the tree at each release, and its quick start serves with a key file ([releases](docs/RELEASE.md)).
- Qwen 3.5-family linear-attention layers run one fused GPU step at every type, f16 included ([GPU backends](docs/src/backends-gpu.md)).
- BF16 files run their prompts on TF32 tensor cores with CUDA, in the F16 precision class ([CUDA](docs/CUDA.md#bf16-prompt-tile)).
- `--dtype bf16` runs BF16 files' prompts on BF16 tensor cores with CUDA ([CUDA](docs/CUDA.md#bf16-prompt-tile)).
- Qwen3.6-27B decodes faster with ROCm on four MI50s as one tensor group at `--dtype int8` ([tensor split](docs/TENSOR-SPLIT.md)).
- Qwen3.5 and Qwen3.6 decode each full attention layer in one ROCm launch ([kernel notes](docs/src/backends-gpu.md)).
- ROCm and CUDA tensor groups sum one part of a prompt beside the other part's work, batched prompts included ([tensor split](docs/TENSOR-SPLIT.md)).
- DeepSeek-V4.1 decodes short contexts faster by skipping its indexer ([DeepSeek-V4.1](docs/src/model-arch-deepseek4.md)).
- Q8_0 int8 decode of 9 or 10 sequences at once is faster on ROCm ([ROCm](docs/ROCM.md)).
- Qwen3.6-35B-A3B and other MoE files with K-quant downs decode faster on an MI50 with ROCm ([kernel notes](docs/src/backends-gpu.md)).
- `--verbose` placement lines call their weights figure an estimate of device memory, given per member for a tensor group ([usage](docs/USAGE.md)).
- A DSpine drafter also drafts beside a model split as a tensor group (`--tensor-width`) ([speculative decoding](docs/SPECULATIVE.md)).
- A DSpine drafter asked for fewer drafts runs a shorter block, so a small `--draft-max` decodes faster ([speculative decoding](docs/SPECULATIVE.md)).
- A DSpine drafter's candidate picks over the vocabulary run several times faster on an MI50 ([ROCm tuning](docs/DEVICE-TUNING.md)).

### Fixed

- `bench --profile` on ROCm and CUDA no longer books the host's launch time into decode kernels, and says how many ops it timed ([usage](docs/USAGE.md)).
- Small split Vulkan float-weight products no longer give infinity for a finite sum ([Vulkan](docs/VULKAN.md#kernel-notes)).
- Small split Vulkan products that add to their output sum their parts before adding it, as unsplit ones do ([Vulkan](docs/VULKAN.md#kernel-notes)).
- GPU tensor groups no longer stall for 30 s and give up a sum when device memory is freed during lane sums ([GPU backends](docs/src/backends-gpu.md)).
- A GPU tensor group keeps each device at four hardware queues however many sum collectives are open ([GPU backends](docs/src/backends-gpu.md)).
- `serve` with an automatic KV budget no longer runs a nearly empty GPU out of memory after loading ([placement](docs/src/model-place.md)).
- `--yarn` now reaches a DFlash or DSpine drafter's rope, as it reaches the target ([usage](docs/USAGE.md)).
- `--lookup-storage` without lookup tables and `--cpu-moe` or `--n-cpu-moe` with the CPU alone are refused, not ignored ([usage](docs/USAGE.md)).
- `logits --verbose` and the `experts:` placement line show `--per-token`, `--cpu-moe` and `--moe-stream-from` at work ([usage](docs/USAGE.md#llmx-logits)).
- `serve` refuses disk cache flags without `--disk-cache-bytes` and `--state-checkpoints` without recurrent layers ([usage](docs/USAGE.md#llmx-serve)).
- `serve` reports its cache types, threads and load mode at start and in `/v1/health` ([usage](docs/USAGE.md#llmx-serve)).
- MiniMax-M3.1 GGUF files load as published: llmx reads their `minimax-m2` tokenizer ([tokenizer](docs/src/tokenizer-tokenizer.md)).
- The automatic KV budget on Vulkan no longer asks for one buffer past the driver's largest allocation ([usage](docs/USAGE.md)).
- DeepSeek-V4.1 on a tensor group of two or more devices gives the right logits on prompts of 128 tokens or more ([tensor split](docs/TENSOR-SPLIT.md)).
- `--mmproj` no longer runs a card out of memory: the fit leaves room for the vision encoder before the KV ([vision](docs/VISION.md)).
- Qwen3.8-Flash-Next's MTP drafts after an image use the image's positions, as the model's layers do ([vision](docs/VISION.md)).
- A tensor group no longer hangs when one of its cards fails; its sums give up after 30 seconds ([tensor split](docs/TENSOR-SPLIT.md)).
- A load that keeps `--moe-stream-from` or `--expert-cache` layers on the host says why in one line ([usage](docs/USAGE.md)).
- Vulkan prompt products over float weights sum in 256-input blocks, so long F32 rows stay near float32 rounding ([Vulkan](docs/VULKAN.md#kernel-notes)).
- Vulkan prompts over F16 weights run up to 16 times faster and no longer lose the device on long calls ([Vulkan](docs/VULKAN.md#kernel-notes)).
- MiniMax-M3 files whose indexer block size is not a power of two are refused at load, not mid-pass ([MiniMax](docs/MINIMAX.md)).
- Image encoders on the CPU and Vulkan sum attention by blocks, so a 1080p image meets its HF bound ([precision](docs/PRECISION.md#attention-value-sums)).
- `--verbose`'s kv line counts every card of a tensor group and names recurrent state bytes ([usage](docs/USAGE.md)).
- `--draft-max` without `--drafter` is refused instead of ignored ([speculative decoding](docs/SPECULATIVE.md)).
- `--threads` and `--threads-batch` on a placement with nothing on the CPU now say they change nothing ([usage](docs/USAGE.md)).
- Chat templates that use Jinja's `float` filter, such as Inkling-2-Mini's, render ([chat templates](docs/src/inference-chat.md)).
- MiniMax-M3.1's attention runs on Vulkan with `--cpu-moe` instead of failing ([MiniMax](docs/MINIMAX.md)).
- DeepSeek-V4.1-Flash with its DSpark drafter fits the ten-card layer split ([speculative decoding](docs/SPECULATIVE.md)).
- A pipelined prompt is no longer cut into one-row slices after its stages grow faster ([slicing](docs/src/model-slicing.md)).

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
