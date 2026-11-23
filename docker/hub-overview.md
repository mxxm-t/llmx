# llmx

llmx runs large language models on your own machine: on the CPU, on AMD and NVIDIA GPUs through Vulkan, and on AMD GPUs through ROCm.
It reads GGUF files and dense Qwen3 and Qwen 3.5 safetensors directories, downloads models from Hugging Face, chats in the terminal and serves an OpenAI-compatible API, and it splits one model over several GPUs when it does not fit on one.

Source, documentation and release notes: https://github.com/mxxm-t/llmx

## Tags

| Tag | What it holds |
|---|---|
| `vulkan`, `X.Y.Z-vulkan` | llmx with the Vulkan loader and Mesa drivers (RADV for AMD cards); also runs on the CPU |
| `rocm7`, `X.Y.Z-rocm7` | llmx with the ROCm 7 HIP runtime for AMD GPUs |
| `rocm10`, `X.Y.Z-rocm10` | llmx with the ROCm 10 HIP runtime, for releases that include this image |
| `chat`, `chat-X.Y.Z` | the Chat companion and health dashboard, for use with a running llmx server |

Release notes list the exact images and digests for each version. The unversioned tags follow the latest published release.
The same images are on GitHub: `ghcr.io/mxxm-t/llmx`.
Images are signed with cosign; the public key `cosign.pub` is attached to each GitHub release:

```sh
cosign verify --key cosign.pub mxxm/llmx:X.Y.Z-rocm7
```

## Quick start

The Vulkan and ROCm images run `llmx`. Mount a host directory at `/models` so the downloaded weights are available to each command.

Download a model:

```sh
model=$(docker run --rm -v "$HOME/models:/models" mxxm/llmx:vulkan \
  pull Qwen/Qwen3-0.6B-GGUF:Q8_0 --cache-dir /models)
```

The shell variable `model` holds the path inside the container. You can also set it to the path of a GGUF file you already have, such as `model=/models/my-model.gguf`.

Chat on an AMD or NVIDIA GPU through Vulkan:

```sh
docker run --rm -it --device /dev/dri -v "$HOME/models:/models" mxxm/llmx:vulkan \
  chat "$model" -n 512 --device vulkan:0
```

Chat on an AMD GPU through ROCm:

```sh
docker run --rm -it --device /dev/kfd --device /dev/dri --group-add video \
  -v "$HOME/models:/models" mxxm/llmx:rocm7 \
  chat "$model" -n 512 --device rocm:0
```

Serve an OpenAI-compatible API on local port 8080. Create `$HOME/llmx-keys.txt` with one key per line. Each key needs at least 16 printable ASCII characters without spaces.

```sh
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  -v "$HOME/models:/models" -v "$HOME/llmx-keys.txt:/run/llmx-keys:ro" \
  -p 127.0.0.1:8080:8080 mxxm/llmx:rocm7 \
  serve "$model" --device rocm:0 --host 0.0.0.0 --port 8080 --api-key-file /run/llmx-keys
```

Send your key in the `Authorization: Bearer KEY` header.
Without `--device` llmx runs on the CPU.
Use `llmx --help` for the command list and `llmx serve --help` for server options.

## License

MIT. See the repository for details.
