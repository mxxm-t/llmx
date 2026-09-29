#!/bin/bash
# fetch.sh: the Qwen3-8B Q6_K (Qwen's own repository, the revision the Q8_0 comes from) and Q4_0 (bartowski's conversion) files, with their SHA-256.
M=/opt/claude-work/llmx-p2-perf2/models
mkdir -p $M
cd $M
curl -sL --retry 5 -o Qwen3-8B-Q6_K.gguf.part https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q6_K.gguf && mv Qwen3-8B-Q6_K.gguf.part Qwen3-8B-Q6_K.gguf
curl -sL --retry 5 -o Qwen3-8B-Q4_0.gguf.part https://huggingface.co/bartowski/Qwen_Qwen3-8B-GGUF/resolve/0b69f75b7472688e6808490aa2b85efdb81b5ce7/Qwen_Qwen3-8B-Q4_0.gguf && mv Qwen3-8B-Q4_0.gguf.part Qwen3-8B-Q4_0.gguf
sha256sum *.gguf > SHA256SUMS
touch fetch.done
