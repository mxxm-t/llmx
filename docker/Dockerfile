# Builds and runs llmx with the Vulkan backend on a Linux machine whose
# host has no Vulkan driver or SDK installed.
# Everything the backend needs lives in the image: the loader it opens at run time, the Mesa driver that enumerates the cards, the headers and the shader compiler the build needs.
# The host supplies only the cards themselves, through /dev/dri, so a
# shared machine keeps its own package set untouched.
#
# Build the image from the repository root:
#   docker build -t llmx-dev docker
# Then, with the repository mounted and the render nodes passed in:
#   docker run --rm -it --device /dev/dri \
#     --group-add "$(getent group render | cut -d: -f3)" \
#     -v "$PWD:/llmx" -w /llmx llmx-dev
#
# Inside, the ordinary build and test commands apply (docs/BUILD.md):
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLLMX_HAS_BACKEND_VULKAN=ON
#   cmake --build build --parallel
#   ctest --test-dir build --output-on-failure
#   ./build/llmx bench --model <model.gguf> --device vulkan:0
FROM debian:trixie-slim

# Debian trixie carries a shader compiler and Vulkan headers new enough for these shaders, so no vendor repository is added here.
# curl is here because `llmx pull` runs it as a child process to fetch models from the Hub.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        glslc \
        libvulkan-dev \
        mesa-vulkan-drivers \
        python3 \
        vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /llmx
CMD ["/bin/bash"]
