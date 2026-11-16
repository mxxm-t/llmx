# The release images: the binary of the Linux tarball over the runtime one device kind needs (docs/RELEASE.md).
# The build context is a directory holding that binary as `llmx`; each image is one target:
#   docker build --target vulkan -t llmx:vulkan -f docker/release/Dockerfile <dir>
#   docker build --target rocm7 -t llmx:rocm7 -f docker/release/Dockerfile <dir>
# The host supplies the cards, through /dev/dri for Vulkan and /dev/dri and /dev/kfd for ROCm.

# The Vulkan loader and Mesa's drivers, RADV for AMD cards among them; curl is what `llmx pull` downloads with.
FROM debian:trixie-slim AS vulkan
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libvulkan1 mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*
COPY llmx /usr/local/bin/llmx
ENTRYPOINT ["llmx"]
CMD ["--help"]

# The HIP runtime of ROCm 7, which llmx opens when a ROCm device is asked for; the kernels themselves are in the binary.
FROM ubuntu:24.04 AS rocm7
ARG ROCM_VERSION=7.2.1
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > /etc/apt/trusted.gpg.d/rocm.gpg \
    && echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" > /etc/apt/sources.list.d/rocm.list \
    && printf 'Package: *\nPin: origin repo.radeon.com\nPin-Priority: 600\n' > /etc/apt/preferences.d/rocm-pin-600 \
    && apt-get update \
    && apt-get install -y --no-install-recommends hip-runtime-amd \
    && apt-get purge -y gnupg && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH=/opt/rocm/lib
COPY llmx /usr/local/bin/llmx
ENTRYPOINT ["llmx"]
CMD ["--help"]
