# The release images: the binary of the Linux tarball over the runtime one device kind needs, and the tarball's chat companion and health dashboard (docs/RELEASE.md).
# The build context is the extracted tarball, `llmx` and `tools/` at its top; each image is one target:
#   docker build --target vulkan -t llmx:vulkan -f docker/release/Dockerfile <dir>
#   docker build --target rocm7 -t llmx:rocm7 -f docker/release/Dockerfile <dir>
#   docker build --target rocm10 -t llmx:rocm10 -f docker/release/Dockerfile <dir>
#   docker build --target chat -t llmx:chat -f docker/release/Dockerfile <dir>
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

# The HIP runtime of ROCm 10 from AMD's ROCm Core SDK repository, which loads the binary's own gfx906 code objects although ROCm 10 ships no gfx906 library kernels (docs/ROCM.md).
# Its runtime package depends on the whole compiler and the base package holds librocprofiler-register, so this stage unpacks both and keeps only libamdhip64, the libraries it loads and libdrm's data, amdgpu.ids.
FROM ubuntu:24.04 AS rocm10-runtime
ARG ROCM10_VERSION=10.2.0
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && curl -fsSL https://stable.repo.amd.com/rocm/gpg/packages.gpg -o /etc/apt/trusted.gpg.d/amdrocm.asc \
    && echo "deb [arch=amd64] https://stable.repo.amd.com/rocm/core/packages/ubuntu2404 stable main" > /etc/apt/sources.list.d/amdrocm.list \
    && apt-get update \
    && MM=$(echo "$ROCM10_VERSION" | cut -d. -f1,2) \
    && cd /tmp && apt-get download "amdrocm-base$MM=$ROCM10_VERSION-*" "amdrocm-runtime$MM=$ROCM10_VERSION-*" "amdrocm-sysdeps$MM=$ROCM10_VERSION-*" \
    && for d in /tmp/*.deb; do dpkg-deb -x "$d" /tmp/x; done \
    && src=$(dirname "$(find /tmp/x -name 'libamdhip64.so.7' | head -1)") \
    && mkdir -p /opt/rocm/lib/rocm_sysdeps/lib /opt/rocm/share/doc \
    && for lib in "$src/libamdhip64.so.7" $(ldd "$src/libamdhip64.so.7" | awk '$3 ~ /^\/tmp\/x\// {print $3}'); do \
         cp -L "$lib" "/opt/rocm/lib/${lib#$src/}"; done \
    && mkdir -p /opt/rocm/lib/rocm_sysdeps/share \
    && cp -r "$src/rocm_sysdeps/share/libdrm" /opt/rocm/lib/rocm_sysdeps/share/ \
    && cp -r "$src/../share/doc/." /opt/rocm/share/doc/

FROM ubuntu:24.04 AS rocm10
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=rocm10-runtime /opt/rocm /opt/rocm
ENV LD_LIBRARY_PATH=/opt/rocm/lib
COPY llmx /usr/local/bin/llmx
ENTRYPOINT ["llmx"]
CMD ["--help"]

# The chat companion, which needs only Python's standard library, run as an unprivileged user with its conversations in /data.
# It listens on 127.0.0.1:8183 unless given --host, so it suits the host's network; the health dashboard runs from the same image with another command.
FROM python:3.12-slim AS chat
COPY tools/chat_ui /app/chat_ui
COPY tools/health_dashboard /app/health_dashboard
RUN useradd --system --uid 10001 --user-group --home-dir /data --no-create-home llmx     && mkdir -p /data /app/chat_ui/.pdf-reader     && chown llmx:llmx /data /app/chat_ui/.pdf-reader
USER llmx
VOLUME /data
EXPOSE 8183
STOPSIGNAL SIGINT
CMD ["python", "/app/chat_ui/server.py", "--data-dir", "/data", "--upstream", "http://127.0.0.1:8080"]
