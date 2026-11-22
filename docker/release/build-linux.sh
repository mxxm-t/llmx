#!/usr/bin/env bash
# Builds the Linux release tarball: one llmx binary with the CPU, Vulkan and ROCm backends (docs/RELEASE.md).
# It runs in a fresh ubuntu:22.04 container with the repository mounted at /src, so the binary needs glibc 2.35 or later and nothing else the host may lack.
# Usage, from the repository root:
#   docker run --rm -v "$PWD:/src" -w /src ubuntu:22.04 bash docker/release/build-linux.sh [nightly]
# The tarball lands in dist/; with nightly the build identifier and the tarball name it a nightly build of the commit (docs/RELEASE.md, The nightly channel).
set -euo pipefail

case "${1:-}" in
    "") nightly=OFF ;;
    nightly) nightly=ON ;;
    *) echo "usage: build-linux.sh [nightly]" >&2; exit 2 ;;
esac

VULKAN_SDK_VERSION=1.4.335.0
ROCM_VERSION=7.2.1

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    binutils ca-certificates cmake curl file g++-12 git gnupg make python3 xz-utils
update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-12 100

# The HIP headers and the compiler that writes the kernels' code objects, from AMD's repository at the release the backend is checked with.
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > /etc/apt/trusted.gpg.d/rocm.gpg
echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} jammy main" > /etc/apt/sources.list.d/rocm.list
printf 'Package: *\nPin: origin repo.radeon.com\nPin-Priority: 600\n' > /etc/apt/preferences.d/rocm-pin-600
apt-get update
apt-get install -y --no-install-recommends hip-dev rocm-device-libs

# The Vulkan headers and glslc from the SDK archive, since the distribution's glslc is older than the shader extensions the kernels use.
mkdir -p /opt/vulkan-sdk
curl -fsSL "https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz" \
    | tar -xJ -C /opt/vulkan-sdk --strip-components=1 "${VULKAN_SDK_VERSION}/x86_64/include/vulkan" "${VULKAN_SDK_VERSION}/x86_64/include/vk_video" "${VULKAN_SDK_VERSION}/x86_64/bin/glslc"
export VULKAN_SDK=/opt/vulkan-sdk/x86_64

# The container runs as root over a tree another user owns, and the build identifier reads its revision through git.
git config --global --add safe.directory /src

version=$(sed -n 's/^project(llmx VERSION \([0-9.]*\) .*/\1/p' CMakeLists.txt)
if [ "$nightly" = ON ]; then
    name=llmx-nightly-linux-x86_64
    identifier="${version}+nightly\.[0-9]\{8\}\.g[0-9a-f]\{12,\}"
else
    name=llmx-${version}-linux-x86_64
    identifier="${version}+g[0-9a-f]\{12,\}"
fi
rm -rf build-release "dist/${name}" "dist/${name}.tar.gz"
mkdir -p dist

# libstdc++ and libgcc are linked in, so the binary needs only the C library; the Vulkan loader and the HIP runtime are opened at run time.
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    -DLLMX_HAS_BACKEND_VULKAN=ON -DLLMX_HAS_BACKEND_ROCM=ON -DLLMX_NIGHTLY="$nightly" \
    "-DCMAKE_EXE_LINKER_FLAGS=-static-libstdc++ -static-libgcc"
cmake --build build-release --parallel "$(nproc)"

bin=build-release/llmx
strip "$bin"

# The binary may need only the C library's own objects: a GPU runtime or a C++ library among them would stop it on a machine without one.
needed=$(readelf -d "$bin" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p')
echo "NEEDED: $(echo $needed)"
for lib in $needed; do
    case "$lib" in
        libc.so.6|libm.so.6|libdl.so.2|libpthread.so.0|librt.so.1|ld-linux-x86-64.so.2) ;;
        *) echo "error: $bin needs $lib" >&2; exit 1 ;;
    esac
done
for lib in libvulkan.so.1 libamdhip64.so; do
    grep -qa "$lib" "$bin" || { echo "error: $bin does not name $lib for dlopen" >&2; exit 1; }
done
echo "GLIBC: $(objdump -T "$bin" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)"

"$bin" --version
"$bin" --version | grep -q "^llmx ${identifier} " || { echo "error: the build identifier lacks the release or the commit, or the tree is dirty" >&2; exit 1; }

mkdir -p "dist/${name}"
cp "$bin" README.md "dist/${name}/"
cp LICENSE THIRD_PARTY_NOTICES.md "dist/${name}/"
bash docker/release/companion.sh "dist/${name}"
tar -C dist -czf "dist/${name}.tar.gz" "$name"
rm -rf "dist/${name}"
echo "dist/${name}.tar.gz"
