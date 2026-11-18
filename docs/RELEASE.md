# Releases

A release is a version tag on a gated `main` commit.
`.github/workflows/release.yml` builds its downloads and images from that tag and leaves them in a draft release; the owner publishes the draft once the device smoke runs below have passed.
This page says what each artifact holds, how to cut a release, what to check before publishing it, and which repository secrets the signing steps read.

## What a release holds

| Artifact | Contents | Runs on |
|---|---|---|
| `llmx-X.Y.Z-linux-x86_64.tar.gz` | `llmx`, one binary with the CPU, Vulkan and ROCm backends, and `README.md` | x86-64 Linux with AVX2, FMA and F16C and glibc 2.35 or later |
| `llmx-X.Y.Z-windows-x64.zip` | `llmx.exe` with the CPU and Vulkan backends, and `README.md` | x86-64 Windows with AVX2, FMA and F16C |
| `SHA256SUMS` | the SHA-256 of both archives | |
| `SHA256SUMS.minisig` | the minisign signature of `SHA256SUMS`, once the owner's key is set | |
| `llmx.pub`, `cosign.pub` | the public minisign and cosign keys, once the owner has set them (Signing keys, below) | |
| `ghcr.io/mxxm-t/llmx:X.Y.Z-vulkan` | the Linux binary over Debian trixie with the Vulkan loader and Mesa's drivers | a Linux host passing `/dev/dri` |
| `ghcr.io/mxxm-t/llmx:X.Y.Z-rocm7` | the Linux binary over Ubuntu 24.04 with the ROCm 7.2.1 HIP runtime | a Linux host with the amdgpu driver passing `/dev/kfd` and `/dev/dri` |

Both archives hold `LICENSE` and `THIRD_PARTY_NOTICES.md`.

**The Linux binary needs nothing but the C library.**
`docker/release/build-linux.sh` builds it in an Ubuntu 22.04 container with libstdc++ and libgcc linked in, so the oldest C library it accepts is that release's.
The Vulkan loader (`libvulkan.so.1`) and the HIP runtime (`libamdhip64.so`) are opened when a device of their kind is first asked for, as every build does ([BUILD](BUILD.md)), so the binary runs on the CPU on a machine with neither, and `--device vulkan:N` or `--device rocm:N` there fails with a message.
The script refuses a binary that names any other shared library, or that does not name those two for opening at run time, and prints the newest glibc symbol version it uses.
The ROCm kernels are embedded for the default list of AMD architectures, gfx906 (MI50) and later ([BUILD](BUILD.md), GPU architectures), and the Vulkan shaders for every device.

**The Windows binary** is the CMake Vulkan build of [BUILD](BUILD.md) with the Vulkan SDK the workflow pins; it opens the `vulkan-1.dll` the GPU driver installs.

**The images** hold the tarball's binary itself, so every Linux artifact runs the same bytes.
Their entry point is `llmx`, so the arguments of `docker run` are its command line.
The server listens on `127.0.0.1` unless told otherwise, so serving from a container needs `--host 0.0.0.0`:

```sh
docker run --rm --device /dev/dri --group-add "$(getent group render | cut -d: -f3)" \
  -v /path/to/models:/models -p 8080:8080 \
  ghcr.io/mxxm-t/llmx:X.Y.Z-vulkan serve /models/model.gguf --device vulkan:0 --host 0.0.0.0
docker run --rm --device /dev/kfd --device /dev/dri --group-add "$(getent group render | cut -d: -f3)" \
  -v /path/to/models:/models -p 8080:8080 \
  ghcr.io/mxxm-t/llmx:X.Y.Z-rocm7 serve /models/model.gguf --device rocm:0 --host 0.0.0.0
```

While the release is a draft only the versioned tags exist, and they can already be pulled from GHCR before the smoke runs: a draft keeps the release unpublished, not those two tags.
Publishing it adds the floating tags `vulkan` and `rocm7` and copies all four to Docker Hub, by digest, when its secrets are set.
An image for ROCm 10 and a CUDA image come with later releases.

## Cutting a release

1. Pick a `main` commit whose landing gates passed ([STATUS](STATUS.md)) and whose hosted run is green.
2. On a branch off it, set the version in both places the build reads (AGENTS.md, Checkpoints): `project(llmx VERSION X.Y.Z ...)` in `CMakeLists.txt` and `LLMX_RELEASE_VERSION` in `src/config.hpp`.
   This change is docs-and-version only and lands like any other branch.
3. Tag the landed commit `vX.Y.Z` and push the tag to GitHub.
   The workflow's first job refuses a tag on a commit `main` does not hold or that differs from either version, and the build scripts refuse a binary whose `--version` lacks the commit or reports a dirty tree.
   The Linux job's 180-minute limit is twice an estimate: it stays untimed on a hosted runner until the first tag run, which is best pushed to a test repository first.
4. The workflow builds the tarball, the zip and the two images, pushes the images' versioned tags to GHCR, signs what it has keys for, writes `SHA256SUMS` and creates the draft release with every file attached and the images' digests in its notes.
5. Run the smoke checklist below on the draft's files.
6. The owner publishes the draft, which runs the workflow's `publish` job.

The tarball can be built the same way on any Linux machine with Docker, from the repository root at the tagged commit:

```sh
docker run --rm -v "$PWD:/src" -w /src ubuntu:22.04 bash docker/release/build-linux.sh
```

It writes `dist/llmx-X.Y.Z-linux-x86_64.tar.gz`, and leaves its build tree in `build-release`, both of which git ignores.

## Smoke checklist before publishing

Each item runs on the draft's own files, checked against `SHA256SUMS` first.

- **Any Linux machine, no GPU stack:** in a plain `ubuntu:24.04` container, the tarball's `llmx --version` prints `X.Y.Z+g` and the tagged commit, and `llmx generate` on Qwen3-0.6B Q8_0 with `--device cpu` writes text.
  The workflow runs the `--version` half of this.
- **An MI50 on Linux:** the tarball's binary, and each image, run `generate` on Qwen3-0.6B Q8_0 with `--device vulkan:0` (the host's Vulkan loader and RADV for the tarball, the `vulkan` image's for itself) and with `--device rocm:0` (a ROCm 7 runtime on the host for the tarball, the `rocm7` image's for itself); each writes the same text as `--device cpu` for the first tokens.
- **A Radeon VII on Windows:** the zip's `llmx.exe --version`, and `generate` on Qwen3-0.6B Q8_0 with `--device vulkan:0`.
- **Signatures, when the keys are set:** `minisign -Vm SHA256SUMS -p llmx.pub` and `cosign verify --key cosign.pub ghcr.io/mxxm-t/llmx:X.Y.Z-vulkan` (and `-rocm7`) succeed.

Record the results, with each binary's `--version`, in the release's STATUS entry before the owner publishes.

A failed smoke run deletes the draft release, the `vX.Y.Z` tag and the two version-tagged images `X.Y.Z-vulkan` and `X.Y.Z-rocm7` from GHCR, so nothing of that version stays pullable; the fix lands on `main` and the release is cut again from step 1.

## Signing keys

The signing steps read their keys from these repository secrets and skip with a warning while a secret is unset, so a release without them ships unsigned.
The keys are the owner's: the workflow never creates one.
The owner stores the public halves as the repository variables `MINISIGN_PUBLIC_KEY` (the contents of the minisign `.pub` file) and `COSIGN_PUBLIC_KEY` (the contents of `cosign.pub`), and the `draft` job attaches them to the release as `llmx.pub` and `cosign.pub`, or warns that it did not.

| Secret | Read by | Holds |
|---|---|---|
| `MINISIGN_SECRET_KEY` | `draft` | the minisign secret key file's contents, which sign `SHA256SUMS` |
| `MINISIGN_PASSWORD` | `draft` | that key's password |
| `COSIGN_PRIVATE_KEY` | `images`, `publish` | a cosign private key (`cosign generate-key-pair`), which signs each image by digest |
| `COSIGN_PASSWORD` | `images`, `publish` | that key's password |
| `DOCKERHUB_USERNAME` | `publish` | the Docker Hub account the images are mirrored to, as `<account>/llmx` |
| `DOCKERHUB_TOKEN` | `publish` | an access token of that account with write access |

GHCR needs no secret: the workflow pushes with its own token, under the repository's owner.

## License and contributions

llmx is MIT licensed (`LICENSE`), and the notices of the MIT code it contains are in `THIRD_PARTY_NOTICES.md`.
Before the first outside contribution is merged, the project either moves to Apache-2.0 or adopts a contributor agreement that allows relicensing; a sign-off alone does not allow it.
