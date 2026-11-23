# Releases

A release is a version tag on a gated `main` commit.
`.github/workflows/release.yml` builds its downloads and images from that tag and leaves them in a draft release, which is published once the device smoke runs below have passed.
This page says what each artifact holds, how the changelog is kept, how to cut a release, what to check before publishing it, how the nightly channel follows `main`, and which repository secrets the signing steps read.

## What a release holds

| Artifact | Contents | Runs on |
|---|---|---|
| `llmx-X.Y.Z-linux-x86_64.tar.gz` | `llmx`, one binary with the CPU, Vulkan and ROCm backends, `README.md` and `tools/` | x86-64 Linux with AVX2, FMA and F16C and glibc 2.35 or later |
| `llmx-X.Y.Z-windows-x64.zip` | `llmx.exe` with the CPU and Vulkan backends, `README.md` and `tools/` | x86-64 Windows with AVX2, FMA and F16C |
| `SHA256SUMS` | the SHA-256 of both archives | |
| `SHA256SUMS.minisig` | the minisign signature of `SHA256SUMS`, once the owner's key is set | |
| `llmx.pub`, `cosign.pub` | the public minisign and cosign keys, once the owner has set them (Signing keys, below) | |
| `ghcr.io/mxxm-t/llmx:X.Y.Z-vulkan` | the Linux binary over Debian trixie with the Vulkan loader and Mesa's drivers | a Linux host passing `/dev/dri` |
| `ghcr.io/mxxm-t/llmx:X.Y.Z-rocm7` | the Linux binary over Ubuntu 24.04 with the ROCm 7.2.1 HIP runtime | a Linux host with the amdgpu driver passing `/dev/kfd` and `/dev/dri` |
| `ghcr.io/mxxm-t/llmx:X.Y.Z-rocm10` | the Linux binary over Ubuntu 24.04 with the ROCm 10.2.0 HIP runtime, `libamdhip64` and the libraries it loads alone | the same host as `rocm7` |
| `ghcr.io/mxxm-t/llmx:chat-X.Y.Z` | the archives' `tools/` over Python 3.12: the chat companion, and the health dashboard as another command | any Linux host, beside a server it reaches |

Both archives hold `LICENSE` and `THIRD_PARTY_NOTICES.md`.

**`tools/` is the chat companion and the health dashboard,** `tools/chat_ui` and `tools/health_dashboard` as the commit holds them without their tests, which `docker/release/companion.sh` copies for both archives, so the image gets the same files from the tarball.
Both need only Python 3's standard library and run from the extracted directory as from a checkout, against a running `llmx serve`:

```sh
python3 tools/chat_ui/server.py --upstream http://127.0.0.1:8080 --api-key-file llmx.key                  # http://127.0.0.1:8183
python3 tools/health_dashboard/server.py --upstream http://127.0.0.1:8080/v1/health --api-key-file llmx.key # http://127.0.0.1:8182
```

The key file holds one of the server's keys ([SERVER](SERVER.md#health-dashboard)); a server started with `--no-auth` needs none.
Both listen on 127.0.0.1 unless given `--host`, and the companion serves another address only with `--password-file` ([the companion's README](../tools/chat_ui/README.md)).

**The Linux binary needs nothing but the C library.**
`docker/release/build-linux.sh` builds it in an Ubuntu 22.04 container with libstdc++ and libgcc linked in, so the oldest C library it accepts is that release's.
The Vulkan loader (`libvulkan.so.1`) and the HIP runtime (`libamdhip64.so`) are opened when a device of their kind is first asked for, as every build does ([BUILD](BUILD.md)), so the binary runs on the CPU on a machine with neither, and `--device vulkan:N` or `--device rocm:N` there fails with a message.
The script refuses a binary that names any other shared library, or that does not name those two for opening at run time, and prints the newest glibc symbol version it uses.
The ROCm kernels are embedded for the default list of AMD architectures, gfx906 (MI50) and later ([BUILD](BUILD.md), GPU architectures), and the Vulkan shaders for every device.

**The Windows binary** is the CMake Vulkan build of [BUILD](BUILD.md) with the Vulkan SDK the workflow pins; it opens the `vulkan-1.dll` the GPU driver installs.

## The images

**The images** hold the tarball's binary itself, so every Linux artifact runs the same bytes.
The `vulkan` and `rocm7` images' entry point is `llmx`, so the arguments of `docker run` are its command line.
To serve clients on other machines, bind to `0.0.0.0` inside the container and publish its port. Create a private key file on the host with one randomly generated bearer key per line, then mount it read-only. Clients send the key in the `Authorization` header as described in [Server authentication](SERVER.md#authentication).

```sh
docker run --rm --device /dev/dri --group-add "$(getent group render | cut -d: -f3)" \
  -v /path/to/models:/models -v /path/to/llmx.keys:/run/llmx.keys:ro -p 8080:8080 \
  ghcr.io/mxxm-t/llmx:X.Y.Z-vulkan serve /models/model.gguf --device vulkan:0 --host 0.0.0.0 --api-key-file /run/llmx.keys
docker run --rm --device /dev/kfd --device /dev/dri --group-add "$(getent group render | cut -d: -f3)" \
  -v /path/to/models:/models -v /path/to/llmx.keys:/run/llmx.keys:ro -p 8080:8080 \
  ghcr.io/mxxm-t/llmx:X.Y.Z-rocm7 serve /models/model.gguf --device rocm:0 --host 0.0.0.0 --api-key-file /run/llmx.keys
```

The `chat` image runs the companion as an unprivileged user with its conversations in the volume `/data`, on port 8183 of 127.0.0.1, so it shares the host's network with the server; its default command reaches a server on `127.0.0.1:8080` without a key, and a command of its own replaces that, the health dashboard's among them:

```sh
docker run --rm --network host -v llmx-chat:/data -v /path/to/llmx.key:/run/llmx.key:ro ghcr.io/mxxm-t/llmx:chat-X.Y.Z \
  python /app/chat_ui/server.py --data-dir /data --upstream http://127.0.0.1:8080 --api-key-file /run/llmx.key
docker run --rm --network host -v /path/to/llmx.key:/run/llmx.key:ro ghcr.io/mxxm-t/llmx:chat-X.Y.Z \
  python /app/health_dashboard/server.py --upstream http://127.0.0.1:8080/v1/health --api-key-file /run/llmx.key
```

`docker/release/images.sh` owns the images: their four targets in `docker/release/Dockerfile`, their tag names, pushing, the Docker Hub copies and the cosign signatures.
`images.sh build VERSION CONTEXT [push]` builds every target from the extracted tarball and with `push` sends each to GHCR by its digest alone, signs it and prints `TARGET=sha256:DIGEST` lines, so an image gains a tag only when its run publishes.
`images.sh tag NAME [hub] [floating] TARGET=sha256:DIGEST...` points NAME's tags at those digests, or with `tags` in place of the digests at NAME's existing GHCR tags; it tags Docker Hub too with `hub` and the floating tags too with `floating`, signs the Docker Hub copies, and refuses to run without one source for each target.

While the release is a draft only the versioned tags exist, and they can already be pulled from GHCR before the smoke runs: a draft keeps the release unpublished, not those tags.
Publishing it adds the floating tags `vulkan`, `rocm7`, `rocm10` and `chat` and copies all eight to Docker Hub, by digest, when its secrets are set.
It also sets the Docker Hub overview from `docker/hub-overview.md`, whose `X.Y.Z` becomes the version, through `docker/hub-overview.sh`; the nightly channel leaves it alone.
The `Docker Hub overview` workflow (`.github/workflows/hub-overview.yml`, run by hand with a version) sets it again for a published release without cutting one.
The `rocm10` image holds the same binary as `rocm7`: ROCm 10's HIP runtime keeps the name `libamdhip64.so.7` and loads the binary's gfx906 code objects on an MI50, although ROCm 10 ships no gfx906 kernels of its own ([ROCM](ROCM.md), ROCm 10).
A CUDA image comes with a later release.

## The changelog

[`CHANGELOG.md`](../CHANGELOG.md) lists what each release changed for its users, newest first, in the style of Keep a Changelog: a `## [Unreleased]` section on top, then one `## [X.Y.Z] - YYYY-MM-DD` section per release, each with `### Added`, `### Changed`, `### Deprecated`, `### Removed`, `### Fixed` or `### Security` subsections.
It is written for someone running llmx: what they can do now, or what behaves differently, in plain words, without branch names, internal tests or measurements of the gates.
Each entry is one line of about 120 characters and one sentence, ends with a link to the doc that explains the change, and leaves reasons, examples, flag lists and history to that doc.

- **Each landing adds its lines.** A change a user notices adds a line under Unreleased in the commit that lands it; a change nobody outside the project notices adds none. No check sees a missing line, so review holds it.
- **The release moves them.** The branch that sets a new version (Cutting a release, below) moves the Unreleased lines under `## [X.Y.Z]` with the date of the tag and leaves Unreleased empty.
- **The notes are its sections.** `tools/release_notes.py X.Y.Z` prints a version's section, with each relative link made a GitHub URL at the version's tag (at `main` for Unreleased) because a release page would not resolve it, which opens the release's notes above its downloads; it fails for a version without a section, or whose section lists no `- ` entry, and so does the release workflow's first job, before any build.
  The nightly pre-release's notes give the Unreleased section, or name the last release when it is empty.
- **The docs check holds its form.** The `changelog` check of the `docs` component ([TESTS](TESTS.md)) fails a changelog that Unreleased does not open, a version heading of another form or out of order, another kind of subsection, a reference-style link, which a printed section would lose, an entry of more than 160 characters or 2 sentences, and a `CMakeLists.txt` version without notes, so a version change that forgot the changelog fails at landing, not at the tag.

## Cutting a release

1. Pick a `main` commit whose landing gates passed ([STATUS](STATUS.md)) and whose hosted run is green.
2. On a branch off it, set the version in both places the build reads (AGENTS.md, Checkpoints): `project(llmx VERSION X.Y.Z ...)` in `CMakeLists.txt` and `LLMX_RELEASE_VERSION` in `src/config.hpp`, and move the Unreleased lines of `CHANGELOG.md` under `## [X.Y.Z] - YYYY-MM-DD` (The changelog, above).
   This change is docs-and-version only and lands like any other branch.
3. Tag the landed commit `vX.Y.Z` and push the tag to GitHub.
   The workflow's first job refuses a tag on a commit `main` does not hold, that differs from either version or whose version has no notes in `CHANGELOG.md`, and the build scripts refuse a binary whose `--version` lacks the commit or reports a dirty tree.
   The Linux job's 180-minute limit is twice an estimate: it stays untimed on a hosted runner until the first tag run, which is best pushed to a test repository first.
4. The workflow's jobs, which `.github/workflows/release-build.yml` holds for this workflow and the nightly one, build the tarball, the zip and the four images and push the images to GHCR by digest, signing what they have keys for and writing `SHA256SUMS`; the `draft` job gives the images their versioned tags and creates the draft release with every file attached, and the version's changelog section and the images' digests in its notes.
5. Run the smoke checklist below on the draft's files.
6. Once every smoke item passes, the release operator publishes the draft, which runs the workflow's `publish` job, and checks that job passed; after a failed item nothing is published, and the owner decides what happens to the draft.

The tarball can be built the same way on any Linux machine with Docker, from the repository root at the tagged commit:

```sh
docker run --rm -v "$PWD:/src" -w /src ubuntu:22.04 bash docker/release/build-linux.sh
```

It writes `dist/llmx-X.Y.Z-linux-x86_64.tar.gz`, and leaves its build tree in `build-release`, both of which git ignores.

## Smoke checklist before publishing

Each item runs on the draft's own files, checked against `SHA256SUMS` first.

- **Any Linux machine, no GPU stack:** in a plain `ubuntu:24.04` container, the tarball's `llmx --version` prints `X.Y.Z+g` and the tagged commit, and `llmx generate` on Qwen3-0.6B Q8_0 with `--device cpu` writes text.
  The workflow runs the `--version` half of this.
- **An MI50 on Linux:** the tarball's binary, and each image, run `generate` on Qwen3-0.6B Q8_0 with `--device vulkan:0` (the host's Vulkan loader and RADV for the tarball, the `vulkan` image's for itself) and with `--device rocm:0` (a ROCm 7 runtime on the host for the tarball, the `rocm7` and `rocm10` images' for themselves); each writes the same text as `--device cpu` for the first tokens.
- **A Radeon VII on Windows:** the zip's `llmx.exe --version`, and `generate` on Qwen3-0.6B Q8_0 with `--device vulkan:0`.
- **The companion:** on the MI50, against `llmx serve` with a key file, the tarball's `tools/chat_ui/server.py` and the `chat-X.Y.Z` image each stream a reply in the browser, and the tarball's `tools/health_dashboard/server.py` and the image's alternate command each chart the server's health.
- **Signatures, when the keys are set:** `minisign -Vm SHA256SUMS -p llmx.pub` and `cosign verify --key cosign.pub ghcr.io/mxxm-t/llmx:X.Y.Z-vulkan` (and `X.Y.Z-rocm7`, `X.Y.Z-rocm10`, `chat-X.Y.Z`) succeed.

Record the results, with each binary's `--version`, in the release's STATUS entry.

A failed smoke run keeps the draft, the tag and the images while the owner decides; the usual outcome deletes the draft release, the `vX.Y.Z` tag and the four version-tagged images `X.Y.Z-vulkan`, `X.Y.Z-rocm7`, `X.Y.Z-rocm10` and `chat-X.Y.Z` from GHCR, so nothing of that version stays pullable, and the fix lands on `main` and the release is cut again from step 1.

## The nightly channel

Every push to `main` on GitHub rebuilds one pre-release, `nightly`, in place: `.github/workflows/nightly.yml` runs the release's build jobs on that commit and replaces the release's files, so it holds the same artifacts as a version release under nightly names.

| Artifact | Contents |
|---|---|
| `llmx-nightly-linux-x86_64.tar.gz`, `llmx-nightly-windows-x64.zip` | the archives of a release, built from that commit |
| `SHA256SUMS`, `SHA256SUMS.minisig`, `llmx.pub`, `cosign.pub` | as for a release, signed with the same keys |
| `ghcr.io/mxxm-t/llmx:nightly-vulkan`, `nightly-rocm7`, `nightly-rocm10`, `nightly-chat` | the four images, also on Docker Hub as `mxxm/llmx:nightly-*`, signed |

- **The binary says it is a nightly build:** `--version` prints `X.Y.Z+nightly.YYYYMMDD.g<commit>`, where X.Y.Z is the release `main` carries and the date the commit's, in UTC ([BUILD](BUILD.md#the-build-identifier)); the build scripts refuse any other form.
- **No announcement:** the run moves the tag `nightly` to the commit and edits the release it already has, which stays a pre-release and is never marked latest, so watchers hear of it once, when the first run creates it.
  The floating `vulkan`, `rocm7`, `rocm10` and `chat` tags stay on the newest published release.
- **A newer push cancels an older build.** Its images then carry no tag, since a build pushes them by digest alone; the publish job tags them, mirrors them and replaces the files only while `main` still points at its commit, and in a concurrency group no push cancels, so the channel never moves backwards and never stops half replaced.
  While the files are replaced, for under a minute, `SHA256SUMS` may name the previous archives.
- **No smoke run checks it:** a nightly build has passed the landing gates of its commit, not the smoke checklist above.
- **Its notes list what is not yet released:** the Unreleased section of `CHANGELOG.md` (The changelog, above).
- **A manual run on another branch** (`workflow_dispatch`) builds every file and image and publishes nothing; a fork or the Gitea mirror runs none of it.

## Signing keys

The signing steps read their keys from these repository secrets and skip with a warning while a secret is unset, so a release without them ships unsigned.
The keys are the owner's: the workflow never creates one.
The owner stores the public halves as the repository variables `MINISIGN_PUBLIC_KEY` (the contents of the minisign `.pub` file) and `COSIGN_PUBLIC_KEY` (the contents of `cosign.pub`), and the `checksums` job attaches them to the release as `llmx.pub` and `cosign.pub`, or warns that it did not.

| Secret | Read by | Holds |
|---|---|---|
| `MINISIGN_SECRET_KEY` | `checksums` | the minisign secret key file's contents, which sign `SHA256SUMS` |
| `MINISIGN_PASSWORD` | `checksums` | that key's password |
| `COSIGN_PRIVATE_KEY` | `images`, `publish` and the nightly `publish` | a cosign private key (`cosign generate-key-pair`), which signs each image by digest |
| `COSIGN_PASSWORD` | `images`, `publish` and the nightly `publish` | that key's password |
| `DOCKERHUB_USERNAME` | `publish` and the nightly `publish` | the Docker Hub account the images are mirrored to, as `<account>/llmx` |
| `DOCKERHUB_TOKEN` | `publish` and the nightly `publish` | an access token of that account with write access |

GHCR needs no secret: the workflow pushes with its own token, under the repository's owner.

## License and contributions

llmx is MIT licensed (`LICENSE`), and the notices of the MIT code it contains are in `THIRD_PARTY_NOTICES.md`.
Before the first outside contribution is merged, the project either moves to Apache-2.0 or adopts a contributor agreement that allows relicensing; a sign-off alone does not allow it.
