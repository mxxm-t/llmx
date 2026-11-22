#!/usr/bin/env bash
# The release images' one owner: their targets and tag names, pushing by digest, tagging, the Docker Hub mirror and cosign signatures (docs/RELEASE.md, The images).
# It reads GITHUB_REPOSITORY, GITHUB_SHA, the DOCKERHUB_ and COSIGN_ secrets and a docker logged in to GHCR, and docs/RELEASE.md gives its two commands.
set -euo pipefail

COSIGN_VERSION=v2.6.5
COSIGN_SHA256=c3b4f5410e608af03a5eb0aaac84a4313d8da131248e08ff1759ac70c79d1644
TARGETS="vulkan rocm7 rocm10 chat"
image="ghcr.io/${GITHUB_REPOSITORY,,}"

# X.Y.Z-vulkan, X.Y.Z-rocm7, X.Y.Z-rocm10 and chat-X.Y.Z for a release, nightly-TARGET for the nightly channel.
tag_of() {
    if [ "$2" = chat ] && [ "$1" != nightly ]; then echo "chat-$1"; else echo "$1-$2"; fi
}

sign() {
    if [ $# -eq 0 ]; then return; fi
    if [ -z "${COSIGN_PRIVATE_KEY:-}" ]; then echo "::warning::COSIGN_PRIVATE_KEY is not set, the images are not signed" >&2; return; fi
    local dir
    dir=$(mktemp -d)
    curl -fsSL -o "$dir/cosign" "https://github.com/sigstore/cosign/releases/download/$COSIGN_VERSION/cosign-linux-amd64"
    echo "$COSIGN_SHA256  $dir/cosign" | sha256sum -c - >&2
    chmod +x "$dir/cosign"
    for ref in "$@"; do "$dir/cosign" sign --yes --key env://COSIGN_PRIVATE_KEY "$ref" >&2; done
    rm -rf "$dir"
}

digest_of() {
    docker buildx imagetools inspect "$1" --format '{{json .Manifest.Digest}}' | tr -d '"'
}

build() {
    local version=$1 context=$2 push=${3:-} target meta refs=()
    # A pushed image carries no tag yet, so a build that fails or is overtaken never moves one; `tag` names it once its run decides to publish.
    docker buildx create --use --driver docker-container > /dev/null
    for target in $TARGETS; do
        meta=$(mktemp)
        docker buildx build --target "$target" -f docker/release/Dockerfile \
            --label "org.opencontainers.image.source=https://github.com/${GITHUB_REPOSITORY}" \
            --label "org.opencontainers.image.version=$version" \
            --label "org.opencontainers.image.revision=${GITHUB_SHA}" \
            --metadata-file "$meta" \
            ${push:+--output "type=image,name=$image,push-by-digest=true,push=true"} "$context" >&2
        if [ -n "$push" ]; then
            refs+=("$target=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["containerimage.digest"])' "$meta")")
        fi
        rm -f "$meta"
    done
    sign "${refs[@]/#*=/$image@}"
    printf '%s\n' "${refs[@]}"
}

tag() {
    local name=$1 hub="" floating="" target ref pair copies=() names
    shift
    while [ $# -gt 0 ] && { [ "$1" = hub ] || [ "$1" = floating ]; }; do
        if [ "$1" = hub ]; then hub=1; else floating=1; fi
        shift
    done
    # The sources are the digests a build printed, or with `tags` NAME's tags on GHCR; no source is an error, never a default.
    if [ "${1:-}" = tags ] && [ $# -eq 1 ]; then
        set --
        for target in $TARGETS; do set -- "$@" "$target=$image:$(tag_of "$name" "$target")"; done
    fi
    if [ $# -ne "$(wc -w <<< "$TARGETS")" ]; then echo "images.sh tag: want one source for each of $TARGETS, got: $*" >&2; exit 2; fi
    if [ -n "$hub" ]; then
        if [ -n "${DOCKERHUB_TOKEN:-}" ]; then
            echo "$DOCKERHUB_TOKEN" | docker login docker.io -u "$DOCKERHUB_USERNAME" --password-stdin
            hub="docker.io/$DOCKERHUB_USERNAME/llmx"
        else
            echo "::warning::DOCKERHUB_TOKEN is not set, Docker Hub is not mirrored"
            hub=""
        fi
    fi
    for pair in "$@"; do
        target=${pair%%=*}
        ref=${pair#*=}
        case "$ref" in sha256:*) ref="$image@$ref" ;; esac
        names=(-t "$image:$(tag_of "$name" "$target")")
        if [ -n "$floating" ]; then names+=(-t "$image:$target"); fi
        docker buildx imagetools create "${names[@]}" "$ref"
        if [ -n "$hub" ]; then
            names=(-t "$hub:$(tag_of "$name" "$target")")
            if [ -n "$floating" ]; then names+=(-t "$hub:$target"); fi
            docker buildx imagetools create "${names[@]}" "$ref"
            copies+=("$hub@$(digest_of "$hub:$(tag_of "$name" "$target")")")
        fi
    done
    sign "${copies[@]}"
}

command=$1
shift
case "$command" in
    build) build "$@" ;;
    tag) tag "$@" ;;
    *) echo "usage: images.sh build|tag ..." >&2; exit 2 ;;
esac
