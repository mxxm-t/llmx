#!/usr/bin/env bash
# Sets the Docker Hub overview of the release images from docker/hub-overview.md, X.Y.Z replaced by VERSION (docs/RELEASE.md, The images).
# It reads DOCKERHUB_USERNAME and DOCKERHUB_TOKEN and prints neither the token nor the login's JWT.
set -euo pipefail

version=${1:?usage: hub-overview.sh VERSION}
here=$(cd "$(dirname "$0")" && pwd)
if [ -z "${DOCKERHUB_TOKEN:-}" ]; then echo "::warning::DOCKERHUB_TOKEN is not set, the Docker Hub overview is not updated"; exit 0; fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
jq -n --arg u "$DOCKERHUB_USERNAME" --arg p "$DOCKERHUB_TOKEN" '{username: $u, password: $p}' > "$work/login.json"
curl -sS -o "$work/login.out" -w '%{http_code}' -H 'Content-Type: application/json' -d @"$work/login.json" https://hub.docker.com/v2/users/login > "$work/code"
if [ "$(cat "$work/code")" != 200 ]; then echo "Docker Hub login failed with HTTP $(cat "$work/code")" >&2; exit 1; fi
jq -n --rawfile d "$here/hub-overview.md" --arg v "$version" \
    '{description: "Run large language models locally on CPU, Vulkan and ROCm GPUs; OpenAI-compatible server. GGUF.", full_description: ($d | split("X.Y.Z") | join($v))}' > "$work/body.json"
jq -r '"Authorization: JWT " + .token' "$work/login.out" > "$work/header"
code=$(curl -sS -o /dev/null -w '%{http_code}' -X PATCH -H @"$work/header" -H 'Content-Type: application/json' -d @"$work/body.json" "https://hub.docker.com/v2/repositories/$DOCKERHUB_USERNAME/llmx/")
case "$code" in 2??) echo "Docker Hub overview set for $version" ;; *) echo "Docker Hub overview update failed with HTTP $code" >&2; exit 1 ;; esac
