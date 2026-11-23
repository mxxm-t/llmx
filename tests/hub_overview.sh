#!/usr/bin/env bash
# Exercise the authenticated release script with a fake Docker Hub and the real jq.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp "$root/docker/hub-overview.sh" "$root/docker/hub-overview.md" "$work/"
echo 'Literal XxYyZ and placeholder X.Y.Z.' >> "$work/hub-overview.md"
export DOCKERHUB_USERNAME=fixture DOCKERHUB_TOKEN=fixture HUB_BODY="$work/body"

curl() {
    local url="${!#}" out="" data="" method=""
    while (($#)); do
        case "$1" in
            -o) out=$2; shift 2 ;;
            -d) data=$2; shift 2 ;;
            -X) method=$2; shift 2 ;;
            *) shift ;;
        esac
    done
    case "$url" in
        https://hub.docker.com/v2/users/login)
            printf '{"token":"fixture"}' > "$out"
            printf '200' ;;
        https://hub.docker.com/v2/repositories/fixture/llmx/)
            [[ "$method" == PATCH && "$data" == @* ]] || return 1
            cp "${data#@}" "$HUB_BODY"
            printf '200' ;;
        *) return 1 ;;
    esac
}
export -f curl
bash "$work/hub-overview.sh" 1.2.3
sed 's/X[.]Y[.]Z/1.2.3/g' "$work/hub-overview.md" > "$work/expected"
jq -j '.full_description' "$HUB_BODY" > "$work/actual"
cmp "$work/expected" "$work/actual"
