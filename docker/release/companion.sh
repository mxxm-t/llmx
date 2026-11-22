#!/usr/bin/env bash
# companion.sh DEST copies the chat companion and the health dashboard, tools/chat_ui and tools/health_dashboard as git tracks them, into DEST/tools without their tests (docs/RELEASE.md).
# The release archives and the chat image take these files from here alone, so every artifact ships the same set.
set -euo pipefail

dest=$(mkdir -p "$1" && cd "$1" && pwd)
cd "$(dirname "$0")/../.."
git ls-files tools/chat_ui tools/health_dashboard | grep -v -e '/test_' -e '\.mjs$' | while read -r file; do
    mkdir -p "$dest/$(dirname "$file")"
    cp "$file" "$dest/$file"
done
test -f "$dest/tools/chat_ui/server.py" && test -f "$dest/tools/health_dashboard/server.py"
