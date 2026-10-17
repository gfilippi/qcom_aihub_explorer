#!/usr/bin/env bash
# Fetch qualcomm/ai-hub-models at a given release tag into tools/ai-hub-models-<tag>/.
# Usage: tools/fetch_aihub_repo.sh [TAG]      (default: latest v* tag)
#        tools/fetch_aihub_repo.sh --list     (list available release tags)
set -euo pipefail

REPO_URL="https://github.com/qualcomm/ai-hub-models.git"
TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

list_tags() {
    git ls-remote --tags --refs "$REPO_URL" | awk -F/ '{print $3}' | grep -E '^v[0-9]' | sort -V
}

if [[ "${1:-}" == "--list" ]]; then
    list_tags
    exit 0
fi

TAG="${1:-$(list_tags | tail -1)}"
DEST="$TOOLS_DIR/ai-hub-models-$TAG"

if [[ -d "$DEST/.git" ]]; then
    echo "Already present: $DEST"
else
    git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$TAG" "$REPO_URL" "$DEST"
    echo "Fetched $TAG into $DEST"
fi
