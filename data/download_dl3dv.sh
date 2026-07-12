#!/usr/bin/env bash
# Download DL3DV test split.
# Size: ~4 GB
#
# Usage:
#   bash data/download_dl3dv.sh [OUTPUT_DIR]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${1:-$ROOT/datasets/dl3dv}"
mkdir -p "$OUT_DIR"

echo "Downloading DL3DV test split to $OUT_DIR ..."

DL3DV_URL="https://huggingface.co/MAdrid1011/SCARF-checkpoints/resolve/main/datasets/dl3dv_test.tar.gz"

TMP="$OUT_DIR/dl3dv_test.tar.gz"
wget -q --show-progress -O "$TMP" "$DL3DV_URL"
tar -xzf "$TMP" -C "$OUT_DIR" --strip-components=1
rm "$TMP"

echo "DL3DV test split ready at: $OUT_DIR"
