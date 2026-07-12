#!/usr/bin/env bash
# Download ACID test split in MVSplat/TranSplat format.
#
# Source: https://drive.google.com/drive/folders/1joiezNCyQK2BvWMnfwHJpm2V77c7iYGe
# (same Google Drive folder as Re10K, contains both datasets)
#
# Usage:
#   bash data/download_acid.sh [OUTPUT_DIR]
#   default OUTPUT_DIR: datasets/acid

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${1:-$ROOT/datasets/acid}"
mkdir -p "$OUT_DIR"

if ! command -v gdown &>/dev/null; then
    echo "Installing gdown ..."
    pip install -q gdown
fi

echo "Downloading ACID test split to $OUT_DIR ..."
gdown --fuzzy \
    "https://drive.google.com/drive/folders/1joiezNCyQK2BvWMnfwHJpm2V77c7iYGe" \
    --folder -O "$OUT_DIR" -q

echo "ACID test split ready at: $OUT_DIR"
