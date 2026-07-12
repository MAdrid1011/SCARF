#!/usr/bin/env bash
# Download RealEstate10K (Re10K) test split in MVSplat/TranSplat format.
#
# Source: https://drive.google.com/drive/folders/1joiezNCyQK2BvWMnfwHJpm2V77c7iYGe
# (small subset provided by the pixelSplat / TranSplat authors, sufficient for evaluation)
#
# Full dataset conversion: https://github.com/dcharatan/real_estate_10k_tools
#
# Usage:
#   bash data/download_re10k.sh [OUTPUT_DIR]
#   default OUTPUT_DIR: datasets/re10k

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${1:-$ROOT/datasets/re10k}"
mkdir -p "$OUT_DIR"

if ! command -v gdown &>/dev/null; then
    echo "Installing gdown ..."
    pip install -q gdown
fi

echo "Downloading Re10K test split to $OUT_DIR ..."
# Folder ID from the TranSplat / pixelSplat data page
gdown --fuzzy \
    "https://drive.google.com/drive/folders/1joiezNCyQK2BvWMnfwHJpm2V77c7iYGe" \
    --folder -O "$OUT_DIR" -q

echo "Re10K test split ready at: $OUT_DIR"
