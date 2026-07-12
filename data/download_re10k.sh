#!/usr/bin/env bash
# Download RealEstate10K (Re10K) test split.
# Size: ~3 GB
#
# Usage:
#   bash data/download_re10k.sh [OUTPUT_DIR]
#
# Default OUTPUT_DIR: datasets/re10k

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${1:-$ROOT/datasets/re10k}"
mkdir -p "$OUT_DIR"

echo "Downloading Re10K test split to $OUT_DIR ..."

# Pre-packaged test split (same format as MVSplat/TranSplat training data)
RE10K_URL="https://huggingface.co/MAdrid1011/SCARF-checkpoints/resolve/main/datasets/re10k_test.tar.gz"

TMP="$OUT_DIR/re10k_test.tar.gz"
wget -q --show-progress -O "$TMP" "$RE10K_URL"
tar -xzf "$TMP" -C "$OUT_DIR" --strip-components=1
rm "$TMP"

echo "Re10K test split ready at: $OUT_DIR"
