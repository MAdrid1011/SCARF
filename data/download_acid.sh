#!/usr/bin/env bash
# Download ACID test split.
# Size: ~3 GB
#
# Usage:
#   bash data/download_acid.sh [OUTPUT_DIR]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${1:-$ROOT/datasets/acid}"
mkdir -p "$OUT_DIR"

echo "Downloading ACID test split to $OUT_DIR ..."

ACID_URL="https://huggingface.co/MAdrid1011/SCARF-checkpoints/resolve/main/datasets/acid_test.tar.gz"

TMP="$OUT_DIR/acid_test.tar.gz"
wget -q --show-progress -O "$TMP" "$ACID_URL"
tar -xzf "$TMP" -C "$OUT_DIR" --strip-components=1
rm "$TMP"

echo "ACID test split ready at: $OUT_DIR"
