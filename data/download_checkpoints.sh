#!/usr/bin/env bash
# Download pre-trained checkpoints for TranSplat, MVSplat, and DepthSplat.
# Total size: ~3 GB
#
# Usage:
#   bash data/download_checkpoints.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CKPT_DIR="$ROOT/checkpoints"
mkdir -p "$CKPT_DIR"

echo "Downloading checkpoints to $CKPT_DIR ..."

# --- TranSplat (re10k) ---
TRANSPLAT_URL="https://huggingface.co/MAdrid1011/SCARF-checkpoints/resolve/main/transplat_re10k.ckpt"
if [ ! -f "$CKPT_DIR/transplat_re10k.ckpt" ]; then
    echo "  transplat_re10k.ckpt ..."
    wget -q --show-progress -O "$CKPT_DIR/transplat_re10k.ckpt" "$TRANSPLAT_URL"
else
    echo "  transplat_re10k.ckpt already exists, skipping."
fi

# --- MVSplat (re10k) ---
MVSPLAT_URL="https://huggingface.co/MAdrid1011/SCARF-checkpoints/resolve/main/mvsplat_re10k.ckpt"
if [ ! -f "$CKPT_DIR/mvsplat_re10k.ckpt" ]; then
    echo "  mvsplat_re10k.ckpt ..."
    wget -q --show-progress -O "$CKPT_DIR/mvsplat_re10k.ckpt" "$MVSPLAT_URL"
else
    echo "  mvsplat_re10k.ckpt already exists, skipping."
fi

# --- DepthSplat (re10k) ---
DEPTHSPLAT_URL="https://huggingface.co/MAdrid1011/SCARF-checkpoints/resolve/main/depthsplat_re10k.ckpt"
if [ ! -f "$CKPT_DIR/depthsplat_re10k.ckpt" ]; then
    echo "  depthsplat_re10k.ckpt ..."
    wget -q --show-progress -O "$CKPT_DIR/depthsplat_re10k.ckpt" "$DEPTHSPLAT_URL"
else
    echo "  depthsplat_re10k.ckpt already exists, skipping."
fi

echo ""
echo "Checkpoints downloaded to: $CKPT_DIR"
echo "  $(ls -1 "$CKPT_DIR"/*.ckpt 2>/dev/null | wc -l) checkpoint(s) ready."
