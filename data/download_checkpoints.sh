#!/usr/bin/env bash
# Download pre-trained checkpoints for TranSplat, MVSplat, and DepthSplat
# from their original public repositories.
#
# Sources:
#   TranSplat  — https://huggingface.co/xingyoujun/transplat
#   MVSplat    — https://drive.google.com/drive/folders/14_E_5R6ojOWnLSrSVLVEMHnTiKsfddjU
#   DepthSplat — https://huggingface.co/haofeixu/depthsplat
#
# Usage:
#   bash data/download_checkpoints.sh
#
# Requires: wget (for HuggingFace), gdown (for Google Drive)
#   pip install gdown

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# TranSplat — HuggingFace
# ---------------------------------------------------------------------------
TRANSPLAT_CKPT="$ROOT/transplat/checkpoints/re10k.ckpt"
mkdir -p "$(dirname "$TRANSPLAT_CKPT")"
if [ ! -f "$TRANSPLAT_CKPT" ]; then
    echo "Downloading TranSplat re10k.ckpt from HuggingFace ..."
    wget -q --show-progress \
        "https://huggingface.co/xingyoujun/transplat/resolve/main/re10k.ckpt" \
        -O "$TRANSPLAT_CKPT"
else
    echo "TranSplat: $TRANSPLAT_CKPT already exists, skipping."
fi

# ---------------------------------------------------------------------------
# MVSplat — Google Drive (requires gdown)
# ---------------------------------------------------------------------------
MVSPLAT_CKPT="$ROOT/mvsplat/checkpoints/re10k.ckpt"
mkdir -p "$(dirname "$MVSPLAT_CKPT")"
if [ ! -f "$MVSPLAT_CKPT" ]; then
    echo "Downloading MVSplat re10k.ckpt from Google Drive ..."
    if ! command -v gdown &>/dev/null; then
        echo "  gdown not found. Installing ..."
        pip install -q gdown
    fi
    # File ID for re10k.ckpt inside folder 14_E_5R6ojOWnLSrSVLVEMHnTiKsfddjU
    gdown --fuzzy \
        "https://drive.google.com/drive/folders/14_E_5R6ojOWnLSrSVLVEMHnTiKsfddjU" \
        --folder -O "$(dirname "$MVSPLAT_CKPT")" -q
    # gdown downloads folder contents; rename if needed
    if [ ! -f "$MVSPLAT_CKPT" ] && [ -f "$(dirname "$MVSPLAT_CKPT")/re10k.ckpt" ]; then
        mv "$(dirname "$MVSPLAT_CKPT")/re10k.ckpt" "$MVSPLAT_CKPT"
    fi
else
    echo "MVSplat: $MVSPLAT_CKPT already exists, skipping."
fi

# ---------------------------------------------------------------------------
# DepthSplat — HuggingFace (large model, re10k 256×256)
# ---------------------------------------------------------------------------
DEPTHSPLAT_CKPT="$ROOT/depthsplat/checkpoints/re10k.ckpt"
mkdir -p "$(dirname "$DEPTHSPLAT_CKPT")"
if [ ! -f "$DEPTHSPLAT_CKPT" ]; then
    echo "Downloading DepthSplat re10k checkpoint from HuggingFace ..."
    wget -q --show-progress \
        "https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth" \
        -O "$DEPTHSPLAT_CKPT"
else
    echo "DepthSplat: $DEPTHSPLAT_CKPT already exists, skipping."
fi

echo ""
echo "=== Checkpoint download complete ==="
for f in "$TRANSPLAT_CKPT" "$MVSPLAT_CKPT" "$DEPTHSPLAT_CKPT"; do
    if [ -f "$f" ]; then
        echo "  OK  $f"
    else
        echo "  MISSING  $f"
    fi
done
