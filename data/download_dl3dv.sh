#!/usr/bin/env bash
# Download DL3DV test split.
#
# DL3DV is hosted at https://dl3dv-10k.github.io/DL3DV-10K/
# The preprocessed evaluation split used by DepthSplat is available at:
#   https://huggingface.co/datasets/haofeixu/depthsplat-data
#
# Usage:
#   bash data/download_dl3dv.sh [OUTPUT_DIR]
#   default OUTPUT_DIR: datasets/dl3dv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${1:-$ROOT/datasets/dl3dv}"
mkdir -p "$OUT_DIR"

echo "Downloading DL3DV test split to $OUT_DIR ..."
echo "Source: https://huggingface.co/datasets/haofeixu/depthsplat-data"
echo ""
echo "Using huggingface_hub snapshot download ..."

python3 - <<EOF
from huggingface_hub import snapshot_download
import shutil, pathlib

dest = pathlib.Path("$OUT_DIR")
tmp = snapshot_download(
    repo_id="haofeixu/depthsplat-data",
    repo_type="dataset",
    allow_patterns=["dl3dv/**"],
    local_dir=str(dest.parent / "depthsplat-data-tmp"),
)
src = pathlib.Path(tmp) / "dl3dv"
if src.exists():
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(src), str(dest))
    shutil.rmtree(str(pathlib.Path(tmp).parent / "depthsplat-data-tmp"), ignore_errors=True)
    print(f"DL3DV ready at: {dest}")
else:
    print(f"Downloaded to: {tmp}")
    print("Move the dl3dv/ subfolder to $OUT_DIR manually.")
EOF
