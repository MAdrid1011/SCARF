#!/usr/bin/env bash
# SCARF — one-command environment setup
# Creates conda env 'scarf' with PyTorch 2.1.2+cu121 and all dependencies.
#
# Usage:
#   conda create -n scarf python=3.10 -y && conda activate scarf
#   bash install.sh
#
# CPU-only (no CUDA):
#   SCARF_CPU_ONLY=1 bash install.sh

set -euo pipefail

CPU_ONLY="${SCARF_CPU_ONLY:-0}"

echo "=== SCARF install ==="
echo "Python: $(python --version)"

if [ "$CPU_ONLY" = "1" ]; then
    echo "Mode: CPU-only"
    pip install torch==2.1.2 torchvision==0.16.2 \
        --extra-index-url https://download.pytorch.org/whl/cpu
else
    echo "Mode: CUDA 12.1"
    pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121
fi

pip install -r requirements.txt

echo ""
echo "=== SCARF install complete ==="
echo "Run 'bash scripts/run_ae.sh quick' to verify."
