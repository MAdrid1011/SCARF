#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="all"
if [[ $# -gt 0 ]]; then
    [[ "$1" == "--profile" && $# -eq 2 ]] || {
        echo "Usage: bash data/download_checkpoints.sh [--profile quick|classic|depthsplat|all]" >&2
        exit 2
    }
    PROFILE="$2"
fi
python3 "${ROOT}/data/download_assets.py" \
    --profile "${PROFILE}" \
    --manifest "${ROOT}/artifact/manifests/checkpoints.json"
exec python3 "${ROOT}/data/download_assets.py" \
    --profile "${PROFILE}" \
    --manifest "${ROOT}/artifact/manifests/runtime_assets.json"
