#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${1:-${ROOT}/datasets/dl3dv}"
RAW_DIR="${SCARF_DL3DV_RAW:-${ROOT}/downloads/dl3dv-benchmark}"
REVISION="9684e8382278c5e18173c1e72bd246daf2874539"
SOURCE="https://huggingface.co/datasets/DL3DV/DL3DV-10K-Benchmark"

python3 -c 'import huggingface_hub' >/dev/null 2>&1 || {
    echo "huggingface_hub is required; install the locked profile first" >&2
    exit 2
}
python3 "${ROOT}/data/download_dl3dv_benchmark.py" \
    --output-dir "${RAW_DIR}" \
    --index "${ROOT}/depthsplat/assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json" \
    --revision "${REVISION}" \
    --workers "${SCARF_DL3DV_WORKERS:-8}"

python3 "${ROOT}/data/convert_dl3dv.py" \
    --input-dir "${RAW_DIR}" \
    --output-dir "${OUT_ROOT}/native" \
    --scene-source-plan "${RAW_DIR}/.scarf-dl3dv-source.json" \
    --scene-source-key native \
    --target-height 270 \
    --target-width 480 \
    --schema depthsplat-native-270x480-v1 \
    --source "${SOURCE}" \
    --revision "${REVISION}" \
    --scene-index "${ROOT}/depthsplat/assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json"
python3 "${ROOT}/data/generate_chunk_index.py" --stage "${OUT_ROOT}/native/test"
python3 "${ROOT}/data/build_manifest.py" \
    --root "${OUT_ROOT}/native" \
    --name dl3dv-native \
    --source "${SOURCE}" \
    --revision "${REVISION}"

python3 "${ROOT}/data/convert_dl3dv.py" \
    --input-dir "${RAW_DIR}" \
    --output-dir "${OUT_ROOT}/re10k" \
    --scene-source-plan "${RAW_DIR}/.scarf-dl3dv-source.json" \
    --scene-source-key re10k \
    --target-height 360 \
    --target-width 640 \
    --resize \
    --schema re10k-compatible-360x640-v1 \
    --source "${SOURCE}" \
    --revision "${REVISION}" \
    --scene-index "${ROOT}/depthsplat/assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json"
python3 "${ROOT}/data/generate_chunk_index.py" --stage "${OUT_ROOT}/re10k/test"
python3 "${ROOT}/data/build_manifest.py" \
    --root "${OUT_ROOT}/re10k" \
    --name dl3dv-re10k \
    --source "${SOURCE}" \
    --revision "${REVISION}"
