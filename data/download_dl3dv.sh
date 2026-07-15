#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${1:-${ROOT}/datasets/dl3dv}"
RAW_DIR="${SCARF_DL3DV_RAW:-${ROOT}/downloads/dl3dv-benchmark}"
REVISION="9684e8382278c5e18173c1e72bd246daf2874539"
SOURCE="huggingface:DL3DV/DL3DV-Benchmark"

python3 -c 'import huggingface_hub' >/dev/null 2>&1 || {
    echo "huggingface_hub is required; install the locked profile first" >&2
    exit 2
}
python3 - <<PY
from huggingface_hub import get_token, snapshot_download

if get_token() is None:
    raise SystemExit(
        "DL3DV-Benchmark is gated. Accept its Hugging Face access terms and "
        "run `hf auth login` before this command."
    )
snapshot_download(
    repo_id="DL3DV/DL3DV-Benchmark",
    repo_type="dataset",
    revision="${REVISION}",
    local_dir="${RAW_DIR}",
    allow_patterns=[
        "*/nerfstudio/transforms.json",
        "*/nerfstudio/images_4/*",
        "*/nerfstudio/images_8/*",
    ],
)
PY

python3 "${ROOT}/data/convert_dl3dv.py" \
    --input-dir "${RAW_DIR}" \
    --output-dir "${OUT_ROOT}/native" \
    --image-subdir images_8 \
    --target-height 270 \
    --target-width 480 \
    --schema depthsplat-native-270x480-v1 \
    --source "${SOURCE}" \
    --revision "${REVISION}"
python3 "${ROOT}/data/generate_chunk_index.py" --stage "${OUT_ROOT}/native/test"
python3 "${ROOT}/data/build_manifest.py" \
    --root "${OUT_ROOT}/native" \
    --name dl3dv-native \
    --source "${SOURCE}" \
    --revision "${REVISION}"

python3 "${ROOT}/data/convert_dl3dv.py" \
    --input-dir "${RAW_DIR}" \
    --output-dir "${OUT_ROOT}/re10k" \
    --image-subdir images_4 \
    --target-height 360 \
    --target-width 640 \
    --resize \
    --schema re10k-compatible-360x640-v1 \
    --source "${SOURCE}" \
    --revision "${REVISION}"
python3 "${ROOT}/data/generate_chunk_index.py" --stage "${OUT_ROOT}/re10k/test"
python3 "${ROOT}/data/build_manifest.py" \
    --root "${OUT_ROOT}/re10k" \
    --name dl3dv-re10k \
    --source "${SOURCE}" \
    --revision "${REVISION}"
