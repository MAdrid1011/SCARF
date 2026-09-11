#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/scarf
OUTPUT_ROOT="${SCARF_OUTPUT_ROOT:-/results}"

usage() {
    cat <<'EOF'
Usage: docker run --rm --gpus all -v "$PWD/outputs:/results" scarf-ae:1.0.4 [quick [run_ae options...]]

The default command downloads the hash-pinned quick checkpoint if needed and
runs the CUDA Functional quick workflow. Results are written to /results.
EOF
}

case "${1:-quick}" in
    -h|--help|help)
        usage
        exit 0
        ;;
    quick)
        if [[ $# -gt 0 ]]; then
            shift
        fi
        ;;
    *)
        exec "$@"
        ;;
esac

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT"
bash data/download_checkpoints.sh --profile quick
"$SCARF_PYTHON_CLASSIC" data/build_quick_dataset.py \
    --output datasets/quick-re10k
exec bash scripts/run_ae.sh quick --output-root "$OUTPUT_ROOT" "$@"
