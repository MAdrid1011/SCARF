#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RAMULATOR_ROOT="${RAMULATOR_ROOT:-}"
DRAMPOWER_ROOT="${DRAMPOWER_ROOT:-}"
EVENTS=""
OUTPUT_DIR="${ROOT}/outputs/dram"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ramulator-root) RAMULATOR_ROOT="$2"; shift 2 ;;
        --drampower-root) DRAMPOWER_ROOT="$2"; shift 2 ;;
        --events) EVENTS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -n "${RAMULATOR_ROOT}" ]] || { echo "RAMULATOR_ROOT or --ramulator-root is required" >&2; exit 2; }
[[ -n "${DRAMPOWER_ROOT}" ]] || { echo "DRAMPOWER_ROOT or --drampower-root is required" >&2; exit 2; }
[[ -n "${EVENTS}" ]] || { echo "--events must name measured memory-events.jsonl" >&2; exit 2; }
mkdir -p "${OUTPUT_DIR}"
EVENTS_COPY="${OUTPUT_DIR}/memory-events.jsonl"
if [[ "$(realpath -- "${EVENTS}")" != "$(realpath -m -- "${EVENTS_COPY}")" ]]; then
    cp -- "${EVENTS}" "${EVENTS_COPY}"
fi

python3 "${SCRIPT_DIR}/export_trace.py" \
    --input "${EVENTS_COPY}" \
    --output "${OUTPUT_DIR}/ramulator.trace" \
    --manifest "${OUTPUT_DIR}/trace-manifest.json"
python3 "${SCRIPT_DIR}/run_ramulator.py" \
    --ramulator-root "${RAMULATOR_ROOT}" \
    --trace "${OUTPUT_DIR}/ramulator.trace" \
    --output-dir "${OUTPUT_DIR}"
python3 "${SCRIPT_DIR}/convert_commands.py" \
    --input "${OUTPUT_DIR}/ramulator_commands.csv.ch0" \
    --output "${OUTPUT_DIR}/drampower_commands.csv"
python3 "${SCRIPT_DIR}/run_drampower.py" \
    --drampower-root "${DRAMPOWER_ROOT}" \
    --command-trace "${OUTPUT_DIR}/drampower_commands.csv" \
    --output-dir "${OUTPUT_DIR}"
python3 "${SCRIPT_DIR}/collect.py" --output-dir "${OUTPUT_DIR}"
