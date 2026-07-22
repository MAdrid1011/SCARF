#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${ROOT}/outputs/rtl"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            [[ $# -ge 2 ]] || { echo "--output-dir requires a value" >&2; exit 2; }
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${ROOT}/${OUTPUT_DIR}"
fi

for tool in sbt verilator python3; do
    command -v "${tool}" >/dev/null || { echo "required tool not found: ${tool}" >&2; exit 1; }
done

normalize_root_paths() {
    SCARF_LOG_ROOT="${ROOT}" python3 -c '
import os
import sys

root = os.environ["SCARF_LOG_ROOT"]
for line in sys.stdin:
    sys.stdout.write(line.replace(root, "$SCARF_ROOT"))
'
}

mkdir -p "${OUTPUT_DIR}/rtl" "${OUTPUT_DIR}/traces"

(
    cd "${ROOT}/chisel"
    sbt test 2>&1 | normalize_root_paths | tee "${OUTPUT_DIR}/sbt-test.log"
    rm -rf generated
    sbt "runMain scarf.VerilogEmitter" 2>&1 \
        | normalize_root_paths \
        | tee "${OUTPUT_DIR}/emit.log"
    cp generated/ScarfTop.sv "${OUTPUT_DIR}/rtl/ScarfTop.sv"
    cp generated/split/filelist.f "${OUTPUT_DIR}/rtl/filelist.f"
    verilator --lint-only --Wall -Wno-fatal --top-module ScarfTop \
        generated/ScarfTop.sv 2>&1 | tee "${OUTPUT_DIR}/verilator-lint.log"
)

VCD_PATH="$(find "${ROOT}/chisel/test_run_dir" -type f -name '*.vcd' -print -quit)"
[[ -n "${VCD_PATH}" ]] || { echo "representative VCD was not generated" >&2; exit 1; }
cp "${VCD_PATH}" "${OUTPUT_DIR}/traces/fsdr-cache.vcd"

python3 "${SCRIPT_DIR}/rtl_result.py" --output-dir "${OUTPUT_DIR}"
