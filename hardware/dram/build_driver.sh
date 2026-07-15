#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RAMULATOR_ROOT="${RAMULATOR_ROOT:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ramulator-root) RAMULATOR_ROOT="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "${RAMULATOR_ROOT}" ]] || { echo "RAMULATOR_ROOT or --ramulator-root is required" >&2; exit 2; }
cmake -S "${SCRIPT_DIR}" -B "${SCRIPT_DIR}/build" -DRAMULATOR_ROOT="${RAMULATOR_ROOT}"
cmake --build "${SCRIPT_DIR}/build" --parallel
