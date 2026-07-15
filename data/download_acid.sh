#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${ROOT}/data/download_re10k_acid.sh" "${ROOT}/datasets"
