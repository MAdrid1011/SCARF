#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${SCARF_CALIBRATION_CACHE:-${ROOT}/downloads/calibration}"
MIRROR="http://schadenfreude.csail.mit.edu:8000"

command -v curl >/dev/null || {
    echo "curl is required" >&2
    exit 2
}

mkdir -p "${CACHE}"

download() {
    local name="$1"
    local expected_size="$2"
    local archive="${CACHE}/${name}.zip"
    local partial="${archive}.partial"

    if [[ -f "${archive}" && "$(stat -c %s "${archive}")" == "${expected_size}" ]]; then
        printf '%s\n' "${archive}"
        return
    fi
    if [[ -f "${partial}" ]]; then
        local partial_size
        partial_size="$(stat -c %s "${partial}")"
        if [[ "${partial_size}" == "${expected_size}" ]]; then
            mv "${partial}" "${archive}"
            sha256sum "${archive}" >"${archive}.sha256"
            printf '%s\n' "${archive}"
            return
        fi
        if [[ "${SCARF_CALIBRATION_RESTART_PARTIAL:-0}" != "1" ]]; then
            cat >&2 <<EOF
${partial} is an incomplete ${name} download. The official mirror currently
does not support byte-range requests, so it cannot be resumed safely. The
partial file was preserved. Set SCARF_CALIBRATION_RESTART_PARTIAL=1 only to
discard it and restart the complete download.
EOF
            exit 2
        fi
        rm -f "${partial}"
    fi
    curl --fail --location --retry 8 --retry-delay 10 --connect-timeout 30 \
        --output "${partial}" "${MIRROR}/${name}.zip"
    [[ "$(stat -c %s "${partial}")" == "${expected_size}" ]] || {
        echo "Size mismatch for ${MIRROR}/${name}.zip" >&2
        exit 2
    }
    mv "${partial}" "${archive}"
    sha256sum "${archive}" >"${archive}.sha256"
    printf '%s\n' "${archive}"
}

download re10k 553972662205
download acid 173691377409
