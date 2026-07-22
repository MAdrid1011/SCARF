#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS_ROOT="${1:-${ROOT}/datasets}"
CACHE="${SCARF_DOWNLOAD_CACHE:-${ROOT}/downloads/re10k-acid-test}"
MIRROR="http://schadenfreude.csail.mit.edu:8000"

command -v curl >/dev/null || {
    echo "curl is required" >&2
    exit 2
}
command -v unzip >/dev/null || {
    echo "unzip is required" >&2
    exit 2
}

mkdir -p "${CACHE}" "${DATASETS_ROOT}"

download_archive() {
    local name="$1"
    local expected_size="$2"
    local archive="${CACHE}/${name}_test_only.zip"
    local partial="${archive}.partial"
    local url="${MIRROR}/${name}_test_only.zip"

    if [[ -f "${archive}" && "$(stat -c %s "${archive}")" == "${expected_size}" ]]; then
        printf '%s\n' "${archive}"
        return
    fi
    rm -f "${partial}"
    curl -L --fail --retry 3 --retry-delay 5 --connect-timeout 30 \
        --silent --show-error --output "${partial}" "${url}"
    [[ "$(stat -c %s "${partial}")" == "${expected_size}" ]] || {
        echo "Size mismatch for ${url}" >&2
        exit 2
    }
    mv "${partial}" "${archive}"
    printf '%s\n' "${archive}"
}

RE10K_ARCHIVE="$(download_archive re10k 55604889849)"
ACID_ARCHIVE="$(download_archive acid 34827045914)"

for archive in "${RE10K_ARCHIVE}" "${ACID_ARCHIVE}"; do
    unzip -o "${archive}" -d "${DATASETS_ROOT}"
done

for dataset in re10k acid; do
    dataset_root="${DATASETS_ROOT}/${dataset}"
    [[ -f "${dataset_root}/test/index.json" ]] || {
        echo "Prepared ${dataset} test/index.json is missing after extraction" >&2
        exit 2
    }
    archive="${CACHE}/${dataset}_test_only.zip"
    archive_sha256="$(sha256sum "${archive}" | cut -d ' ' -f 1)"
    source_url="${MIRROR}/${dataset}_test_only.zip"
    printf '{"archive_sha256":"%s","http_range_supported":false,"url":"%s"}\n' \
        "${archive_sha256}" "${source_url}" >"${dataset_root}/.scarf-source.json"
    python3 "${ROOT}/data/build_manifest.py" \
        --root "${dataset_root}" \
        --name "${dataset}" \
        --source "${source_url}" \
        --revision "archive-sha256:${archive_sha256}"
done
