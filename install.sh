#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE=""
CHECK_ONLY=0
CPU_ONLY="${SCARF_CPU_ONLY:-0}"
VENV=""

usage() {
    echo "Usage: bash install.sh --profile classic|depthsplat|orin [--venv PATH] [--check-only]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) PROFILE="$2"; shift 2 ;;
        --venv) VENV="$2"; shift 2 ;;
        --check-only) CHECK_ONLY=1; shift ;;
        *) usage; exit 2 ;;
    esac
done

case "${PROFILE}" in
    classic|depthsplat|orin) ;;
    *) usage; exit 2 ;;
esac

GIT_TOPLEVEL="$(git -C "${ROOT}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ ! -f "${ROOT}/release-manifest.json" && "${GIT_TOPLEVEL}" == "${ROOT}" ]]; then
    git -C "${ROOT}" submodule update --init --recursive
fi

if [[ "${PROFILE}" == "orin" ]]; then
    [[ -z "${VENV}" ]] || {
        echo "The Orin profile uses the JetPack system environment; --venv is unsupported." >&2
        exit 2
    }
    [[ "${CHECK_ONLY}" == "1" ]] || {
        echo "The Orin profile uses NVIDIA's JetPack PyTorch stack; only --check-only is supported." >&2
        exit 2
    }
    python3 "${ROOT}/scripts/check_environment.py" --profile orin
    exit $?
fi

PYTHON_BIN="python3"
if [[ -n "${VENV}" ]]; then
    if [[ "${VENV}" != /* ]]; then VENV="${ROOT}/${VENV}"; fi
    if [[ ! -x "${VENV}/bin/python" ]]; then
        [[ "${CHECK_ONLY}" != "1" ]] || {
            echo "Virtual environment does not exist: ${VENV}" >&2
            exit 2
        }
        command -v python3.10 >/dev/null || {
            echo "python3.10 is required to create ${VENV}" >&2
            exit 2
        }
        python3.10 -m venv "${VENV}"
    fi
    PYTHON_BIN="${VENV}/bin/python"
    export PATH="${VENV}/bin:${PATH}"
fi

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"Python 3.10 is required, got {sys.version.split()[0]}")
PY

CC_BIN="$(command -v "${CC:-cc}" || true)"
CXX_BIN="$(command -v "${CXX:-c++}" || true)"
[[ -n "${CC_BIN}" ]] || { echo "C compiler is required" >&2; exit 2; }
[[ -n "${CXX_BIN}" ]] || { echo "C++ compiler is required" >&2; exit 2; }
if [[ "${CPU_ONLY}" != "1" ]]; then
    NVCC_BIN="$(command -v nvcc || true)"
    [[ -n "${NVCC_BIN}" ]] || { echo "CUDA nvcc is required" >&2; exit 2; }
    mapfile -t CUDA_COMPILERS < <(
        "${PYTHON_BIN}" "${ROOT}/scripts/select_cuda_compiler.py" \
            --nvcc "${NVCC_BIN}" --cc "${CC_BIN}" --cxx "${CXX_BIN}"
    )
    [[ "${#CUDA_COMPILERS[@]}" == "2" ]] || {
        echo "Unable to select a CUDA-compatible C/C++ compiler" >&2
        exit 2
    }
    export CC="${CUDA_COMPILERS[0]}"
    export CXX="${CUDA_COMPILERS[1]}"
    export CUDAHOSTCXX="${CXX}"
    echo "Using CUDA host compilers: CC=${CC} CXX=${CXX}"
fi

if [[ "${CHECK_ONLY}" != "1" ]]; then
    "${PYTHON_BIN}" -m pip install --upgrade pip==24.3.1 setuptools==75.6.0 wheel==0.45.1
    if [[ "${PROFILE}" == "classic" ]]; then
        if [[ "${CPU_ONLY}" == "1" ]]; then
            "${PYTHON_BIN}" -m pip install torch==2.1.2 torchvision==0.16.2 \
                --index-url https://download.pytorch.org/whl/cpu
        else
            "${PYTHON_BIN}" -m pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
                --index-url https://download.pytorch.org/whl/cu121
        fi
    else
        [[ "${CPU_ONLY}" != "1" ]] || {
            echo "DepthSplat requires its CUDA profile" >&2
            exit 2
        }
        "${PYTHON_BIN}" -m pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 \
            --index-url https://download.pytorch.org/whl/cu121
    fi
    "${PYTHON_BIN}" -m pip install -r "${ROOT}/environments/${PROFILE}/requirements.lock"
fi

"${PYTHON_BIN}" "${ROOT}/scripts/check_environment.py" --profile "${PROFILE}"
