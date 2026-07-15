#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
IFLOW_ROOT="${IFLOW_ROOT:-}"
IFLOW_IMAGE="${IFLOW_IMAGE:-iedaopensource/iflow:latest}"
PLATFORM="asap7"
STAGE="all"
OUTPUT_DIR="${ROOT}/outputs/physical/asap7"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform) PLATFORM="$2"; shift 2 ;;
        --stage) STAGE="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ "${PLATFORM}" == "asap7" ]] || { echo "only --platform asap7 is supported" >&2; exit 2; }
[[ -n "${IFLOW_ROOT}" ]] || { echo "IFLOW_ROOT must name a clean iFlow checkout" >&2; exit 2; }
if [[ "${OUTPUT_DIR}" != /* ]]; then OUTPUT_DIR="${ROOT}/${OUTPUT_DIR}"; fi
IFLOW_ROOT="$(realpath -- "${IFLOW_ROOT}")"
cd "${ROOT}"

mkdir -p "${OUTPUT_DIR}"
if [[ "${DRY_RUN}" == "1" ]]; then
    TEMP_PARENT="$(mktemp -d "${OUTPUT_DIR}/iflow-preflight.XXXXXX")"
    TEMP_ROOT="${TEMP_PARENT}/iflow"
    trap 'git -C "${IFLOW_ROOT}" worktree remove --force "${TEMP_ROOT}" >/dev/null 2>&1 || true; rm -rf "${TEMP_PARENT}"' EXIT
    git -C "${IFLOW_ROOT}" worktree add --detach "${TEMP_ROOT}" \
        04b4d98b1a69d00bbe04d52b09105667332a295d
    python3 "${SCRIPT_DIR}/preflight.py" \
        --iflow-root "${TEMP_ROOT}" \
        --stage "${STAGE}" \
        --output "${OUTPUT_DIR}/manifest.json" \
        --container-image "${IFLOW_IMAGE}"
    python3 -m json.tool "${OUTPUT_DIR}/manifest.json"
    exit 0
fi

python3 "${SCRIPT_DIR}/resource_guard.py" \
    --output "${OUTPUT_DIR}/resource-validation.json"

RTL_DIR="${OUTPUT_DIR}/rtl-validation"
bash "${ROOT}/scripts/run_rtl.sh" --output-dir "${RTL_DIR}"

RUNTIME="${OUTPUT_DIR}/runtime/iflow"
[[ ! -e "${RUNTIME}" ]] || { echo "runtime already exists: ${RUNTIME}" >&2; exit 1; }
mkdir -p "$(dirname -- "${RUNTIME}")"
git -C "${IFLOW_ROOT}" worktree add --detach "${RUNTIME}" \
    04b4d98b1a69d00bbe04d52b09105667332a295d

python3 "${SCRIPT_DIR}/preflight.py" \
    --iflow-root "${RUNTIME}" \
    --stage "${STAGE}" \
    --output "${OUTPUT_DIR}/manifest.json" \
    --container-image "${IFLOW_IMAGE}"
RESOLVED_IFLOW_IMAGE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["container"]["image_id"])' "${OUTPUT_DIR}/manifest.json")"

# iFlow 04b4d98 resolves tools through a hard-coded sibling named "iFlow".
LEGACY_RUNTIME_LINK="$(dirname -- "${RUNTIME}")/iFlow"
[[ ! -e "${LEGACY_RUNTIME_LINK}" ]] || { echo "legacy runtime link exists: ${LEGACY_RUNTIME_LINK}" >&2; exit 1; }
ln -s "$(basename -- "${RUNTIME}")" "${LEGACY_RUNTIME_LINK}"

python3 "${SCRIPT_DIR}/materialize.py" \
    --runtime "${RUNTIME}" \
    --rtl "${RTL_DIR}/rtl/ScarfTop.sv"

STAGES="$(python3 -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["stages"]))' "${OUTPUT_DIR}/manifest.json")"
mkdir -p "${RUNTIME}/result" "${RUNTIME}/report" "${RUNTIME}/log" "${RUNTIME}/work"
DOCKER_MOUNTS=(
    -v "${RUNTIME}/scripts:/opt/iFlow/scripts"
    -v "${RUNTIME}/rtl:/opt/iFlow/rtl"
    -v "${RUNTIME}/foundry:/opt/iFlow/foundry"
    -v "${RUNTIME}/result:/opt/iFlow/result"
    -v "${RUNTIME}/report:/opt/iFlow/report"
    -v "${RUNTIME}/log:/opt/iFlow/log"
    -v "${RUNTIME}/work:/opt/iFlow/work"
)

IFS=',' read -r -a STAGE_LIST <<<"${STAGES}"
: >"${OUTPUT_DIR}/iflow.log"
for flow_stage in "${STAGE_LIST[@]}"; do
    SCARF_IFLOW_RUNTIME="${RUNTIME}" SCARF_IFLOW_IMAGE="${RESOLVED_IFLOW_IMAGE}" \
        /usr/bin/time -v -o "${OUTPUT_DIR}/time-${flow_stage}.log" \
        bash "hardware/iflow/run_stage.sh" "${flow_stage}" \
        2>&1 | tee "${OUTPUT_DIR}/iflow-${flow_stage}.log" \
        | tee -a "${OUTPUT_DIR}/iflow.log"
    python3 "${SCRIPT_DIR}/validate_stages.py" \
        --runtime "${RUNTIME}" \
        --stage "${flow_stage}" \
        --output "${OUTPUT_DIR}/stage-${flow_stage}-validation.json"
done

python3 "${SCRIPT_DIR}/validate_stages.py" \
    --runtime "${RUNTIME}" \
    --stage "${STAGE}" \
    --output "${OUTPUT_DIR}/stage-validation.json"

if [[ "${STAGES##*,}" != "layout" ]]; then
    echo "Requested iFlow stages completed; routed PPA requires --stage all."
    exit 0
fi

python3 "${SCRIPT_DIR}/report_ppa.py" \
    --runtime "${RUNTIME}" \
    --output "${OUTPUT_DIR}/ppa-report.log" \
    --path-root /opt/iFlow \
    --prepare-only

docker run --rm --user "$(id -u):$(id -g)" \
    "${DOCKER_MOUNTS[@]}" \
    -v "${OUTPUT_DIR}:/opt/scarf-output" \
    -w /opt/iFlow "${RESOLVED_IFLOW_IMAGE}" \
    /opt/iFlow/tools/OpenROADae191807/bin/openroad /opt/scarf-output/ppa-report.tcl \
    2>&1 | tee "${OUTPUT_DIR}/ppa-report.log"

python3 "${SCRIPT_DIR}/physical_result.py" \
    --runtime "${RUNTIME}" \
    --manifest "${OUTPUT_DIR}/manifest.json" \
    --output "${OUTPUT_DIR}/ppa.json"
