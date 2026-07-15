#!/usr/bin/env bash
set -euo pipefail

RUNTIME="${SCARF_IFLOW_RUNTIME:?SCARF_IFLOW_RUNTIME is required}"
IMAGE="${SCARF_IFLOW_IMAGE:?SCARF_IFLOW_IMAGE is required}"
STAGE="${1:?iFlow stage is required}"

docker run --rm --user "$(id -u):$(id -g)" \
    -v "${RUNTIME}/scripts:/opt/iFlow/scripts" \
    -v "${RUNTIME}/rtl:/opt/iFlow/rtl" \
    -v "${RUNTIME}/foundry:/opt/iFlow/foundry" \
    -v "${RUNTIME}/result:/opt/iFlow/result" \
    -v "${RUNTIME}/report:/opt/iFlow/report" \
    -v "${RUNTIME}/log:/opt/iFlow/log" \
    -v "${RUNTIME}/work:/opt/iFlow/work" \
    -w /opt/iFlow/scripts "${IMAGE}" \
    ./run_flow.py -d ScarfTop -s "${STAGE}" \
    -f asap7 -t HS -c TYP -v AE -l AE
