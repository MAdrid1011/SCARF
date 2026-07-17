"""Fail-closed execution-dependency contracts for SAES S2/S3 accounting.

Tile routing statistics alone do not prove that a model can avoid dense S2/S3
work.  Each supported encoder needs target-free evidence that retained outputs
are independent of every skipped position before a SAES route may reduce the
S2/S3 cycle model.  Until then the route counters remain useful diagnostic and
control/merge-ledger inputs, but their executable bypass fraction is zero.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CONTRACT_VERSION = "saes-s2s3-execution-dependency-v1"

# These are architecture contracts, not quality results.  A later positive
# result must include a target-free dependency audit and a reconciled sparse
# implementation before this table can permit a nonzero cycle saving.
_MODEL_CONTRACTS: dict[str, dict[str, Any]] = {
    "transplat": {
        "model": "transplat",
        "s2_s3_sparse_execution_verified": False,
        "status": "dense_dependency_detected",
        "evidence": {
            "kind": "saes_s2s3_dense_dependency_audit",
            "dataset": "dl3dv",
            "sample_index": 0,
            "results_path": (
                "outputs/ae_dl3dv_repair_diagnostics/"
                "transplat_sample0_s2s3_dependency_audit_v2/results.json"
            ),
            "results_sha256": (
                "8a73027989cfaafce2145b6370d2209450725eb33dfbd7c9afa0e5c5ead82679"
            ),
        },
        "reason": (
            "zeroing non-probe refinement activations changed retained raw "
            "Gaussian-head values, so the unmodified dense path cannot "
            "directly bypass non-probe S2/S3 work"
        ),
    },
    "mvsplat": {
        "model": "mvsplat",
        "s2_s3_sparse_execution_verified": False,
        "status": "dense_dependency_detected",
        "evidence": {
            "kind": "saes_s2s3_dense_dependency_audit",
            "dataset": "dl3dv",
            "sample_index": 0,
            "results_path": (
                "outputs/ae_dl3dv_repair_diagnostics/"
                "mvsplat_sample0_s2s3_dependency_audit_v1/results.json"
            ),
            "results_sha256": (
                "ac5610772240887f7f5004c050fbc0ad08261337dd18b02f3430c7cf00174ea5"
            ),
        },
        "reason": (
            "zeroing non-probe refinement activations changed retained raw "
            "Gaussian-head values, so the unmodified dense path cannot "
            "directly bypass non-probe S2/S3 work"
        ),
    },
    "depthsplat": {
        "model": "depthsplat",
        "s2_s3_sparse_execution_verified": False,
        "status": "dense_dependency_detected",
        "evidence": {
            "kind": "saes_s2s3_dense_dependency_audit",
            "dataset": "dl3dv",
            "sample_index": 0,
            "results_path": (
                "outputs/ae_dl3dv_repair_diagnostics/"
                "depthsplat_sample0_s2s3_dependency_audit_v1/results.json"
            ),
            "results_sha256": (
                "893747ecb7d3336f90b9f7afdf052cd3d946d8728b37ec5ebf558533a4befd76"
            ),
        },
        "reason": (
            "zeroing non-probe Gaussian-regressor activations changed retained "
            "raw Gaussian-head values, so the unmodified dense path cannot "
            "directly bypass non-probe S2/S3 work"
        ),
    },
}


def resolve_s2_s3_execution_contract(model_type: str | None) -> dict[str, Any]:
    """Return a copy of the model-specific SAES sparse-execution contract.

    Missing model identity is intentionally accepted as an explicit unknown
    contract.  That keeps diagnostic callers executable while forcing their
    S2/S3 SAES savings to zero.  A misspelled nonempty model is an error rather
    than silently inheriting a different model's evidence.
    """
    if model_type is None:
        return {
            "contract_version": CONTRACT_VERSION,
            "model": None,
            "s2_s3_sparse_execution_verified": False,
            "status": "model_identity_missing",
            "evidence": None,
            "reason": "no model-specific sparse S2/S3 execution contract was supplied",
        }
    if not isinstance(model_type, str) or not model_type.strip():
        raise ValueError("model_type must be a nonempty string or None")
    normalized = model_type.strip().lower()
    if normalized not in _MODEL_CONTRACTS:
        raise ValueError(
            "unsupported SAES execution-dependency model: "
            f"{model_type}; expected one of {', '.join(sorted(_MODEL_CONTRACTS))}"
        )
    return {
        "contract_version": CONTRACT_VERSION,
        **deepcopy(_MODEL_CONTRACTS[normalized]),
    }


def s2_s3_saving_ratio(
    requested_ratio: float, execution_contract: dict[str, Any]
) -> float:
    """Gate a requested SAES S2/S3 bypass ratio on verified execution evidence."""
    if isinstance(requested_ratio, bool) or not isinstance(requested_ratio, (int, float)):
        raise ValueError("requested SAES saving ratio must be numeric")
    ratio = float(requested_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("requested SAES saving ratio must be in [0, 1]")
    if execution_contract.get("s2_s3_sparse_execution_verified") is not True:
        return 0.0
    return ratio
