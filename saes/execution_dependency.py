"""Fail-closed, stage-specific SAES execution contracts.

Route counters do not prove a whole encoder can skip every non-probe position.
The contracts below therefore distinguish dense shared work from the only
currently executable sparse primitive: same-weight selected outputs of the
classic two-convolution Gaussian head.  The primitive does not authorize a
global S2/S3 saving until a quality pipeline has actually bound and recorded
its selected-output events.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CONTRACT_VERSION = "saes-stage-specific-execution-dependency-v2"


def _dense_stage(reason: str, *, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "execution": "dense_required",
        "savings_permitted": False,
        "reason": reason,
        "evidence": evidence,
    }


def _classic_contract(
    model: str, *, results_path: str, results_sha256: str
) -> dict[str, Any]:
    dense_evidence = {
        "kind": "saes_s2s3_dense_dependency_audit",
        "dataset": "dl3dv",
        "sample_index": 0,
        "results_path": results_path,
        "results_sha256": results_sha256,
    }
    return {
        "model": model,
        "s2_s3_sparse_execution_verified": False,
        "status": "stage_specific_partial_replay",
        "stages": {
            "dense_shared_trunk": _dense_stage(
                "the raw-head masking audit changed retained values when "
                "non-probe refinement activations were zeroed; S1, cost-volume, "
                "refinement, and the first Gaussian-head convolution remain dense",
                evidence=dense_evidence,
            ),
            "s2_candidate_search": _dense_stage(
                "no probe-only candidate-search implementation has yet completed "
                "a target-free equivalence and quality-pipeline audit"
            ),
            "selected_output_head": {
                "execution": "same_weight_replay_available_audit_required",
                "savings_permitted": False,
                "head_structure": "Conv3x3->GELU->Conv3x3",
                "first_conv_closure": "dense_for_repeated_T4_corner_selection",
                "second_conv": "selected_outputs_only",
                "implementation": "saes.selected_output_replay",
                "audit_entrypoint": "scripts/saes_selected_output_replay_audit.py",
                "reason": (
                    "the standalone primitive is not yet bound into a quality "
                    "pipeline record with its actual selected-output event trace"
                ),
            },
            "s3_retained_descriptor": _dense_stage(
                "representative materialization has not passed the quality gate; "
                "descriptor merge work remains separately charged"
            ),
            "s4_conversion": _dense_stage(
                "GaussianAdapter conversion is not a zero-cost consequence of "
                "selected raw-head replay"
            ),
        },
        "reason": (
            "only a bounded selected-output replay primitive is available; no "
            "whole-encoder S2/S3 bypass is currently claimed"
        ),
    }


_MODEL_CONTRACTS: dict[str, dict[str, Any]] = {
    "transplat": _classic_contract(
        "transplat",
        results_path=(
            "outputs/ae_dl3dv_repair_diagnostics/"
            "transplat_sample0_s2s3_dependency_audit_v2/results.json"
        ),
        results_sha256="8a73027989cfaafce2145b6370d2209450725eb33dfbd7c9afa0e5c5ead82679",
    ),
    "mvsplat": _classic_contract(
        "mvsplat",
        results_path=(
            "outputs/ae_dl3dv_repair_diagnostics/"
            "mvsplat_sample0_s2s3_dependency_audit_v1/results.json"
        ),
        results_sha256="ac5610772240887f7f5004c050fbc0ad08261337dd18b02f3430c7cf00174ea5",
    ),
    "depthsplat": {
        "model": "depthsplat",
        "s2_s3_sparse_execution_verified": False,
        "status": "stage_specific_dense_closure",
        "stages": {
            "dense_shared_trunk": _dense_stage(
                "the Gaussian-regressor masking audit changed retained raw-head "
                "values; all upstream feature, S2, and regressor work remains dense",
                evidence={
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
            ),
            "s2_candidate_search": _dense_stage(
                "no probe-only candidate-search implementation has passed an audit"
            ),
            "selected_output_head": {
                "execution": "last_conv_replay_not_implemented",
                "savings_permitted": False,
                "first_three_conv_closure": "dense_for_repeated_T4_selection",
                "footprint_reference": "saes.depthsplat_s3_footprint",
                "reason": (
                    "the four-convolution adaptor needs an explicit same-weight "
                    "final-convolution replay before selected head work can be counted"
                ),
            },
            "s3_retained_descriptor": _dense_stage(
                "representative materialization has not passed the quality gate"
            ),
            "s4_conversion": _dense_stage(
                "GaussianAdapter conversion has no sparse replay evidence"
            ),
        },
        "reason": "DepthSplat has only the audited dense-closure footprint today",
    },
}


def resolve_s2_s3_execution_contract(model_type: str | None) -> dict[str, Any]:
    """Return a copy of the model-specific, fail-closed execution contract."""
    if model_type is None:
        return {
            "contract_version": CONTRACT_VERSION,
            "model": None,
            "s2_s3_sparse_execution_verified": False,
            "status": "model_identity_missing",
            "stages": {
                name: _dense_stage("no model-specific execution contract was supplied")
                for name in (
                    "dense_shared_trunk",
                    "s2_candidate_search",
                    "selected_output_head",
                    "s3_retained_descriptor",
                    "s4_conversion",
                )
            },
            "reason": "no model-specific sparse execution contract was supplied",
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
    """Gate a global S2/S3 bypass ratio on a full pipeline proof.

    A selected-head primitive is intentionally insufficient here: the caller's
    ``gauss_gen`` total also includes dense refinement and S4-adjacent work.
    """
    if isinstance(requested_ratio, bool) or not isinstance(requested_ratio, (int, float)):
        raise ValueError("requested SAES saving ratio must be numeric")
    ratio = float(requested_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("requested SAES saving ratio must be in [0, 1]")
    if execution_contract.get("s2_s3_sparse_execution_verified") is not True:
        return 0.0
    return ratio
