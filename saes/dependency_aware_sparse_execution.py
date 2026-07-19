"""Fail-closed contracts for a future dependency-aware S2/S3 producer.

This module deliberately does not execute a model, estimate a saving, or
produce Gaussian descriptors.  It defines the minimum source-input boundary
and per-operator phase ledger a native producer must satisfy before an
execution trace can be considered for a strict S2/S3 audit.  In particular,
the contract rejects dense S2/S3 intermediates at its root and makes a global
dependency closure visible instead of allowing it to be counted as sparse.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import torch


DEPENDENCY_AWARE_EXECUTION_VERSION = "saes-dependency-aware-sparse-execution-v1"
_PHASES = ("primary", "secondary", "full")
_DEPENDENCY_KINDS = frozenset(
    {
        "local_spatial",
        "global_reduction",
        "cross_view_attention",
        "resample",
        "full",
    }
)

# These values are explicitly rejected even if a future root-input contract
# accidentally grows too broad.  They are outputs of an already dense S2/S3
# path or target-side inputs, not legal roots for a sparse producer.
_FORBIDDEN_ROOT_FIELDS = frozenset(
    {
        "depths",
        "densities",
        "refine_out",
        "raw_head",
        "raw_gaussians",
        "gaussians",
        "descriptors",
        "target_rgb",
        "target_images",
        "target_camera",
        "target_intrinsics",
        "target_extrinsics",
        "target_indices",
        "secondary_request_mask",
        "full_request_mask",
    }
)


@dataclass(frozen=True)
class SparseProducerRootInputContract:
    """Declared pre-S2 roots for one native model-specific producer."""

    model: str
    required_fields: frozenset[str]


@dataclass(frozen=True)
class ValidatedSparseProducerRootInputs:
    """Key-only evidence that a producer started before dense S2/S3 outputs.

    Values are intentionally not retained here.  A source-bound producer owns
    those values, while this contract only records its allowed input boundary.
    """

    model: str
    field_names: tuple[str, ...]


_ROOT_INPUT_CONTRACTS: dict[str, SparseProducerRootInputContract] = {
    "transplat": SparseProducerRootInputContract(
        model="transplat",
        required_fields=frozenset(
            {
                "trans_features",
                "cnn_features",
                "da_depth",
                "dino_feature",
                "context_images",
                "context_intrinsics",
                "context_extrinsics",
                "near",
                "far",
                "primary_request_mask",
            }
        ),
    ),
    "mvsplat": SparseProducerRootInputContract(
        model="mvsplat",
        required_fields=frozenset(
            {
                "trans_features",
                "cnn_features",
                "context_images",
                "context_intrinsics",
                "context_extrinsics",
                "near",
                "far",
                "primary_request_mask",
            }
        ),
    ),
    "depthsplat": SparseProducerRootInputContract(
        model="depthsplat",
        required_fields=frozenset(
            {
                "cnn_features",
                "mv_features",
                "mono_features",
                "context_images",
                "context_intrinsics",
                "context_extrinsics",
                "min_depth",
                "max_depth",
                "primary_request_mask",
            }
        ),
    ),
}


def root_input_contract(model: str) -> SparseProducerRootInputContract:
    """Return the explicit pre-S2 root contract for one supported model."""

    try:
        return _ROOT_INPUT_CONTRACTS[model]
    except KeyError as error:
        raise ValueError(f"unsupported dependency-aware sparse producer model: {model}") from error


def validate_sparse_producer_root_inputs(
    model: str, values: Mapping[str, object]
) -> ValidatedSparseProducerRootInputs:
    """Reject any producer root that can conceal a dense S2/S3 execution."""

    contract = root_input_contract(model)
    if not isinstance(values, Mapping):
        raise TypeError("sparse producer root inputs must be a mapping")
    if any(not isinstance(field, str) or not field for field in values):
        raise ValueError("sparse producer root input names must be nonempty strings")
    fields = frozenset(values)
    forbidden = sorted(fields & _FORBIDDEN_ROOT_FIELDS)
    if forbidden:
        raise ValueError(
            "sparse producer root inputs include forbidden dense S2/S3 or target fields: "
            + ", ".join(forbidden)
        )
    missing = sorted(contract.required_fields - fields)
    if missing:
        raise ValueError(
            "sparse producer root inputs are missing required fields: "
            + ", ".join(missing)
        )
    unexpected = sorted(fields - contract.required_fields)
    if unexpected:
        raise ValueError(
            "sparse producer root inputs contain fields not permitted by the model contract: "
            + ", ".join(unexpected)
        )
    return ValidatedSparseProducerRootInputs(
        model=model,
        field_names=tuple(sorted(fields)),
    )


def _mask_sha256(mask: torch.Tensor) -> str:
    value = mask.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _validate_mask(mask: torch.Tensor, *, name: str) -> torch.Tensor:
    if (
        not torch.is_tensor(mask)
        or mask.dtype != torch.bool
        or mask.ndim < 2
        or any(size <= 0 for size in mask.shape)
    ):
        raise ValueError(f"{name} must be a nonempty bool tensor with at least two dimensions")
    return mask.detach().clone()


class DependencyAwarePhaseLedger:
    """Record actual dependency closure work in primary, secondary, Full order.

    A producer provides the masks it truly dispatched.  The ledger verifies
    that only new closure positions were executed and all prior positions were
    recorded as reuse.  It does not infer a closure from a model or turn a
    passed bookkeeping test into a sparse-execution claim.
    """

    def __init__(self, root_inputs: ValidatedSparseProducerRootInputs) -> None:
        if not isinstance(root_inputs, ValidatedSparseProducerRootInputs):
            raise TypeError("dependency-aware ledger requires validated producer roots")
        self._root_inputs = root_inputs
        self._phase_index = -1
        self._active_phase: str | None = None
        self._sealed = False
        self._computed_masks: dict[str, torch.Tensor] = {}
        self._operator_events: list[dict[str, Any]] = []

    def begin_phase(self, phase: str) -> None:
        """Open exactly the next route phase before recording any operator work."""

        if self._sealed:
            raise RuntimeError("dependency-aware phase ledger has been finalized")
        if phase not in _PHASES:
            raise ValueError("phase must be primary, secondary, or full")
        expected_index = self._phase_index + 1
        if expected_index >= len(_PHASES) or _PHASES[expected_index] != phase:
            expected = "none" if expected_index >= len(_PHASES) else _PHASES[expected_index]
            raise ValueError(f"dependency-aware phases must execute primary -> secondary -> full; expected {expected}")
        self._phase_index = expected_index
        self._active_phase = phase

    def record_operator(
        self,
        operator: str,
        *,
        requested_mask: torch.Tensor,
        dependency_closure_mask: torch.Tensor,
        executed_mask: torch.Tensor,
        reused_mask: torch.Tensor,
        dependency_kind: str,
    ) -> dict[str, Any]:
        """Verify and record one operator's actual phase-local dispatch masks."""

        if self._sealed:
            raise RuntimeError("dependency-aware phase ledger has been finalized")
        if self._active_phase is None:
            raise RuntimeError("begin_phase must be called before recording operator work")
        if not isinstance(operator, str) or not operator:
            raise ValueError("operator must be a nonempty string")
        if dependency_kind not in _DEPENDENCY_KINDS:
            allowed = ", ".join(sorted(_DEPENDENCY_KINDS))
            raise ValueError(f"dependency_kind must be one of {allowed}")
        requested = _validate_mask(requested_mask, name="requested_mask")
        closure = _validate_mask(
            dependency_closure_mask, name="dependency_closure_mask"
        )
        executed = _validate_mask(executed_mask, name="executed_mask")
        reused = _validate_mask(reused_mask, name="reused_mask")
        masks = (closure, executed, reused)
        if any(mask.shape != requested.shape or mask.device != requested.device for mask in masks):
            raise ValueError("operator masks must share one shape and device")
        if bool((requested & ~closure).any()):
            raise ValueError("requested_mask must be contained in dependency_closure_mask")

        prior = self._computed_masks.get(operator)
        if prior is None:
            prior = torch.zeros_like(closure)
        elif prior.shape != closure.shape or prior.device != closure.device:
            raise ValueError("operator closure layout changed between route phases")
        expected_executed = closure & ~prior
        expected_reused = closure & prior
        if not torch.equal(executed, expected_executed):
            raise ValueError("executed_mask must equal the new dependency closure")
        if not torch.equal(reused, expected_reused):
            raise ValueError("reused_mask must equal the previously computed closure")

        self._computed_masks[operator] = prior | closure
        event = {
            "phase": self._active_phase,
            "operator": operator,
            "dependency_kind": dependency_kind,
            "requested_mask_sha256": _mask_sha256(requested),
            "dependency_closure_mask_sha256": _mask_sha256(closure),
            "executed_mask_sha256": _mask_sha256(executed),
            "reused_mask_sha256": _mask_sha256(reused),
            "requested_positions": int(requested.sum().item()),
            "dependency_closure_positions": int(closure.sum().item()),
            "executed_positions": int(executed.sum().item()),
            "reused_positions": int(reused.sum().item()),
            "dense_closure": bool(closure.all().item()),
        }
        self._operator_events.append(event)
        return dict(event)

    def finalize(self) -> dict[str, Any]:
        """Seal a non-claim trace after every route phase has been opened."""

        if self._sealed:
            raise RuntimeError("dependency-aware phase ledger has already been finalized")
        if self._phase_index != len(_PHASES) - 1:
            raise ValueError("primary, secondary, and full phases must all be opened")
        events = [dict(event) for event in self._operator_events]
        dense_operators = sorted(
            {
                str(event["operator"])
                for event in events
                if event["dense_closure"]
            }
        )
        self._sealed = True
        return {
            "schema_version": DEPENDENCY_AWARE_EXECUTION_VERSION,
            "model": self._root_inputs.model,
            "root_input_fields": list(self._root_inputs.field_names),
            "source_bound": False,
            "execution_scope": "pre_s2_root_input_and_operator_ledger_contract_only",
            "paper_result_eligible": False,
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "strict_s2_s3_savings_claimed": False,
            "requires_native_source_bound_producer": True,
            "operator_events": events,
            "operator_final_computed_positions": {
                operator: int(mask.sum().item())
                for operator, mask in sorted(self._computed_masks.items())
            },
            "dense_closure_operators": dense_operators,
        }


__all__ = [
    "DEPENDENCY_AWARE_EXECUTION_VERSION",
    "DependencyAwarePhaseLedger",
    "SparseProducerRootInputContract",
    "ValidatedSparseProducerRootInputs",
    "root_input_contract",
    "validate_sparse_producer_root_inputs",
]
