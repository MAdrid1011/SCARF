"""Execution-boundary checks for normal SCARF result records."""

from __future__ import annotations

from typing import Any, Mapping


REPRESENTATIVE_SAES_MATERIALIZATION = "representative"
ASSIGNMENT_CONSENSUS_PSEUDO_DESCRIPTOR_MATERIALIZATION = (
    "assignment-consensus-adapter-pseudo-descriptor-diagnostic"
)
RUN_CLASSES = frozenset(("claim", "functional", "diagnostic"))


def build_execution_contract(
    *, run_class: str, saes_materialization: str
) -> dict[str, str]:
    """Build the provenance contract attached to ordinary result records."""
    contract = {
        "run_class": run_class,
        "saes_materialization": saes_materialization,
    }
    validate_execution_contract(contract)
    return contract


def validate_execution_contract(contract: Any) -> dict[str, str]:
    """Validate the small, versioned-by-presence execution contract."""
    if not isinstance(contract, dict) or set(contract) != {
        "run_class",
        "saes_materialization",
    }:
        raise ValueError("provenance.execution_contract is invalid")
    run_class = contract["run_class"]
    materialization = contract["saes_materialization"]
    if not isinstance(run_class, str) or run_class not in RUN_CLASSES:
        raise ValueError("provenance.execution_contract.run_class is invalid")
    if not isinstance(materialization, str) or not materialization:
        raise ValueError(
            "provenance.execution_contract.saes_materialization is invalid"
        )
    return {
        "run_class": run_class,
        "saes_materialization": materialization,
    }


def execution_contract(record: Mapping[str, Any]) -> dict[str, str] | None:
    """Return a validated contract when a record was produced after this gate."""
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        return None
    contract = provenance.get("execution_contract")
    if contract is None:
        return None
    return validate_execution_contract(contract)


def reject_assignment_consensus_result(
    record: Mapping[str, Any], *, surface: str
) -> dict[str, str] | None:
    """Reject virtual consensus descriptors from normal result surfaces."""
    contract = execution_contract(record)
    if (
        contract is not None
        and contract["saes_materialization"]
        == ASSIGNMENT_CONSENSUS_PSEUDO_DESCRIPTOR_MATERIALIZATION
    ):
        raise ValueError(
            "assignment-consensus pseudo descriptors are a virtual-output "
            f"diagnostic and cannot enter {surface}"
        )
    return contract


def require_paper_execution_contract(
    record: Mapping[str, Any], *, surface: str
) -> dict[str, str] | None:
    """Ensure a tagged paper-eligible record came from the representative path."""
    contract = reject_assignment_consensus_result(record, surface=surface)
    provenance = record.get("provenance")
    dataset = provenance.get("dataset") if isinstance(provenance, dict) else None
    paper_result_eligible = (
        dataset.get("paper_result_eligible") if isinstance(dataset, dict) else None
    )
    if paper_result_eligible is True and contract is not None and (
        contract["run_class"] != "claim"
        or contract["saes_materialization"]
        != REPRESENTATIVE_SAES_MATERIALIZATION
    ):
        raise ValueError(
            "paper-result-eligible execution must use claim run class and "
            "representative SAES materialization"
        )
    return contract
