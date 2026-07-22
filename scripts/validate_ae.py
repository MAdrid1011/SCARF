#!/usr/bin/env python3
"""Validate the complete MICRO artifact evidence set against paper targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_result import (
    aggregate_execution_binding_mismatches,
    aggregate_execution_binding_values,
    validate,
)
from scripts.execution_contract import require_paper_execution_contract
from scripts.result_record import claim_ablation_speedups


METRICS = ("psnr_db", "ssim", "lpips")
MECHANISM_RESULT_MODES = ("mechanisms", "ablation", "all", "quality")
VALIDATION_PROFILES = ("author-preflight", "evaluator-final")
HEX64 = set("0123456789abcdef")
DATASET_REPRESENTATIONS = {
    "depthsplat/acid": "acid-native",
    "depthsplat/dl3dv": "depthsplat-native-270x480-v1",
    "depthsplat/re10k": "re10k-native",
    "mvsplat/acid": "acid-native",
    "mvsplat/dl3dv": "re10k-compatible-360x640-v1",
    "mvsplat/re10k": "re10k-native",
    "transplat/acid": "acid-native",
    "transplat/dl3dv": "re10k-compatible-360x640-v1",
    "transplat/re10k": "re10k-native",
}


def validate_result_catalog(
    catalog: dict[str, Any], *, require_key_results: bool
) -> None:
    from scripts.generate_report import validate_figure_catalog

    contract = load(ROOT / "artifact/evaluation_catalog.json")
    validate_figure_catalog(
        catalog, contract, require_key_results=require_key_results
    )


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value.lower()) <= HEX64


def protocol_check(protocol: dict[str, Any], expected_pairs: set[str]) -> dict[str, Any]:
    pairs = protocol.get("pairs")
    complete = protocol.get("status") == "finalized" and isinstance(pairs, dict)
    complete = complete and protocol.get("target_view_policy") == (
        "all_sampler_selected_target_views"
    )
    complete = complete and protocol.get("sample_quality_aggregation") == (
        "arithmetic mean over selected target views"
    )
    if isinstance(pairs, dict):
        complete = complete and expected_pairs <= set(pairs)
        for pair, record in pairs.items():
            if pair not in expected_pairs:
                continue
            complete = complete and isinstance(record, dict)
            if not isinstance(record, dict):
                continue
            count = record.get("sample_count")
            complete = complete and isinstance(count, int) and not isinstance(count, bool) and count > 0
            complete = complete and _sha256(record.get("sample_selection_sha256"))
            complete = complete and _sha256(record.get("dataset_tree_sha256"))
            complete = complete and bool(record.get("context_target_rule"))
            complete = complete and record.get("dataset_representation") == (
                DATASET_REPRESENTATIONS.get(pair)
            )
    return {
        "claim": "evaluation:sample_protocol_finalized",
        "actual": protocol.get("status"),
        "target": "finalized",
        "pass": bool(complete),
    }


def find_pair_result(output: Path, pair: str, modes: tuple[str, ...]) -> Path:
    directory = pair.replace("/", "_")
    candidates = [output / mode / directory / "results.json" for mode in modes]
    matches = [path for path in candidates if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"missing software result for {pair}")
    return matches[0]


def find_functional_reference(
    output: Path,
) -> tuple[str, str, Path] | None:
    quick = output / "quick/mvsplat_re10k/results.json"
    if quick.is_file():
        return "quick", "mvsplat/re10k", quick
    for pair in sorted(DATASET_REPRESENTATIONS):
        path = output / "quality" / pair.replace("/", "_") / "results.json"
        if path.is_file():
            return "quality", pair, path
    return None


def relative_check(claim: str, actual: float, target: float, tolerance: float) -> dict[str, Any]:
    relative_error = abs(actual - target) / target
    return {
        "claim": claim,
        "actual": actual,
        "target": target,
        "tolerance": tolerance,
        "relative_error": relative_error,
        "pass": relative_error <= tolerance,
    }


def require_aggregate(result: dict[str, Any], pair: str) -> None:
    require_paper_execution_contract(result, surface=f"AE validation for {pair}")
    evaluation = result.get("provenance", {}).get("evaluation", {})
    if evaluation.get("kind") != "dataset_aggregate":
        raise ValueError(f"{pair} is not a dataset aggregate")
    count = evaluation.get("sample_count")
    if not isinstance(count, int) or count <= 0:
        raise ValueError(f"{pair} has an invalid aggregate sample count")


def clean_source_check(
    result: dict[str, Any], pair: str, workflow: str
) -> dict[str, Any]:
    provenance = result.get("provenance", {})
    commit = provenance.get("git_commit")
    clean = provenance.get("git_dirty") is False
    valid_commit = (
        isinstance(commit, str)
        and len(commit) == 40
        and set(commit.lower()) <= HEX64
    )
    return {
        "claim": f"provenance:{pair}:{workflow}:clean_commit",
        "actual": {"git_commit": commit, "git_dirty": provenance.get("git_dirty")},
        "target": "40-character git commit and git_dirty=false",
        "pass": clean and valid_commit,
    }


def source_binding_check(
    result: dict[str, Any],
    reference: dict[str, Any],
    component: str,
    reference_name: str = "quick",
) -> dict[str, Any]:
    """Require functional evidence to share a verified source identity."""
    provenance = result.get("provenance", {})
    actual = {
        "git_commit": provenance.get("git_commit"),
        "git_dirty": provenance.get("git_dirty"),
        "source_identity": provenance.get("source_identity"),
        "submodules": provenance.get("submodules"),
    }
    target = {
        "git_commit": reference.get("git_commit"),
        "git_dirty": False,
        "source_identity": reference.get("source_identity"),
        "submodules": reference.get("submodules"),
    }
    return {
        "claim": f"provenance:{component}:matches_{reference_name}_source",
        "actual": actual,
        "target": target,
        "pass": actual == target,
    }


def quality_mechanism_binding_check(
    quality: dict[str, Any], mechanism: dict[str, Any], pair: str
) -> dict[str, Any]:
    """Require Table 1 and mechanism aggregates to share one execution set."""
    mismatches = aggregate_execution_binding_mismatches(quality, mechanism)
    return {
        "claim": f"binding:{pair}:quality_mechanism_execution",
        "actual": {
            "quality": aggregate_execution_binding_values(quality),
            "mechanism": aggregate_execution_binding_values(mechanism),
        },
        "target": "identical mechanism config, checkpoint, selection, and trace set",
        "mismatches": mismatches,
        "pass": not mismatches,
    }


def quality_mechanism_presence_check(
    pair: str, *, quality_claimed: bool, mechanism_claimed: bool
) -> dict[str, Any]:
    """Require both claim surfaces before validating their aggregate bindings."""
    return {
        "claim": f"binding:{pair}:quality_mechanism_presence",
        "actual": {
            "quality_claimed": quality_claimed,
            "mechanism_claimed": mechanism_claimed,
        },
        "target": "matching claim-quality and mechanism aggregate records",
        "pass": quality_claimed and mechanism_claimed,
    }


def environment_provenance_check(
    result: dict[str, Any], pair: str, output: Path
) -> dict[str, Any]:
    declared = result.get("provenance", {}).get("environment", {})
    profile = declared.get("profile")
    path = output / "environments" / f"{profile}.json"
    try:
        record = load(path)
        details = record["details"]
        canonical_record = {
            "profile": profile,
            "python": details["python"],
            "implementation": details["implementation"],
            "torch": details["torch"],
            "torchvision": details["torchvision"],
            "torch_cuda": details["torch_cuda"],
            "lock_sha256": details["lock_sha256"],
        }
        canonical = json.dumps(
            canonical_record, sort_keys=True, separators=(",", ":")
        ).encode()
        digest = hashlib.sha256(canonical).hexdigest()
        passed = (
            record.get("profile") == profile
            and record.get("contract")
            and digest == declared.get("digest_sha256")
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        digest = None
        passed = False
    return {
        "claim": f"provenance:{pair}:environment_digest",
        "actual": digest,
        "target": declared.get("digest_sha256"),
        "pass": bool(passed),
    }


def hardware_claim_checks(
    output: Path, claim_status: dict[str, Any]
) -> list[dict[str, Any]]:
    checks = []
    if claim_status["physical_asap7"] == "CLAIMED":
        physical = load(output / "physical/asap7/ppa.json")
        checks.append(
            {
                "claim": "hardware:asap7_routed_ppa",
                "pass": physical.get("physical_valid") is True,
            }
        )
    else:
        checks.append(
            {
                "claim": "hardware:asap7_not_claimed",
                "actual": claim_status["physical_asap7"],
                "target": "NOT_CLAIMED_RESOURCE_LIMIT",
                "pass": claim_status["physical_asap7"]
                == "NOT_CLAIMED_RESOURCE_LIMIT",
            }
        )
    if claim_status["deepscale"] == "CLAIMED":
        scaled = load(output / "physical/asap7/ppa_28nm_estimated.json")
        checks.append(
            {
                "claim": "hardware:deepscale_7_to_28",
                "pass": claim_status["physical_asap7"] == "CLAIMED"
                and scaled.get("validation", {}).get("raw_preserved") is True
                and scaled.get("validation", {}).get("foundry_measurement") is False
                and scaled.get("scaling", {}).get("source_node_nm") == 7
                and scaled.get("scaling", {}).get("target_node_nm") == 28,
            }
        )
    else:
        checks.append(
            {
                "claim": "hardware:deepscale_not_claimed",
                "actual": claim_status["deepscale"],
                "target": "NOT_CLAIMED_NO_PHYSICAL_INPUT",
                "pass": claim_status["deepscale"]
                == "NOT_CLAIMED_NO_PHYSICAL_INPUT",
            }
        )
    return checks


def physical_proxy_active(catalog: dict[str, Any]) -> bool:
    """Require DRAM proxy evidence unless the catalog explicitly pauses it."""
    records = {
        record.get("id"): record
        for record in catalog.get("results", [])
        if isinstance(record, dict)
    }
    return any(
        records.get(result_id, {}).get("current_state") != "PAUSED_BY_SCOPE"
        for result_id in ("figure9", "table4")
    )


def orin_evidence_reason(
    result: dict[str, Any], expected_selection_sha256: str, base_dir: Path | None = None
) -> str | None:
    """Return why an aggregate cannot support an independent Orin result."""
    from hardware.orin.run import sha256_file, validate_measurement_record

    if result.get("schema_version") != "2.1":
        return "result does not use the trace-bound schema"
    if result.get("evidence_class") != "independent_measurement":
        return "result is not independent measurement evidence"
    if result.get("performance", {}).get("baseline_source") != "orin_nx_cuda_events":
        return "result baseline is not sourced from Orin CUDA events"
    try:
        validate(result)
    except (KeyError, TypeError, ValueError) as exc:
        return f"result fails validation: {exc}"

    provenance = result.get("provenance")
    if not isinstance(provenance, dict):
        return "result provenance is missing"
    if provenance.get("execution_contract") != {
        "run_class": "claim",
        "saes_materialization": "representative",
    }:
        return "result was not executed through the representative claim path"
    evaluation = provenance.get("evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("kind") != "dataset_aggregate":
        return "result is not a dataset aggregate"
    if evaluation.get("sample_selection_sha256") != expected_selection_sha256:
        return "result sample selection does not match the frozen Orin protocol"
    trace_set = evaluation.get("execution_trace_set")
    if not isinstance(trace_set, list) or not trace_set:
        return "result has no aggregate execution trace set"
    trace_by_index = {
        entry.get("sample_index"): entry
        for entry in trace_set
        if isinstance(entry, dict)
    }
    sample_results = evaluation.get("sample_results")
    if (
        not isinstance(sample_results, list)
        or not sample_results
        or len(sample_results) != len(trace_by_index)
    ):
        return "result has no complete sample measurement evidence"
    dataset = provenance.get("dataset")
    checkpoint = provenance.get("checkpoint")
    if not isinstance(dataset, dict) or not isinstance(checkpoint, dict):
        return "result dataset or checkpoint provenance is missing"
    if base_dir is None:
        return "Orin measurement base directory is unavailable"
    base_dir = base_dir.resolve()
    for item in sample_results:
        if not isinstance(item, dict):
            return "result has an invalid sample measurement entry"
        sample_index = item.get("sample_index")
        trace = trace_by_index.get(sample_index)
        if not isinstance(sample_index, int) or isinstance(sample_index, bool) or not isinstance(
            trace, dict
        ):
            return "result sample measurement does not match its execution trace"
        evidence = item.get("orin_measurement")
        if not isinstance(evidence, dict):
            return "result sample is missing an Orin measurement"
        relative_path = Path(str(evidence.get("path", "")))
        if relative_path.is_absolute():
            return "Orin measurement path is not portable"
        path = (base_dir / relative_path).resolve()
        try:
            path.relative_to(base_dir)
        except ValueError:
            return "Orin measurement path escapes its pair evidence directory"
        try:
            if not path.is_file() or sha256_file(path) != evidence.get("sha256"):
                return "Orin measurement hash does not match aggregate evidence"
            measurement = load(path)
            validate_measurement_record(
                measurement,
                expected_selection_sha256=expected_selection_sha256,
                measurement_path=path,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return f"Orin measurement is invalid: {exc}"
        selection = measurement.get("selection", {})
        if (
            selection.get("sample_index") != sample_index
            or selection.get("execution_index") != trace.get("execution_index")
            or selection.get("dataset_tree_sha256") != dataset.get("tree_sha256")
            or selection.get("checkpoint_sha256") != checkpoint.get("sha256")
            or trace.get("orin_measurement_sha256") != evidence.get("sha256")
        ):
            return "Orin measurement provenance does not match its aggregate trace"
    return None


def orin_evidence_complete(
    result: dict[str, Any], expected_selection_sha256: str, base_dir: Path | None = None
) -> bool:
    return orin_evidence_reason(result, expected_selection_sha256, base_dir) is None


def figure8_catalog_evidence_check(
    output: Path,
    catalog: dict[str, Any],
    protocol: dict[str, Any],
    claim_state: Any,
) -> dict[str, Any]:
    """Prevent a catalog row from upgrading a pending Orin claim to PASS."""
    rows = catalog.get("results")
    figure8 = next(
        (
            row
            for row in rows
            if isinstance(row, dict) and row.get("id") == "figure8"
        ),
        None,
    ) if isinstance(rows, list) else None
    status = figure8.get("status") if isinstance(figure8, dict) else None
    target = "CLAIMED plus complete independent Orin measurement evidence"
    if status != "PASS":
        return {
            "claim": "figure8:catalog_independent_orin_evidence",
            "actual": status,
            "target": target,
            "pass": claim_state != "CLAIMED",
        }
    if claim_state != "CLAIMED":
        return {
            "claim": "figure8:catalog_independent_orin_evidence",
            "actual": claim_state,
            "target": target,
            "pass": False,
        }

    pairs = protocol.get("pairs")
    failures: list[str] = []
    if not isinstance(pairs, dict) or set(pairs) != set(DATASET_REPRESENTATIONS):
        failures.append("frozen evaluation protocol does not define the nine pairs")
    else:
        for pair in sorted(pairs):
            try:
                path = find_pair_result(output, pair, ("speedup", "orin"))
                result = load(path)
                validate(result)
                require_aggregate(result, pair)
                reason = orin_evidence_reason(
                    result,
                    pairs[pair]["sample_selection_sha256"],
                    path.parent,
                )
                if reason is not None:
                    failures.append(f"{pair}: {reason}")
            except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
                failures.append(f"{pair}: {exc}")
    return {
        "claim": "figure8:catalog_independent_orin_evidence",
        "actual": failures or "complete",
        "target": target,
        "pass": not failures,
    }


def figure8_pending_state_check(state: Any) -> dict[str, Any]:
    """Accept the declared pending-Orin intent without treating it as a pass."""
    return {
        "claim": "figure8:awaiting_independent_orin_evaluation",
        "actual": state,
        "target": "CLAIMED_AWAITING_INDEPENDENT_ORIN_EVALUATION",
        "pass": state == "CLAIMED_AWAITING_INDEPENDENT_ORIN_EVALUATION",
    }


def sensitivity_claim_checks(
    sensitivity: dict[str, Any], expected: dict[str, Any]
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[float, list[dict[str, Any]]]] = {}
    for run in sensitivity.get("runs", []):
        grouped.setdefault(run.get("study"), {}).setdefault(
            float(run.get("value")), []
        ).append(run.get("metrics", {}))

    summary: dict[str, dict[float, dict[str, float]]] = {}
    for study, grid in expected["sensitivity_grids"].items():
        summary[study] = {}
        for value in grid["values"]:
            metrics = grouped.get(study, {}).get(float(value), [])
            if len(metrics) != 9:
                continue
            speedups = [float(item["performance"]["speedup"]) for item in metrics]
            degradation = max(
                float(item["quality"]["change"]["psnr_degradation_pct"])
                for item in metrics
            )
            summary[study][float(value)] = {
                "throughput": math.exp(
                    sum(math.log(item) for item in speedups) / len(speedups)
                ),
                "degradation": degradation,
            }

    claims = expected["sensitivity_claims"]
    ratio_tolerance = float(claims["throughput_ratio_tolerance"])
    quality_tolerance = float(claims["quality_absolute_tolerance_pct"])
    checks = []

    def peak_check(study: str, target: float, claim: str) -> None:
        rows = summary.get(study, {})
        actual = max(rows, key=lambda value: rows[value]["throughput"]) if rows else None
        checks.append({"claim": claim, "actual": actual, "target": target, "pass": actual == target})

    peak_check(
        "fsdr_cache_size",
        float(claims["cache_throughput_peak"]),
        "figure13:cache_throughput_peak",
    )
    peak_check(
        "saes_feature_variance",
        float(claims["feature_variance_throughput_peak"]),
        "figure15:feature_variance_throughput_peak",
    )

    hamming = summary.get("fsdr_hamming_threshold", {})
    h_claim = claims["hamming"]
    for label, numerator, denominator in (
        ("3_over_1", 3.0, 1.0),
        ("5_over_3", 5.0, 3.0),
        ("5_over_4", 5.0, 4.0),
    ):
        actual = (
            hamming[numerator]["throughput"] / hamming[denominator]["throughput"]
            if numerator in hamming and denominator in hamming
            else None
        )
        target = float(h_claim[f"throughput_ratio_{label}"])
        checks.append(
            {
                "claim": f"figure14:throughput_ratio_{label}",
                "actual": actual,
                "target": target,
                "tolerance": ratio_tolerance,
                "pass": actual is not None and abs(actual - target) <= ratio_tolerance,
            }
        )
    degradation_three = hamming.get(3.0, {}).get("degradation")
    checks.append(
        {
            "claim": "figure14:degradation_at_3",
            "actual": degradation_three,
            "target": f"<= {h_claim['max_degradation_at_3_pct']}",
            "pass": degradation_three is not None
            and degradation_three <= float(h_claim["max_degradation_at_3_pct"]),
        }
    )
    degradation_five = hamming.get(5.0, {}).get("degradation")
    checks.append(
        {
            "claim": "figure14:degradation_at_5",
            "actual": degradation_five,
            "target": float(h_claim["degradation_at_5_pct"]),
            "tolerance": quality_tolerance,
            "pass": degradation_five is not None
            and abs(degradation_five - float(h_claim["degradation_at_5_pct"]))
            <= quality_tolerance,
        }
    )
    ordered_hamming = [hamming.get(float(value), {}).get("degradation") for value in range(1, 6)]
    checks.append(
        {
            "claim": "figure14:degradation_monotonic",
            "actual": ordered_hamming,
            "target": "nondecreasing",
            "pass": all(value is not None for value in ordered_hamming)
            and all(a <= b for a, b in zip(ordered_hamming, ordered_hamming[1:])),
        }
    )

    depth = summary.get("saes_depth_variance", {})
    depth_values = sorted(depth)
    depth_throughput = [depth[value]["throughput"] for value in depth_values]
    checks.append(
        {
            "claim": "figure15:depth_throughput_monotonic",
            "actual": depth_throughput,
            "target": "nondecreasing",
            "pass": len(depth_throughput) == 5
            and all(a <= b for a, b in zip(depth_throughput, depth_throughput[1:])),
        }
    )
    checks.append(
        {
            "claim": "figure15:depth_quality_breakpoint",
            "actual": {value: depth[value]["degradation"] for value in depth_values},
            "target": "degradation at 0.2 exceeds degradation at 0.1",
            "pass": 0.1 in depth
            and 0.2 in depth
            and depth[0.2]["degradation"] > depth[0.1]["degradation"],
        }
    )

    tile = summary.get("saes_tile_size", {})
    tile_default = tile.get(4.0, {}).get("throughput")
    for value_text, target_value in claims["tile_throughput_ratio_to_4"].items():
        value = float(value_text)
        actual = (
            tile[value]["throughput"] / tile_default
            if value in tile and tile_default
            else None
        )
        checks.append(
            {
                "claim": f"figure16:throughput_ratio_{value_text}_to_4",
                "actual": actual,
                "target": float(target_value),
                "tolerance": ratio_tolerance,
                "pass": actual is not None
                and abs(actual - float(target_value)) <= ratio_tolerance,
            }
        )
    return checks


def validate_complete(
    output: Path,
    expected_path: Path,
    *,
    require_key_results: bool = False,
    allow_missing_quick: bool = False,
    validation_profile: str = "author-preflight",
) -> dict[str, Any]:
    if validation_profile not in VALIDATION_PROFILES:
        raise ValueError(f"unknown validation profile: {validation_profile}")
    require_key_results = bool(
        require_key_results or validation_profile == "evaluator-final"
    )
    expected = load(expected_path)
    tolerances = expected["tolerances"]
    protocol = load(ROOT / "artifact/evaluation_protocol.json")
    evaluation_catalog = load(ROOT / "artifact/evaluation_catalog.json")
    claim_status = load(ROOT / "artifact/claim_status.json")
    quality_pairs = {
        pair
        for pair, state in claim_status["software_pairs"].items()
        if state == "CLAIMED"
    }
    mechanism_pairs = {
        pair
        for pair, state in claim_status["mechanism_pairs"].items()
        if state == "CLAIMED"
    }
    checks = [protocol_check(protocol, quality_pairs)]
    for pair in sorted(quality_pairs | mechanism_pairs):
        checks.append(
            quality_mechanism_presence_check(
                pair,
                quality_claimed=pair in quality_pairs,
                mechanism_claimed=pair in mechanism_pairs,
            )
        )
    reference = find_functional_reference(output)
    quick_path = output / "quick/mvsplat_re10k/results.json"
    if reference is None:
        if not allow_missing_quick:
            load(quick_path)
        raise FileNotFoundError("missing quick or quality functional reference")
    reference_workflow, reference_pair, reference_path = reference
    functional_reference = load(reference_path)
    validate(functional_reference)
    require_aggregate(functional_reference, reference_pair)
    checks.extend(
        (
            clean_source_check(
                functional_reference, reference_pair, reference_workflow
            ),
            environment_provenance_check(
                functional_reference, reference_pair, output
            ),
        )
    )
    if reference_workflow == "quick":
        checks.append(
            {
                "claim": "functional:quick_fixture_not_paper_evidence",
                "pass": functional_reference["provenance"]["dataset"].get(
                    "functional_fixture"
                )
                is True
                and functional_reference["provenance"]["dataset"].get(
                    "paper_result_eligible"
                )
                is False,
            }
        )
    elif not allow_missing_quick:
        load(quick_path)
    else:
        checks.append(
            {
                "claim": "functional:quick_fixture_not_run",
                "status": "SKIPPED",
                "target": "quick is not part of this workflow",
                "pass": True,
            }
        )
    if not quality_pairs:
        software_states = {
            pair: state
            for pair, state in claim_status["software_pairs"].items()
            if not pair.endswith("/dl3dv")
        }
        checks.append(
            {
                "claim": "table1:not_claimed_sparse_quality_mismatch",
                "actual": software_states,
                "pass": bool(software_states)
                and all(
                    str(state).startswith("NOT_CLAIMED")
                    for state in software_states.values()
                ),
            }
        )
    quality_results = {}
    for pair in (item for item in expected["table1"] if item in quality_pairs):
        targets = expected["table1"][pair]
        path = find_pair_result(output, pair, ("quality", "all", "ablation"))
        result = load(path)
        validate(result)
        require_aggregate(result, pair)
        checks.append(clean_source_check(result, pair, "quality"))
        checks.append(environment_provenance_check(result, pair, output))
        checks.append(
            {
                "claim": f"protocol:{pair}:paper_dataset_eligible",
                "pass": result["provenance"]["dataset"].get(
                    "functional_fixture"
                )
                is False
                and result["provenance"]["dataset"].get(
                    "paper_result_eligible"
                )
                is True,
            }
        )
        checks.append(
            {
                "claim": f"protocol:{pair}:quality_selection",
                "actual": result["provenance"]["evaluation"].get("sample_selection_sha256"),
                "target": protocol["pairs"][pair].get("sample_selection_sha256"),
                "pass": result["provenance"]["evaluation"].get("sample_selection_sha256")
                == protocol["pairs"][pair].get("sample_selection_sha256"),
            }
        )
        checks.append(
            {
                "claim": f"protocol:{pair}:dataset_tree",
                "actual": result["provenance"]["dataset"].get("tree_sha256"),
                "target": protocol["pairs"][pair].get("dataset_tree_sha256"),
                "pass": result["provenance"]["dataset"].get("tree_sha256")
                == protocol["pairs"][pair].get("dataset_tree_sha256"),
            }
        )
        checks.append(
            {
                "claim": f"protocol:{pair}:dataset_representation",
                "actual": result["provenance"]["dataset"].get("representation"),
                "target": protocol["pairs"][pair].get("dataset_representation"),
                "pass": result["provenance"]["dataset"].get("representation")
                == protocol["pairs"][pair].get("dataset_representation"),
            }
        )
        quality_results[pair] = result
        for variant in ("baseline", "scarf"):
            for metric, target in zip(METRICS, targets[variant]):
                actual = result["quality"][variant][metric]
                tolerance = tolerances[metric]
                passed = abs(actual - target) <= tolerance
                checks.append(
                    {
                        "claim": f"table1:{pair}:{variant}:{metric}",
                        "actual": actual,
                        "target": target,
                        "tolerance": tolerance,
                        "pass": passed,
                    }
                )
    if claim_status["figure8"] == "CLAIMED":
        speedups = []
        for pair in (item for item in expected["table1"] if item in quality_pairs):
            path = find_pair_result(output, pair, ("speedup", "orin"))
            result = load(path)
            validate(result)
            require_aggregate(result, pair)
            checks.append(clean_source_check(result, pair, "speedup"))
            checks.append(
                {
                    "claim": f"protocol:{pair}:speedup_paper_dataset_eligible",
                    "pass": result["provenance"]["dataset"].get(
                        "functional_fixture"
                    )
                    is False
                    and result["provenance"]["dataset"].get(
                        "paper_result_eligible"
                    )
                    is True,
                }
            )
            checks.append(
                {
                    "claim": f"protocol:{pair}:speedup_selection",
                    "pass": result["provenance"]["evaluation"].get("sample_selection_sha256")
                    == protocol["pairs"][pair].get("sample_selection_sha256"),
                }
            )
            checks.append(
                {
                    "claim": f"figure8:{pair}:orin_evidence",
                    "pass": orin_evidence_complete(
                        result,
                        protocol["pairs"][pair]["sample_selection_sha256"],
                        path.parent,
                    ),
                }
            )
            speedups.append(float(result["performance"]["speedup"]))
        geometric_speedup = math.exp(
            sum(math.log(value) for value in speedups) / len(speedups)
        )
        checks.append(relative_check(
            "figure8:geometric_mean_speedup",
            geometric_speedup,
            expected["figure8"]["geometric_mean_speedup"],
            tolerances["speedup_relative"],
        ))
    elif claim_status["figure8"] == "CLAIMED_AWAITING_INDEPENDENT_ORIN_EVALUATION":
        checks.append(figure8_pending_state_check(claim_status["figure8"]))
    else:
        checks.append(
            {
                "claim": "figure8:not_claimed",
                "actual": claim_status["figure8"],
                "target": "NOT_CLAIMED with no fabricated Orin result",
                "pass": str(claim_status["figure8"]).startswith("NOT_CLAIMED"),
            }
        )

    ablation_speedups = {"fsdr": [], "saes": [], "combined": []}
    mechanism_results = {}
    mechanism_paths = {
        "guided_rate": ("fsdr_saes", "fsdr", "guided_rate"),
        "top1_coverage": ("fsdr_saes", "fsdr", "in_window_rate"),
        "level0_rate": ("fsdr_saes", "saes", "level0_ratio"),
        "level1_rate": ("fsdr_saes", "saes", "level1_ratio"),
        "low_variance_agreement": ("fsdr_saes", "preservation", "saes_low_var_agree"),
        "gaussians_saved": ("fsdr_saes", "saes", "modification_ratio"),
    }
    for pair in (item for item in expected["mechanisms"] if item in mechanism_pairs):
        targets = expected["mechanisms"][pair]
        result = load(find_pair_result(output, pair, MECHANISM_RESULT_MODES))
        validate(result)
        require_aggregate(result, pair)
        checks.append(clean_source_check(result, pair, "ablation"))
        checks.append(
            {
                "claim": f"protocol:{pair}:ablation_paper_dataset_eligible",
                "pass": result["provenance"]["dataset"].get(
                    "functional_fixture"
                )
                is False
                and result["provenance"]["dataset"].get(
                    "paper_result_eligible"
                )
                is True,
            }
        )
        checks.append(
            {
                "claim": f"protocol:{pair}:ablation_selection",
                "pass": result["provenance"]["evaluation"].get("sample_selection_sha256")
                == protocol["pairs"][pair].get("sample_selection_sha256"),
            }
        )
        for label, speedup in claim_ablation_speedups(result).items():
            ablation_speedups[label].append(speedup)
        for metric, target in targets.items():
            actual: Any = result
            for key in mechanism_paths[metric]:
                actual = actual[key]
            actual = float(actual)
            tolerance = expected["mechanism_tolerance_absolute"]
            checks.append(
                {
                    "claim": f"figure11:supporting:{pair}:{metric}",
                    "actual": actual,
                    "target": target,
                    "tolerance": tolerance,
                    "pass": abs(actual - target) <= tolerance,
                }
            )
        mechanism_results[pair] = result
    for pair in sorted(set(quality_results) & set(mechanism_results)):
        checks.append(
            quality_mechanism_binding_check(
                quality_results[pair], mechanism_results[pair], pair
            )
        )
    if claim_status["figure11"] == "CLAIMED":
        for label, values in ablation_speedups.items():
            geometric = math.exp(sum(math.log(value) for value in values) / len(values))
            checks.append(relative_check(
                f"figure11:{label}",
                geometric,
                float(expected["ablation"][label]),
                tolerances["speedup_relative"],
            ))
    else:
        checks.append(
            {
                "claim": "figure11:not_claimed",
                "actual": claim_status["figure11"],
                "pass": str(claim_status["figure11"]).startswith("NOT_CLAIMED"),
            }
        )

    rtl = load(output / "rtl/results.json")
    checks.append({"claim": "rtl:functional", "pass": rtl.get("status") == "PASS"})
    checks.append(
        source_binding_check(
            rtl,
            functional_reference["provenance"],
            "rtl",
            reference_workflow,
        )
    )

    if claim_status["sensitivity"] == "CLAIMED":
        sensitivity = load(output / "sensitivity/results.json")
        sens_validation = sensitivity.get("validation", {})
        unique_sensitivity_runs = {
            (item.get("study"), item.get("value"), item.get("model"), item.get("dataset"))
            for item in sensitivity.get("runs", [])
        }
        checks.append(
            {
                "claim": "figures13-16:sensitivity",
                "actual": sens_validation.get("completed_runs"),
                "target": expected["sensitivity_expected_runs"],
                "pass": sens_validation.get("complete") is True
                and sens_validation.get("completed_runs") == expected["sensitivity_expected_runs"]
                and len(unique_sensitivity_runs) == expected["sensitivity_expected_runs"]
                and sensitivity.get("grids") == expected["sensitivity_grids"],
            }
        )
        checks.extend(sensitivity_claim_checks(sensitivity, expected))
        checks.append(
            {
                "claim": "figures13-16:sample_protocol",
                "pass": all(
                    item.get("metrics", {}).get("evaluation", {}).get("sample_selection_sha256")
                    == protocol["pairs"].get(f"{item.get('model')}/{item.get('dataset')}", {}).get(
                        "sample_selection_sha256"
                    )
                    for item in sensitivity.get("runs", [])
                ),
            }
        )
    else:
        checks.append(
            {
                "claim": "figures13-16:not_claimed",
                "actual": claim_status["sensitivity"],
                "pass": str(claim_status["sensitivity"]).startswith("NOT_CLAIMED"),
            }
        )

    if physical_proxy_active(evaluation_catalog):
        dram = load(output / "dram/results.json")
        checks.append(
            {
                "claim": "dram:public_proxy_functional",
                "pass": dram.get("status") == "PASS"
                and dram.get("paper_lpddr4x_reproduced") is False
                and float(dram.get("metrics", {}).get("trace_requests", 0)) > 0
                and float(dram.get("metrics", {}).get("ramulator_memory_cycles", 0)) > 0
                and float(dram.get("metrics", {}).get("drampower_offchip_energy_j", 0)) > 0,
            }
        )
        checks.append(
            source_binding_check(
                dram,
                functional_reference["provenance"],
                "dram",
                reference_workflow,
            )
        )
    else:
        checks.append(
            {
                "claim": "dram:paused_physical_proxy",
                "status": "SKIPPED",
                "pass": True,
            }
        )

    checks.extend(hardware_claim_checks(output, claim_status))
    report = output / "reports/reproduction_report.md"
    catalog = output / "reports/figure_catalog.json"
    catalog_record = load(catalog)
    checks.append(
        figure8_catalog_evidence_check(
            output,
            catalog_record,
            protocol,
            claim_status["figure8"],
        )
    )
    validate_result_catalog(
        catalog_record, require_key_results=require_key_results
    )
    checks.append(
        {
            "claim": "report:paper_comparison_exports",
            "pass": report.is_file()
            and report.stat().st_size > 0
            and catalog.is_file()
            and catalog.stat().st_size > 0,
        }
    )
    passed = all(check["pass"] for check in checks)
    return {
        "schema_version": "1.0",
        "status": "PASS" if passed else "FAIL",
        "validation_profile": validation_profile,
        "require_key_results": bool(require_key_results),
        "checks": checks,
        "summary": {
            "passed": sum(bool(item["pass"]) for item in checks),
            "total": len(checks),
        },
    }


def validate_current_release(
    output: Path,
    *,
    require_key_results: bool = False,
    allow_missing_quick: bool = False,
    validation_profile: str = "author-preflight",
) -> dict[str, Any]:
    """Validate an output root against the validator-owned paper contract."""

    return validate_complete(
        output,
        ROOT / "artifact" / "expected_results.json",
        require_key_results=require_key_results,
        allow_missing_quick=allow_missing_quick,
        validation_profile=validation_profile,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--expected",
        type=Path,
        default=ROOT / "artifact/expected_results.json",
    )
    parser.add_argument(
        "--require-key-results",
        action="store_true",
        help="Reject any mandatory Evaluation result that is not PASS",
    )
    parser.add_argument(
        "--profile",
        choices=VALIDATION_PROFILES,
        default="author-preflight",
        help="Select author preflight or evaluator-final validation strictness",
    )
    parser.add_argument(
        "--allow-missing-quick",
        action="store_true",
        help="Use quality evidence as the functional source reference when quick was not run",
    )
    args = parser.parse_args()
    try:
        record = validate_complete(
            args.input.resolve(),
            args.expected.resolve(),
            require_key_results=(
                args.require_key_results or args.profile == "evaluator-final"
            ),
            allow_missing_quick=args.allow_missing_quick,
            validation_profile=args.profile,
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    output = args.input / "validation.json"
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{record['status']}: {record['summary']['passed']}/{record['summary']['total']} checks")
    return 0 if record["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
