"""Validated configuration and shared sample selection for reviewer runs.

The selection is materialized once before any workflow starts.  Quality,
mechanism, timing, and performance stages consume that same record, so a
report cannot accidentally combine results produced from different subsets.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from data.verify_prepared_dataset import prepared_scene_order
from scripts.ae_config import CLAIMED_MATRIX, resolve_experiment
from scripts.compile_protocol import canonicalize_index, sha256_file
from scripts.evidence_profiles import resolve_evidence_selection


SCHEMA = "scarf-key-results-run-v1"
SELECTION_SCHEMA = "scarf-run-selection-v1"
PROFILES = {"reviewer", "full"}
DEVICES = {"auto", "orin", "proxy"}
REVIEWER_COUNTS = {"re10k": 512, "acid": 512, "dl3dv": 140}
SHA256 = re.compile(r"[0-9a-f]{64}")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha256_bytes(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _positive_count(value: Any) -> int | str:
    if value == "all":
        return value
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("samples_per_pair must be a positive integer or 'all'")
    return value


def _pairs(value: Any) -> list[str]:
    if value == "all":
        return [f"{model}/{dataset}" for model, dataset in CLAIMED_MATRIX]
    if not isinstance(value, list) or not value:
        raise ValueError("pairs must be 'all' or a non-empty list")
    result: list[str] = []
    known = {f"{model}/{dataset}" for model, dataset in CLAIMED_MATRIX}
    for pair in value:
        if not isinstance(pair, str) or pair not in known:
            raise ValueError(f"unknown model/dataset pair: {pair!r}")
        if pair in result:
            raise ValueError(f"duplicate model/dataset pair: {pair}")
        result.append(pair)
    return result


def load_run_config(path: Path) -> tuple[dict[str, Any], str]:
    """Load and validate the public key-results run configuration."""
    path = Path(path).resolve()
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"run config is not valid JSON: {path}") from exc
    if not isinstance(config, dict):
        raise ValueError("run config must be a JSON object")
    if config.get("schema_version") != SCHEMA:
        raise ValueError(f"run config schema_version must be {SCHEMA}")
    profile = config.get("profile")
    if profile not in PROFILES:
        raise ValueError("profile must be reviewer or full")
    count = _positive_count(config.get("samples_per_pair"))
    pairs = _pairs(config.get("pairs"))
    device = config.get("figure8_device", "auto")
    if device not in DEVICES:
        raise ValueError("figure8_device must be auto, orin, or proxy")
    if not isinstance(config.get("continue_if_gpu_contended", True), bool):
        raise ValueError("continue_if_gpu_contended must be boolean")
    output_root = config.get("output_root", "outputs/ae")
    if not isinstance(output_root, str) or not output_root or Path(output_root).is_absolute():
        raise ValueError("output_root must be a relative path")
    normalized = {
        "schema_version": SCHEMA,
        "profile": profile,
        "samples_per_pair": count,
        "pairs": pairs,
        "figure8_device": device,
        "continue_if_gpu_contended": config.get("continue_if_gpu_contended", True),
        "output_root": output_root,
    }
    return normalized, hashlib.sha256(_canonical(normalized)).hexdigest()


def _select_rows(root: Path, pair: str, config: Mapping[str, Any]) -> dict[str, Any]:
    model, dataset = pair.split("/", 1)
    selection = resolve_evidence_selection(model, dataset, root, str(config["profile"]))
    source_rows, _ = canonicalize_index(
        selection.index_path, selection.source_index_sha256
    )
    dataset_root = resolve_experiment(model, dataset, root).dataset_root
    execution_order = prepared_scene_order(
        dataset_root, {row["scene"] for row in source_rows}
    )
    rows, summary = canonicalize_index(
        selection.index_path,
        selection.source_index_sha256,
        execution_scene_order=execution_order,
    )
    requested = config["samples_per_pair"]
    if requested != "all":
        if requested > len(rows):
            raise ValueError(
                f"{pair} requests {requested} samples but the profile has only {len(rows)}"
            )
        rows = rows[:requested]
    stable_rows = [
        {key: row[key] for key in ("sample_index", "scene", "context_indices", "target_indices")}
        for row in rows
    ]
    subset_hash = _sha256_bytes(stable_rows)
    return {
        "model": model,
        "dataset": dataset,
        "index_path": str(selection.index_path.relative_to(root.resolve())),
        "source_index_sha256": summary["source_index_sha256"],
        "source_entry_count": summary["source_entry_count"],
        "source_sample_count": summary["sample_count"],
        "sample_selection_sha256": summary["sample_selection_sha256"],
        "subset_selection_sha256": subset_hash,
        "sample_count": len(rows),
        "samples": rows,
    }


def build_selection(root: Path, config: Mapping[str, Any], config_sha256: str) -> dict[str, Any]:
    """Build the one selection consumed by every reviewer workflow."""
    root = Path(root).resolve()
    pairs = _pairs(config.get("pairs"))
    pair_records = {pair: _select_rows(root, pair, config) for pair in pairs}
    body = {
        "schema_version": SELECTION_SCHEMA,
        "config_sha256": config_sha256,
        "profile": config["profile"],
        "samples_per_pair": config["samples_per_pair"],
        "pairs": pair_records,
    }
    body["selection_sha256"] = _sha256_bytes(body)
    return body


def write_selection(path: Path, selection: Mapping[str, Any]) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def load_selection(path: Path, *, config_sha256: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"run selection is not valid JSON: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != SELECTION_SCHEMA:
        raise ValueError("run selection schema is invalid")
    expected = value.get("selection_sha256")
    body = dict(value)
    body.pop("selection_sha256", None)
    if expected != _sha256_bytes(body):
        raise ValueError("run selection SHA256 is invalid")
    if config_sha256 is not None and value.get("config_sha256") != config_sha256:
        raise ValueError("run selection does not match run config")
    pairs = value.get("pairs")
    if not isinstance(pairs, dict) or not pairs:
        raise ValueError("run selection has no pairs")
    for pair, record in pairs.items():
        if not isinstance(record, dict) or record.get("sample_count") != len(record.get("samples", [])):
            raise ValueError(f"run selection sample count is invalid for {pair}")
        expected_subset = _sha256_bytes([
            {key: row[key] for key in ("sample_index", "scene", "context_indices", "target_indices")}
            for row in record["samples"]
        ])
        if record.get("subset_selection_sha256") != expected_subset:
            raise ValueError(f"run selection subset hash is invalid for {pair}")
    return value


def pair_selection(
    path: Path,
    pair: str,
    *,
    config_sha256: str | None = None,
    expected_source_index_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    selection = load_selection(path, config_sha256=config_sha256)
    try:
        record = selection["pairs"][pair]
    except KeyError as exc:
        raise ValueError(f"run selection has no pair {pair}") from exc
    source_hash = record.get("source_index_sha256")
    if not isinstance(source_hash, str) or SHA256.fullmatch(source_hash) is None:
        raise ValueError(f"run selection source_index_sha256 is invalid for {pair}")
    if expected_source_index_sha256 is not None:
        if SHA256.fullmatch(expected_source_index_sha256) is None:
            raise ValueError("expected source_index_sha256 is invalid")
        if source_hash != expected_source_index_sha256:
            raise ValueError(
                "shared selection source_index_sha256 does not match "
                "--source-index-sha256"
            )
    return list(record["samples"]), str(selection["selection_sha256"])
