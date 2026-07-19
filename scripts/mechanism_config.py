"""Validate and hash the one global SCARF mechanism configuration."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "artifact/mechanism_config.json"
CALIBRATION_CONTRACT = ROOT / "artifact/CALIBRATION.md"
SHA256 = re.compile(r"[0-9a-f]{64}")
FIXED = {
    "fsdr_cache_size": 32,
    "fsdr_contraction_ratio": 4,
    "fsdr_hamming_threshold": 3,
    "fsdr_guidance_policy": "paper-hamming-local-validity",
    "saes_depth_threshold": 0.1,
    "saes_feature_threshold": 0.2,
    "saes_l1_depth_reference": "primary-routing-probes-v1",
    "saes_moment_geometry": "c2w-probe-depth-ray-v1",
    "saes_tile_size": 4,
}
CALIBRATED_SPLIT_PROTOCOL = "dl3dv_train_holdout_v1"
CALIBRATED_SPLIT_HASHES = (
    "selection_sha256",
    "scene_set_sha256",
    "pair_bindings_sha256",
    "trace_set_sha256",
    "candidate_set_sha256",
)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA256 digest")
    return value


def _validated_split_provenance(
    calibration: dict[str, Any], selected: dict[str, Any]
) -> dict[str, Any]:
    """Validate the train/holdout evidence bound into a frozen config."""
    if calibration.get("protocol") != CALIBRATED_SPLIT_PROTOCOL:
        raise ValueError("calibrated mechanism config has no DL3DV train/holdout protocol")
    if calibration.get("train_holdout_scene_disjoint") is not True:
        raise ValueError("calibrated mechanism config train/holdout split is not disjoint")
    split_records: dict[str, dict[str, Any]] = {}
    for split in ("train", "holdout"):
        record = calibration.get(split)
        if not isinstance(record, dict):
            raise ValueError(f"calibrated mechanism config has no {split} evidence")
        normalized = {
            field: _digest(record.get(field), f"calibration.{split}.{field}")
            for field in CALIBRATED_SPLIT_HASHES
        }
        if split == "train":
            normalized["selected_candidate_sha256"] = _digest(
                record.get("selected_candidate_sha256"),
                "calibration.train.selected_candidate_sha256",
            )
        else:
            normalized["validated_parameters_sha256"] = _digest(
                record.get("validated_parameters_sha256"),
                "calibration.holdout.validated_parameters_sha256",
            )
            normalized["validated_candidate_sha256"] = _digest(
                record.get("validated_candidate_sha256"),
                "calibration.holdout.validated_candidate_sha256",
            )
        split_records[split] = normalized
    if split_records["train"]["selection_sha256"] == split_records["holdout"]["selection_sha256"]:
        raise ValueError("calibrated mechanism config reuses one train/holdout selection")
    from scripts.calibration_contract import parameters_sha256

    if split_records["holdout"]["validated_parameters_sha256"] != parameters_sha256(
        selected
    ):
        raise ValueError("calibrated mechanism config holdout tuple does not match selected parameters")
    return split_records


def load_mechanism_config(
    path: Path = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "1.0":
        raise ValueError("mechanism config schema must be 1.0")
    if config.get("global_configuration") is not True:
        raise ValueError("mechanism config must be one global configuration")
    if "pair_overrides" in config:
        raise ValueError("pair-specific mechanism parameters are forbidden")
    if config.get("fixed") != FIXED:
        raise ValueError("mechanism config changes paper-fixed parameters")
    from scripts.saes_execution_identity import validate_saes_execution_identity

    saes_execution_identity = validate_saes_execution_identity(
        config.get("saes_execution_identity")
    )

    projection = config.get("projection")
    if not isinstance(projection, dict) or projection.get("seed") != 42:
        raise ValueError("mechanism projection must use seed 42")
    if projection.get("kind") != "random_hyperplane_lsh":
        raise ValueError("mechanism projection kind is invalid")
    rom_relative = projection.get("rom")
    if not isinstance(rom_relative, str) or not rom_relative:
        raise ValueError("mechanism projection ROM path is missing")
    rom_path = (ROOT / rom_relative).resolve()
    if ROOT not in rom_path.parents or not rom_path.is_file():
        raise ValueError("mechanism projection ROM path is invalid")
    rom_sha256 = sha256_file(rom_path)
    if projection.get("rom_sha256") != rom_sha256:
        raise ValueError("mechanism projection ROM hash mismatch")

    from scripts.calibration_contract import PARAMETER_GRID, validate_candidate

    expected_grid = {name: list(values) for name, values in PARAMETER_GRID.items()}
    if config.get("search_space") != expected_grid:
        raise ValueError("mechanism calibration grid is not the registered grid")
    status = config.get("status")
    calibration = config.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("mechanism calibration provenance is missing")
    contract_sha256 = sha256_file(CALIBRATION_CONTRACT)
    if status == "preregistered":
        if config.get("selected") is not None:
            raise ValueError("preregistered mechanism config cannot select parameters")
        if any(
            calibration.get(field) is not None
            for field in (
                "manifest_sha256",
                "candidate_records_sha256",
            )
        ):
            raise ValueError("preregistered mechanism config has fabricated calibration hashes")
        provenance = {
            "status": status,
            "manifest_sha256": contract_sha256,
            "candidate_records_sha256": None,
            "evaluation_disjoint": False,
            "expected_results_accessed": False,
            "global_configuration": True,
            "saes_execution_route_sha256": saes_execution_identity["route_sha256"],
        }
    elif status == "calibrated":
        selected = config.get("selected")
        validate_candidate({"parameters": selected})
        manifest_sha256 = calibration.get("manifest_sha256")
        candidates_sha256 = calibration.get("candidate_records_sha256")
        if (
            not isinstance(manifest_sha256, str)
            or SHA256.fullmatch(manifest_sha256) is None
            or not isinstance(candidates_sha256, str)
            or SHA256.fullmatch(candidates_sha256) is None
        ):
            raise ValueError("calibrated mechanism config has invalid calibration hashes")
        if calibration.get("evaluation_disjoint") is not True:
            raise ValueError("calibrated mechanism config is not evaluation-disjoint")
        split_provenance = _validated_split_provenance(calibration, selected)
        provenance = {
            "status": status,
            "manifest_sha256": manifest_sha256,
            "candidate_records_sha256": candidates_sha256,
            "evaluation_disjoint": True,
            "expected_results_accessed": False,
            "global_configuration": True,
            "protocol": CALIBRATED_SPLIT_PROTOCOL,
            "train_holdout_scene_disjoint": True,
            "train": split_provenance["train"],
            "holdout": split_provenance["holdout"],
            "saes_execution_route_sha256": saes_execution_identity["route_sha256"],
        }
    else:
        raise ValueError("mechanism config status must be preregistered or calibrated")

    provenance["mechanism_config_sha256"] = sha256_file(path)
    return config, provenance


def require_calibrated_mechanism(
    path: Path = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config, provenance = load_mechanism_config(path)
    if provenance["status"] != "calibrated":
        raise RuntimeError(
            "artifact/mechanism_config.json has not been calibrated; "
            "run `bash scripts/run_ae.sh calibrate` before a claim run"
        )
    if provenance["evaluation_disjoint"] is not True:
        raise RuntimeError("claim mechanism calibration is not evaluation-disjoint")
    return config, provenance
