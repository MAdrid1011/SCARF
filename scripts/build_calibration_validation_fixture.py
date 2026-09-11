#!/usr/bin/env python3
"""Build a non-claim calibration-protocol shape fixture.

The output exercises the 24-train/8-holdout selection contract and records a
candidate tuple for interface validation. It contains no model outputs, no
calibration traces, and no provenance that can support a frozen claim config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calibration_contract import (  # noqa: E402
    QUALITY_LIMITS,
    canonical_sha256,
    canonical_parameters,
)


FIXTURE_SCHEMA_VERSION = "calibration-validation-fixture-v1"
SELECTION_SCHEMA_VERSION = "synthetic-calibration-selection-v1"
FIXTURE_KIND = "non-claim-validation-fixture"
PROTOCOL = "dl3dv_train_holdout_v1"
TRAIN_COUNT = 24
HOLDOUT_COUNT = 8
CANDIDATE_PARAMETERS = canonical_parameters(
    {
        "gamma_depth": 0.075,
        "beta_x": 0.50,
        "beta_f": 0.10,
        "beta_d": 1.00,
    }
)


def _write_json(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject_release_output(path: Path) -> Path:
    resolved = path.resolve()
    forbidden = tuple(
        (ROOT / name).resolve() for name in ("artifact", "outputs", "evidence")
    )
    if any(resolved == item or item in resolved.parents for item in forbidden):
        raise ValueError(
            "calibration validation fixture must be outside artifact/, outputs/, and evidence/"
        )
    return resolved


def _selection_entries(prefix: str, count: int) -> list[dict[str, Any]]:
    return [
        {
            "scene": f"{prefix}-{index:03d}",
            "context_indices": [0, 1],
            "target_indices": [2, 3, 4, 5],
        }
        for index in range(count)
    ]


def _selection_record() -> dict[str, Any]:
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "dataset": "dl3dv",
        "train": _selection_entries("fixture-train", TRAIN_COUNT),
        "holdout": _selection_entries("fixture-holdout", HOLDOUT_COUNT),
    }


def build_fixture(output_dir: Path) -> Path:
    root = _reject_release_output(output_dir)
    selection_path = root / "selection.json"
    _write_json(selection_path, _selection_record())
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    train = selection["train"]
    holdout = selection["holdout"]
    manifest = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "bundle_kind": FIXTURE_KIND,
        "claim_eligible": False,
        "synthetic_fixture": True,
        "not_for_release": True,
        "purpose": "calibration_interface_validation_only",
        "protocol": PROTOCOL,
        "dataset": "dl3dv",
        "selection": {
            "path": "selection.json",
            "sha256": _sha256(selection_path),
            "train_count": len(train),
            "holdout_count": len(holdout),
            "train_scene_set_sha256": canonical_sha256(
                sorted(item["scene"] for item in train)
            ),
            "holdout_scene_set_sha256": canonical_sha256(
                sorted(item["scene"] for item in holdout)
            ),
            "evaluation_disjoint": not (
                {item["scene"] for item in train} & {item["scene"] for item in holdout}
            ),
        },
        "candidate_parameters": CANDIDATE_PARAMETERS,
        "candidate_parameters_sha256": canonical_sha256(CANDIDATE_PARAMETERS),
        "registered_quality_limits": dict(QUALITY_LIMITS),
        "measurement_status": "synthetic_shape_only_no_model_outputs",
    }
    manifest_path = root / "calibration-fixture.json"
    _write_json(manifest_path, manifest)
    validate_fixture(manifest_path, root=root)
    return manifest_path


def validate_fixture(
    manifest_path: Path, *, root: Path | None = None
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"calibration validation fixture not found: {manifest_path}"
        )
    root = Path(root or manifest_path.parent).resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("calibration validation fixture is not valid JSON") from exc
    required = {
        "schema_version",
        "bundle_kind",
        "claim_eligible",
        "synthetic_fixture",
        "not_for_release",
        "purpose",
        "protocol",
        "dataset",
        "selection",
        "candidate_parameters",
        "candidate_parameters_sha256",
        "registered_quality_limits",
        "measurement_status",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != required:
        raise ValueError("calibration validation fixture has an invalid field set")
    if (
        manifest["schema_version"] != FIXTURE_SCHEMA_VERSION
        or manifest["bundle_kind"] != FIXTURE_KIND
        or manifest["claim_eligible"] is not False
        or manifest["synthetic_fixture"] is not True
        or manifest["not_for_release"] is not True
        or manifest["purpose"] != "calibration_interface_validation_only"
        or manifest["protocol"] != PROTOCOL
        or manifest["dataset"] != "dl3dv"
        or manifest["measurement_status"] != "synthetic_shape_only_no_model_outputs"
    ):
        raise ValueError("calibration validation fixture markers are invalid")
    selection = manifest["selection"]
    if not isinstance(selection, Mapping):
        raise ValueError("calibration validation fixture selection is invalid")
    expected_selection = {
        "path",
        "sha256",
        "train_count",
        "holdout_count",
        "train_scene_set_sha256",
        "holdout_scene_set_sha256",
        "evaluation_disjoint",
    }
    if set(selection) != expected_selection:
        raise ValueError("calibration validation fixture selection fields are invalid")
    path = selection["path"]
    if path != "selection.json":
        raise ValueError("calibration validation fixture selection path is invalid")
    selection_path = root / path
    if not selection_path.is_file() or _sha256(selection_path) != selection["sha256"]:
        raise ValueError("calibration validation fixture selection hash mismatch")
    selection_record = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        selection_record.get("schema_version") != SELECTION_SCHEMA_VERSION
        or selection_record.get("protocol") != PROTOCOL
        or selection_record.get("dataset") != "dl3dv"
    ):
        raise ValueError("calibration validation fixture selection metadata is invalid")
    train, holdout = selection_record.get("train"), selection_record.get("holdout")
    if not isinstance(train, list) or not isinstance(holdout, list):
        raise ValueError("calibration validation fixture split entries are missing")
    if len(train) != TRAIN_COUNT or len(holdout) != HOLDOUT_COUNT:
        raise ValueError("calibration validation fixture split counts are invalid")
    train_scenes = {item.get("scene") for item in train}
    holdout_scenes = {item.get("scene") for item in holdout}
    if len(train_scenes) != TRAIN_COUNT or len(holdout_scenes) != HOLDOUT_COUNT:
        raise ValueError("calibration validation fixture contains duplicate scenes")
    if train_scenes & holdout_scenes or selection["evaluation_disjoint"] is not True:
        raise ValueError("calibration validation fixture train/holdout split overlaps")
    for split, entries in (("train", train), ("holdout", holdout)):
        for entry in entries:
            if (
                not isinstance(entry, Mapping)
                or set(entry) != {"scene", "context_indices", "target_indices"}
                or not isinstance(entry["scene"], str)
                or not entry["scene"]
                or entry["context_indices"] != [0, 1]
                or entry["target_indices"] != [2, 3, 4, 5]
            ):
                raise ValueError(
                    f"calibration validation fixture {split} entry is invalid"
                )
    if selection["train_count"] != len(train) or selection["holdout_count"] != len(
        holdout
    ):
        raise ValueError("calibration validation fixture selection count mismatch")
    if selection["train_scene_set_sha256"] != canonical_sha256(sorted(train_scenes)):
        raise ValueError("calibration validation fixture train scene hash mismatch")
    if selection["holdout_scene_set_sha256"] != canonical_sha256(
        sorted(holdout_scenes)
    ):
        raise ValueError("calibration validation fixture holdout scene hash mismatch")
    parameters = canonical_parameters(manifest["candidate_parameters"])
    if parameters != CANDIDATE_PARAMETERS or manifest[
        "candidate_parameters_sha256"
    ] != canonical_sha256(parameters):
        raise ValueError("calibration validation fixture candidate tuple mismatch")
    if manifest["registered_quality_limits"] != QUALITY_LIMITS:
        raise ValueError("calibration validation fixture quality limits mismatch")
    return {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "bundle_kind": FIXTURE_KIND,
        "claim_eligible": False,
        "train_count": len(train),
        "holdout_count": len(holdout),
        "candidate_parameters": parameters,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="create the deterministic fixture")
    build.add_argument("--output-dir", type=Path, required=True)
    validate = commands.add_parser("validate", help="validate an existing fixture")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--root", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            manifest = build_fixture(args.output_dir)
            summary = validate_fixture(manifest, root=manifest.parent)
            print(
                "PASS: built non-claim calibration fixture "
                f"({summary['train_count']} train/{summary['holdout_count']} holdout, "
                f"candidate={summary['candidate_parameters']})"
            )
        else:
            summary = validate_fixture(args.manifest, root=args.root)
            print(
                "PASS: validated non-claim calibration fixture "
                f"({summary['train_count']} train/{summary['holdout_count']} holdout)"
            )
    except (OSError, TypeError, ValueError, KeyError) as exc:
        print(f"FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
