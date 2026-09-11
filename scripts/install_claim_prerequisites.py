#!/usr/bin/env python3
"""Validate and install real calibration and RTL prerequisites for claim runs.

This command intentionally installs only prerequisites.  It does not create
model outputs, timing values, Orin measurements, or a passing validation
report.  It lets a platform owner transfer the frozen calibration output and
complete source-bound timing bundle before invoking the three claim workflows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS = ("transplat", "mvsplat", "depthsplat")
DATASETS = ("re10k", "acid", "dl3dv")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_counts(root: Path, profile: str) -> dict[str, int]:
    if profile == "full":
        protocol = json.loads((root / "artifact/evaluation_protocol.json").read_text(encoding="utf-8"))
        pairs = protocol.get("pairs")
        if not isinstance(pairs, dict):
            raise ValueError("evaluation protocol has no pair records")
        return {
            dataset: int(pairs[f"transplat/{dataset}"]["sample_count"])
            for dataset in DATASETS
        }
    if profile == "reviewer":
        record = json.loads(
            (root / "artifact/protocol/reviewer/manifest.json").read_text(encoding="utf-8")
        )
        datasets = record.get("datasets")
        if not isinstance(datasets, dict):
            raise ValueError("reviewer protocol has no dataset records")
        return {dataset: int(datasets[dataset]["sample_count"]) for dataset in DATASETS}
    raise ValueError(f"unsupported profile: {profile}")


def _find_config(source: Path) -> Path:
    for relative in ("calibration/mechanism_config.json", "artifact/mechanism_config.json"):
        path = source / relative
        if path.is_file():
            return path
    raise FileNotFoundError("missing calibration/mechanism_config.json")


def _validate(source: Path, root: Path, profile: str) -> dict[str, Any]:
    source = source.resolve()
    root = root.resolve()
    if not source.is_dir():
        raise ValueError(f"prerequisite input is not a directory: {source}")
    config_path = _find_config(source)
    from scripts.mechanism_config import require_calibrated_mechanism

    try:
        _config, calibration = require_calibrated_mechanism(config_path)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"calibration configuration is not claim-ready: {exc}") from exc

    manifest_path = source / "timing-backend/manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("missing timing-backend/manifest.json")
    from scripts.claim_timing_backend import load_claim_timing_manifest

    try:
        backend = load_claim_timing_manifest(manifest_path, root=source)
    except (OSError, ValueError) as exc:
        raise ValueError(f"timing backend is not claim-ready: {exc}") from exc

    expected_counts = _expected_counts(root, profile)
    actual_counts = {
        f"{model}/{dataset}": sum(
            1
            for item_model, item_dataset, _index in backend.samples
            if item_model == model and item_dataset == dataset
        )
        for model in MODELS
        for dataset in DATASETS
    }
    missing = {
        pair: {"actual": actual, "expected": expected_counts[pair.split("/", 1)[1]]}
        for pair, actual in actual_counts.items()
        if actual != expected_counts[pair.split("/", 1)[1]]
    }
    if missing:
        raise ValueError(
            "timing backend does not cover the selected profile: "
            + json.dumps(missing, sort_keys=True)
        )
    return {
        "schema_version": "scarf-claim-prerequisites-v1",
        "status": "READY",
        "profile": profile,
        "source": str(source),
        "calibration": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "status": calibration["status"],
            "evaluation_disjoint": calibration["evaluation_disjoint"],
        },
        "timing_backend": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "source_rtl_sha256": backend.backend_provenance["source"]["sha256"],
            "clock_mhz": backend.clock_mhz,
            "sample_counts": actual_counts,
            "sample_count": len(backend.samples),
        },
    }


def _replace_tree(source: Path, target: Path) -> None:
    """Copy a validated bundle through a sibling directory before replacement."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.install-", dir=target.parent) as temporary:
        staged = Path(temporary) / target.name
        shutil.copytree(source, staged, symlinks=False)
        backup = target.with_name(f".{target.name}.previous")
        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(staged, target)
        except BaseException:
            if backup.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup)


def install(
    source: Path,
    *,
    root: Path = ROOT,
    output_root: Path | None = None,
    profile: str = "full",
    check_only: bool = False,
) -> dict[str, Any]:
    root = Path(root).resolve()
    output_root = Path(output_root or root / "outputs").resolve()
    report = _validate(Path(source), root, profile)
    if check_only:
        return {**report, "mode": "check-only"}

    source = Path(source).resolve()
    artifact_config = root / "artifact/mechanism_config.json"
    config_source = _find_config(source)
    temporary_config = artifact_config.with_suffix(".json.install-tmp")
    temporary_config.write_bytes(config_source.read_bytes())
    os.replace(temporary_config, artifact_config)
    _replace_tree(source / "timing-backend", output_root / "timing-backend")
    report.update(
        {
            "mode": "installed",
            "installed_config": str(artifact_config),
            "installed_timing_manifest": str(output_root / "timing-backend/manifest.json"),
            "installed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="real calibration and timing export")
    parser.add_argument("--root", type=Path, default=ROOT, help="release checkout root")
    parser.add_argument("--output-root", type=Path, help="AE output root for timing-backend")
    parser.add_argument("--profile", choices=("full", "reviewer"), default="full")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = install(
            args.input,
            root=args.root,
            output_root=args.output_root,
            profile=args.profile,
            check_only=args.check_only,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
