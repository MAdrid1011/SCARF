#!/usr/bin/env python3
"""Stage a validated AE result tree for deterministic archival release."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "artifact/reference_results"
BASE_CATEGORIES = {
    "quality",
    "datasets",
    "rtl",
    "dram",
    "reports",
    "environments",
    "validation",
}


def required_categories() -> set[str]:
    claims = json.loads((ROOT / "artifact/claim_status.json").read_text(encoding="utf-8"))
    categories = set(BASE_CATEGORIES)
    if claims.get("figure8") == "CLAIMED":
        categories.add("speedup")
    if claims.get("sensitivity") == "CLAIMED":
        categories.add("sensitivity")
    if claims.get("physical_asap7") == "CLAIMED" or claims.get("deepscale") == "CLAIMED":
        categories.add("physical")
    return categories


def required_environment_profiles() -> set[str]:
    claims = json.loads((ROOT / "artifact/claim_status.json").read_text(encoding="utf-8"))
    claimed_models = {
        pair.split("/", 1)[0]
        for pair, state in claims.get("software_pairs", {}).items()
        if state == "CLAIMED"
    }
    profiles = set()
    if claimed_models & {"transplat", "mvsplat"}:
        profiles.add("classic")
    if "depthsplat" in claimed_models:
        profiles.add("depthsplat")
    return profiles


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_files(source: Path) -> dict[Path, Path]:
    patterns = (
        "quality/*/results.json",
        "quality/*/pair-execution.json",
        "quality/*/progress.jsonl",
        "quality/*/samples/*/results.json",
        "speedup/*/results.json",
        "speedup/*/samples/*/results.json",
        "speedup/*/samples/*/orin-evidence/measurement.json",
        "speedup/*/samples/*/orin-evidence/cuda-events.json",
        "speedup/*/orin-profile/*",
        "ablation/*/results.json",
        "ablation/*/pair-execution.json",
        "ablation/*/progress.jsonl",
        "ablation/*/samples/*/results.json",
        "sensitivity/results.json",
        "sensitivity/plan.json",
        "datasets/*.json",
        "rtl/results.json",
        "rtl/*.log",
        "rtl/traces/*.vcd",
        "dram/*.json",
        "dram/*.csv",
        "dram/*.jsonl",
        "dram/*.yaml",
        "dram/*.log",
        "dram/*.trace",
        "physical/asap7/*.json",
        "physical/asap7/*.log",
        "physical/asap7/*.tcl",
        "physical/asap7/runtime/iflow/result/ScarfTop.droute.*/droute_drc.rpt",
        "physical/asap7/runtime/iflow/rtl/ScarfTop/sram-proxies.json",
        "reports/*",
        "environments/*.json",
        "validation.json",
    )
    selected = {}
    for pattern in patterns:
        for path in source.glob(pattern):
            if path.is_file():
                selected[path] = path.relative_to(source)
    for category in ("quality", "ablation"):
        category_root = source / category
        if not category_root.is_dir():
            continue
        for pair_root in sorted(path for path in category_root.iterdir() if path.is_dir()):
            sample_roots = sorted(
                path
                for path in (pair_root / "samples").glob("sample_*")
                if (path / "results.json").is_file()
            )
            if not sample_roots:
                continue
            for path in sample_roots[0].glob("*.png"):
                if path.is_file():
                    selected[path] = path.relative_to(source)
    return selected


def stage(source: Path, destination: Path = DESTINATION) -> dict:
    source = source.resolve()
    validation_path = source / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise ValueError("reference evidence requires validation status PASS")
    evidence = destination / "evidence"
    if evidence.exists():
        raise FileExistsError(f"reference evidence already exists: {evidence}")
    files = selected_files(source)
    categories = {
        relative.parts[0] if relative.name != "validation.json" else "validation"
        for relative in files.values()
    }
    missing = sorted(required_categories() - categories)
    if missing:
        raise ValueError("validated output is missing archive categories: " + ", ".join(missing))
    environment_profiles = {
        relative.stem
        for relative in files.values()
        if relative.parts[:1] == ("environments",) and relative.suffix == ".json"
    }
    missing_profiles = sorted(required_environment_profiles() - environment_profiles)
    if missing_profiles:
        raise ValueError(
            "validated output is missing environment profiles: "
            + ", ".join(missing_profiles)
        )
    records = {}
    for path, relative in sorted(files.items(), key=lambda item: str(item[1])):
        target = evidence / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        records[str(Path("evidence") / relative)] = {
            "size": target.stat().st_size,
            "sha256": sha256_file(target),
        }
    manifest = {
        "schema_version": "1.0",
        "status": "complete",
        "validation_status": "PASS",
        "categories": sorted(categories),
        "files": records,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    args = parser.parse_args()
    try:
        manifest = stage(args.input, args.destination.resolve())
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"staged {len(manifest['files'])} reference evidence files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
