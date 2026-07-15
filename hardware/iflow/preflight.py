#!/usr/bin/env python3
"""Validate a pinned, clean iFlow checkout and build its execution plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


EXPECTED_COMMIT = "04b4d98b1a69d00bbe04d52b09105667332a295d"
DEFAULT_CONTAINER_IMAGE = "iedaopensource/iflow:latest"
DESIGN = "ScarfTop"
STAGES = (
    "synth",
    "floorplan",
    "tapcell",
    "pdn",
    "gplace",
    "resize",
    "dplace",
    "cts",
    "filler",
    "groute",
    "droute",
    "layout",
)
COLLATERAL = (
    "foundry/asap7/lef/asap7_tech_1x_201209.lef",
    "foundry/asap7/lef/asap7sc7p5t_27_R_1x_201211.lef",
    "foundry/asap7/gds/asap7sc7p5t_27_R_1x_201211.gds",
    "foundry/asap7/lib/asap7sc7p5t_AO_RVT_TT_nldm_201020.lib",
    "foundry/asap7/lib/asap7sc7p5t_INVBUF_RVT_TT_nldm_201020.lib",
    "foundry/asap7/lib/asap7sc7p5t_OA_RVT_TT_nldm_201020.lib",
    "foundry/asap7/lib/asap7sc7p5t_SEQ_RVT_TT_nldm_201020.lib",
    "foundry/asap7/lib/asap7sc7p5t_SIMPLE_RVT_TT_nldm_201020.lib",
    "foundry/asap7/verilog/blackbox.v",
    "foundry/asap7/blackbox_map.tcl",
)
REQUIRED = (
    "scripts/run_flow.py",
    "scripts/cfg/flow_cfg.py",
    "scripts/gcd/synth.yosys_0.9.tcl",
    "scripts/gcd/floorplan.openroad_1.2.0.tcl",
    "tools/yosys4be891e8/bin/yosys",
    "tools/OpenROADae191807/bin/openroad",
    *COLLATERAL,
)


class PreflightError(ValueError):
    pass


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise PreflightError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_container_image(image: str) -> dict:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise PreflightError(f"Docker image is unavailable: {image}: {detail}")
    try:
        records = json.loads(result.stdout)
        record = records[0]
    except (json.JSONDecodeError, IndexError, KeyError) as exc:
        raise PreflightError(f"cannot inspect Docker image: {image}") from exc
    image_id = record.get("Id")
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise PreflightError(f"Docker image has no immutable image ID: {image}")
    repo_digests = sorted(record.get("RepoDigests") or [])
    return {
        "engine": "docker",
        "requested_image": image,
        "image_id": image_id,
        "repo_digests": repo_digests,
    }


def parse_stages(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return STAGES
    selected = tuple(item.strip() for item in stage.split(",") if item.strip())
    if not selected:
        raise PreflightError("at least one stage is required")
    unknown = [item for item in selected if item not in STAGES]
    if unknown:
        raise PreflightError(f"unsupported iFlow stage: {', '.join(unknown)}")
    positions = [STAGES.index(item) for item in selected]
    if positions != sorted(set(positions)):
        raise PreflightError("stages must be unique and in physical-flow order")
    return selected


def validate_checkout(root: Path) -> dict:
    root = Path(root).resolve()
    if not (root / ".git").exists():
        raise PreflightError(f"not an iFlow git checkout: {root}")
    commit = _git(root, "rev-parse", "HEAD")
    if commit != EXPECTED_COMMIT:
        raise PreflightError(
            f"iFlow commit mismatch: expected {EXPECTED_COMMIT}, got {commit}"
        )
    dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise PreflightError("iFlow checkout is dirty; use a clean pinned checkout")
    missing = [relative for relative in REQUIRED if not (root / relative).is_file()]
    if missing:
        raise PreflightError(f"missing iFlow files: {', '.join(missing)}")
    not_executable = [
        relative
        for relative in (
            "scripts/run_flow.py",
            "tools/yosys4be891e8/bin/yosys",
            "tools/OpenROADae191807/bin/openroad",
        )
        if not os.access(root / relative, os.X_OK)
    ]
    if not_executable:
        raise PreflightError(f"iFlow tools are not executable: {', '.join(not_executable)}")
    return {
        "worktree": {"path": "runtime/iflow", "path_base": "output_dir"},
        "commit": commit,
        "clean": True,
        "collateral_sha256": {
            relative: sha256_file(root / relative) for relative in COLLATERAL
        },
    }


def build_plan(root: Path, stage: str, container_image: str = DEFAULT_CONTAINER_IMAGE) -> dict:
    checkout = validate_checkout(root)
    stages = parse_stages(stage)
    command = [
        "./run_flow.py",
        "-d",
        DESIGN,
        "-s",
        ",".join(stages),
        "-f",
        "asap7",
        "-t",
        "HS",
        "-c",
        "TYP",
        "-v",
        "AE",
        "-l",
        "AE",
    ]
    return {
        "schema_version": "1.0",
        "design": DESIGN,
        "platform": "asap7",
        "process": "predictive_7nm",
        "stages": list(stages),
        "command": command,
        "checkout": checkout,
        "container": inspect_container_image(container_image),
        "scope": {
            "logic": "standard-cell implementation",
            "sram": "abstract proxy only; not expanded into registers",
            "excluded": ["LPDDR PHY", "pads", "foundry-specific I/O"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iflow-root", type=Path, required=True)
    parser.add_argument("--stage", default="all")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--container-image", default=DEFAULT_CONTAINER_IMAGE)
    args = parser.parse_args()
    try:
        plan = build_plan(args.iflow_root, args.stage, args.container_image)
    except PreflightError as exc:
        print(f"error: {exc}")
        return 2
    text = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
