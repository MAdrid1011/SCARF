"""Build deterministic, schema-valid SCARF experiment records."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("feature", "depth", "gaussian", "ggu")


@lru_cache(maxsize=2)
def build_environment_provenance(profile: str) -> dict[str, Any]:
    if profile not in {"classic", "depthsplat"}:
        raise ValueError(f"unsupported environment profile: {profile}")
    import torch
    import torchvision

    from scripts.check_environment import PROFILE_CONTRACTS

    lock = ROOT / "environments" / profile / "requirements.lock"
    contract = PROFILE_CONTRACTS[profile]
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != contract["python"]:
        raise RuntimeError(
            f"{profile} requires Python {contract['python']}, got {actual_python}"
        )
    if not torch.__version__.startswith(contract["torch"]):
        raise RuntimeError(
            f"{profile} requires PyTorch {contract['torch']}, got {torch.__version__}"
        )
    if not torchvision.__version__.startswith(contract["torchvision"]):
        raise RuntimeError(
            f"{profile} requires TorchVision {contract['torchvision']}, got {torchvision.__version__}"
        )
    if torch.version.cuda is not None and torch.version.cuda != contract["cuda"]:
        raise RuntimeError(
            f"{profile} requires CUDA {contract['cuda']}, got {torch.version.cuda}"
        )
    record = {
        "profile": profile,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "torch": torch.__version__,
        "torchvision": version("torchvision"),
        "torch_cuda": torch.version.cuda,
        "lock_sha256": sha256_file(lock),
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return {**record, "digest_sha256": hashlib.sha256(canonical).hexdigest()}


def sha256_file(path: Path) -> str:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"required provenance file not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=32)
def _cached_sha256_file(path_text: str) -> str:
    return sha256_file(Path(path_text))


def cached_sha256_file(path: Path) -> str:
    path = Path(path).resolve()
    return _cached_sha256_file(str(path))


def require_positive_cycles(cycles: Mapping[str, Any]) -> dict[str, int]:
    checked: dict[str, int] = {}
    for component in COMPONENTS:
        value = cycles.get(component)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{component} cycle count is missing or non-numeric")
        if not math.isfinite(value) or value <= 0 or int(value) != value:
            raise ValueError(f"{component} cycle count must be a positive integer")
        checked[component] = int(value)
    return checked


def strict_stage_error(stage: str, error: BaseException) -> RuntimeError:
    """Build a strict-run failure that preserves the actionable root cause."""
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("strict stage name must be non-empty")
    return RuntimeError(
        f"strict run {stage.strip()} failed; GPU fallback is forbidden: "
        f"{type(error).__name__}: {error}"
    )


def build_quality_record(
    baseline: Mapping[str, float],
    scarf: Mapping[str, float],
) -> dict[str, Any]:
    required = ("psnr_db", "ssim", "lpips")
    for label, values in (("baseline", baseline), ("scarf", scarf)):
        for key in required:
            value = values.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{label}.{key} must be finite")
    baseline_psnr = float(baseline["psnr_db"])
    if baseline_psnr <= 0:
        raise ValueError("baseline.psnr_db must be positive")
    signed = (float(scarf["psnr_db"]) - baseline_psnr) / baseline_psnr * 100.0
    return {
        "baseline": {key: float(baseline[key]) for key in required},
        "scarf": {key: float(scarf[key]) for key in required},
        "change": {
            "psnr_signed_pct": signed,
            "psnr_degradation_pct": max(0.0, -signed),
            "psnr_absolute_pct": abs(signed),
        },
    }


def mean_view_quality(
    quality_views: list[Mapping[str, Any]], variant: str
) -> dict[str, float]:
    """Return the exact arithmetic mean of one variant's per-view metrics."""
    if variant not in {"baseline", "scarf"}:
        raise ValueError(f"unsupported quality variant: {variant}")
    if not quality_views:
        raise ValueError("quality view records cannot be empty")
    aggregate: dict[str, float] = {}
    for metric in ("psnr_db", "ssim", "lpips"):
        values = []
        for view in quality_views:
            metrics = view.get(variant)
            value = metrics.get(metric) if isinstance(metrics, Mapping) else None
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError(f"quality view {variant}.{metric} must be finite")
            values.append(float(value))
        aggregate[metric] = sum(values) / len(values)
    return aggregate


def _git(*args: str, allow_empty: bool = False, root: Path = ROOT) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0 or (not allow_empty and not result.stdout.strip()):
        raise RuntimeError(f"cannot determine git provenance: {' '.join(args)}")
    return result.stdout.strip()


def _submodule_commits(root: Path = ROOT) -> dict[str, str]:
    commits: dict[str, str] = {}
    for name in ("transplat", "mvsplat", "depthsplat"):
        commit = _git("-C", str(root / name), "rev-parse", "HEAD", root=root)
        if len(commit) != 40:
            raise RuntimeError(f"invalid {name} submodule commit: {commit}")
        commits[name] = commit
    return commits


def _archive_source_identity(root: Path) -> dict[str, Any]:
    path = root / "release-manifest.json"
    if not path.is_file():
        raise RuntimeError("cannot determine source provenance from Git or release manifest")
    record = json.loads(path.read_text(encoding="utf-8"))
    commit = record.get("git_commit")
    submodules = record.get("submodules")
    files = record.get("files")
    if (
        record.get("bundle_kind") != "source"
        or record.get("validation", {}).get("pass") is not True
        or not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or not isinstance(submodules, dict)
        or set(submodules) != {"transplat", "mvsplat", "depthsplat"}
        or any(
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
            for value in submodules.values()
        )
        or not isinstance(files, dict)
        or not files
    ):
        raise RuntimeError("release manifest has invalid source provenance")
    for relative, expected in files.items():
        pure = PurePosixPath(relative)
        source = root / pure
        if (
            not isinstance(relative, str)
            or pure.is_absolute()
            or ".." in pure.parts
            or not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
            or not source.is_file()
            or sha256_file(source) != expected
        ):
            raise RuntimeError(f"release source hash mismatch: {relative}")
    return {
        "git_commit": commit,
        "git_dirty": False,
        "submodules": dict(submodules),
        "source": "release_manifest",
    }


@lru_cache(maxsize=2)
def source_identity(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    if (root / "release-manifest.json").is_file():
        return _archive_source_identity(root)
    try:
        return {
            "git_commit": _git("rev-parse", "HEAD", root=root),
            "git_dirty": bool(
                _git("status", "--porcelain", allow_empty=True, root=root)
            ),
            "submodules": _submodule_commits(root),
            "source": "git",
        }
    except RuntimeError:
        return _archive_source_identity(root)


def _display_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return f"<external>/{resolved.name}"


def portable_command(command: list[str]) -> list[str]:
    normalized = []
    for index, value in enumerate(command):
        text = str(value)
        candidate = Path(text)
        if index == 0 and candidate.name.startswith("python"):
            normalized.append("python")
        elif candidate.is_absolute():
            normalized.append(_display_path(candidate))
        else:
            normalized.append(text)
    return normalized


def build_result_record(
    *,
    model: str,
    dataset: str,
    checkpoint: Path,
    checkpoint_load: Mapping[str, Any],
    environment: Mapping[str, Any],
    dataset_manifest: Path,
    dataset_representation: str,
    dataset_tree_sha256: str,
    device: Mapping[str, Any],
    seed: int,
    quality: Mapping[str, Mapping[str, float]],
    quality_views: list[Mapping[str, Any]],
    baseline_cycles: int,
    cycles: Mapping[str, Any],
    cycle_source: str,
    ablation: Mapping[str, Any],
    fsdr_saes: Mapping[str, Any],
    command: list[str],
    runtime_assets: Mapping[str, Any],
    sample_identity: Mapping[str, Any],
    scarf_cycles: int | None = None,
    sample_index: int = 0,
    execution_index: int | None = None,
    num_samples: int = 1,
    baseline_source: str = "diagnostic_device_timing",
    fallback_stages: list[str] | None = None,
) -> dict[str, Any]:
    if not dataset_representation:
        raise ValueError("dataset representation must be recorded")
    environment_digest = environment.get("digest_sha256")
    if (
        not isinstance(environment_digest, str)
        or len(environment_digest) != 64
        or any(character not in "0123456789abcdef" for character in environment_digest)
    ):
        raise ValueError("environment provenance has no SHA256 digest")
    matched_tensors = checkpoint_load.get("matched_tensors")
    matched_fraction = checkpoint_load.get("matched_checkpoint_numel_fraction")
    if (
        not isinstance(matched_tensors, int)
        or isinstance(matched_tensors, bool)
        or matched_tensors <= 0
        or not isinstance(matched_fraction, (int, float))
        or isinstance(matched_fraction, bool)
        or not math.isfinite(matched_fraction)
        or not 0 < matched_fraction <= 1
    ):
        raise ValueError("checkpoint load report has no positive matched coverage")
    if len(dataset_tree_sha256) != 64:
        raise ValueError("dataset tree SHA256 must contain 64 hexadecimal characters")
    try:
        int(dataset_tree_sha256, 16)
    except ValueError as exc:
        raise ValueError("dataset tree SHA256 must be hexadecimal") from exc
    checked_cycles = require_positive_cycles(cycles)
    if baseline_cycles <= 0:
        raise ValueError("baseline cycle count must be positive")
    if scarf_cycles is None:
        scarf_cycles = sum(checked_cycles.values())
    if scarf_cycles <= 0:
        raise ValueError("SCARF cycle count must be positive")
    if execution_index is None:
        execution_index = sample_index
    if (
        num_samples <= 0
        or sample_index < 0
        or execution_index < 0
        or execution_index >= num_samples
    ):
        raise ValueError("invalid sample selection")
    if baseline_source not in {
        "workstation_cuda_events",
        "cpu_perf_counter",
        "orin_nx_cuda_events",
        "diagnostic_device_timing",
    }:
        raise ValueError(f"invalid baseline timing source: {baseline_source}")
    fallback_stages = list(fallback_stages or [])
    if any(
        stage not in COMPONENTS or fallback_stages.count(stage) != 1
        for stage in fallback_stages
    ):
        raise ValueError("fallback stages must be unique SCARF component names")
    scene = sample_identity.get("scene")
    context_indices = sample_identity.get("context_indices")
    target_indices = sample_identity.get("target_indices")
    if not isinstance(scene, str) or not scene:
        raise ValueError("sample identity requires a scene key")
    if not isinstance(context_indices, list) or not context_indices:
        raise ValueError("sample identity requires context indices")
    if not isinstance(target_indices, list) or not target_indices:
        raise ValueError("sample identity requires target indices")
    if len(quality_views) != len(target_indices):
        raise ValueError("quality view records must cover every target index")
    if [view.get("target_index") for view in quality_views] != target_indices:
        raise ValueError("quality view records do not match selected target indices")
    for variant in ("baseline", "scarf"):
        aggregate = mean_view_quality(quality_views, variant)
        for metric, mean in aggregate.items():
            if not math.isclose(
                mean, float(quality[variant][metric]), rel_tol=1e-5, abs_tol=1e-6
            ):
                raise ValueError(
                    f"quality {variant}.{metric} is not the target-view mean"
                )
    quality_record = build_quality_record(quality["baseline"], quality["scarf"])
    quality_record["views"] = [
        {
            "target_index": int(view["target_index"]),
            "baseline": {
                metric: float(view["baseline"][metric])
                for metric in ("psnr_db", "ssim", "lpips")
            },
            "scarf": {
                metric: float(view["scarf"][metric])
                for metric in ("psnr_db", "ssim", "lpips")
            },
        }
        for view in quality_views
    ]
    source = source_identity()
    functional_fixture = dataset_representation == "re10k-synthetic-functional-v1"
    return {
        "schema_version": "1.0",
        "provenance": {
            "git_commit": source["git_commit"],
            "git_dirty": source["git_dirty"],
            "source_identity": source["source"],
            "submodules": source["submodules"],
            "command": portable_command(command),
            "runtime_assets": dict(runtime_assets),
            "seed": int(seed),
            "model": model,
            "environment": dict(environment),
            "device": dict(device),
            "dataset": {
                "name": dataset,
                "representation": dataset_representation,
                "functional_fixture": functional_fixture,
                "paper_result_eligible": not functional_fixture,
                "manifest": _display_path(dataset_manifest),
                "sha256": sha256_file(dataset_manifest),
                "tree_sha256": dataset_tree_sha256,
            },
            "checkpoint": {
                "path": _display_path(checkpoint),
                "sha256": cached_sha256_file(checkpoint),
                "load": dict(checkpoint_load),
            },
            "evaluation": {
                "kind": "sample",
                "sample_index": int(sample_index),
                "execution_index": int(execution_index),
                "candidate_count": int(num_samples),
                "scene": scene,
                "context_indices": [int(value) for value in context_indices],
                "target_indices": [int(value) for value in target_indices],
                "target_view_count": len(target_indices),
                "target_view_aggregation": "arithmetic mean over selected target views",
            },
        },
        "quality": quality_record,
        "performance": {
            "baseline_cycles": int(baseline_cycles),
            "scarf_cycles": int(scarf_cycles),
            "speedup": float(baseline_cycles) / float(scarf_cycles),
            "baseline_source": baseline_source,
            "cycle_source": cycle_source,
            "components": checked_cycles,
        },
        "ablation": dict(ablation),
        "fsdr_saes": dict(fsdr_saes),
        "hardware": {
            "physical_ppa_included": False,
            "source": "run hardware/iflow/run.sh separately",
        },
        "validation": {
            "reproducible": not fallback_stages,
            "reference_fallback_used": bool(fallback_stages),
            "fallback_stages": fallback_stages,
        },
    }


def write_result(record: Mapping[str, Any], output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
