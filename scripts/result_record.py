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

from scripts.execution_contract import (
    ASSIGNMENT_CONSENSUS_PSEUDO_DESCRIPTOR_MATERIALIZATION,
    REPRESENTATIVE_SAES_MATERIALIZATION,
    build_execution_contract,
)
from scripts.mechanism_config import load_mechanism_config


ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("feature", "depth", "gaussian", "ggu")
STAGE_COMPONENTS = {
    "s1": "feature",
    "s2": "depth",
    "s3": "gaussian",
    "s4": "ggu",
}


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


def _nonnegative_count(value: Any, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or int(value) != value
    ):
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def _unit_fraction(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{label} must be a finite fraction in [0, 1]")
    return float(value)


def _default_stage_records(cycles: Mapping[str, int]) -> dict[str, dict[str, Any]]:
    return {
        stage: {
            "cycles": int(cycles[component]),
            "useful_mmcu_slots": 0,
            "scheduled_mmcu_slots": 0,
            "mmcu_slots_available": False,
            "source": "component_simulator_without_mmcu_events",
        }
        for stage, component in STAGE_COMPONENTS.items()
    }


def _copy_stage_records(
    records: Mapping[str, Any] | None, cycles: Mapping[str, int]
) -> dict[str, dict[str, Any]]:
    if records is None:
        return _default_stage_records(cycles)
    if set(records) != set(STAGE_COMPONENTS):
        raise ValueError("stage records must contain exactly s1-s4")
    copied: dict[str, dict[str, Any]] = {}
    for stage in STAGE_COMPONENTS:
        value = records[stage]
        if not isinstance(value, Mapping):
            raise ValueError(f"stage record {stage} must be an object")
        copied[stage] = dict(value)
    return copied


def _default_event_records(fsdr_saes: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    fsdr_value = fsdr_saes.get("fsdr", {})
    fsdr = fsdr_value if isinstance(fsdr_value, Mapping) else {}
    total_pixels = _nonnegative_count(fsdr.get("total_pixels", 0), "FSDR total pixels")
    cache_hits = _nonnegative_count(fsdr.get("cache_hits", 0), "FSDR cache hits")
    guided_pixels = _nonnegative_count(fsdr.get("guided", 0), "FSDR guided pixels")
    covered = _nonnegative_count(
        fsdr.get("guided_top1_covered", 0), "FSDR covered Top-1 pixels"
    )
    missed = _nonnegative_count(
        fsdr.get("guided_top1_missed", 0), "FSDR missed Top-1 pixels"
    )
    discrete_top1_available = (
        fsdr.get("discrete_candidate_evidence") is True
        and covered + missed == guided_pixels
    )
    if not discrete_top1_available:
        covered = 0
        missed = 0
    depth_evaluations_available = fsdr.get("depth_evaluations_available") is True
    feature_buffer_bytes_available = (
        fsdr.get("feature_buffer_bytes_available") is True
    )

    saes_value = fsdr_saes.get("saes", {})
    saes = saes_value if isinstance(saes_value, Mapping) else {}
    total_tiles = _nonnegative_count(
        saes.get("total_tiles_processed", 0), "SAES total tiles"
    )
    level0_tiles = _nonnegative_count(saes.get("level0_tiles", 0), "SAES L0 tiles")
    level1_tiles = _nonnegative_count(saes.get("level1_tiles", 0), "SAES L1 tiles")
    full_tiles = _nonnegative_count(saes.get("full_tiles", 0), "SAES full tiles")
    tile_path_available = (
        total_tiles > 0 and level0_tiles + level1_tiles + full_tiles == total_tiles
    )
    effective_gaussians = _nonnegative_count(
        saes.get("effective_gaussians", 0), "SAES effective Gaussians"
    )
    zeroed_gaussians = _nonnegative_count(
        saes.get("zeroed_gaussians", 0), "SAES zeroed Gaussians"
    )
    gaussian_counts_available = "effective_gaussians" in saes and "zeroed_gaussians" in saes
    s2_evaluations_available = saes.get("s2_evaluations_available") is True
    full_s2_evaluations = (
        _nonnegative_count(
            saes.get("full_s2_evaluations", 0), "SAES full S2 evaluations"
        )
        if s2_evaluations_available
        else 0
    )
    executed_s2_evaluations = (
        _nonnegative_count(
            saes.get("executed_s2_evaluations", 0),
            "SAES executed S2 evaluations",
        )
        if s2_evaluations_available
        else 0
    )
    if executed_s2_evaluations > full_s2_evaluations:
        raise ValueError("SAES executed S2 evaluations exceed the full search")
    hardware_accounting = saes.get("hardware_accounting")
    if hardware_accounting is not None and not isinstance(hardware_accounting, Mapping):
        raise ValueError("SAES hardware accounting must be an object when available")
    execution_dependency = saes.get("execution_dependency")
    if execution_dependency is not None and not isinstance(execution_dependency, Mapping):
        raise ValueError("SAES execution dependency must be an object when available")
    requested_saving = saes.get("s2_s3_saving", {})
    if not isinstance(requested_saving, Mapping):
        raise ValueError("SAES S2/S3 saving must be an object when available")

    return {
        "fsdr": {
            "total_pixels": total_pixels,
            "cache_hits": cache_hits,
            "guided_pixels": guided_pixels,
            "hamming_hits": _nonnegative_count(
                fsdr.get("hamming_hits", cache_hits), "FSDR Hamming hits"
            ),
            "local_valid_hits": _nonnegative_count(
                fsdr.get("local_valid_hits", guided_pixels),
                "FSDR local-valid hits",
            ),
            "local_invalid_fallbacks": _nonnegative_count(
                fsdr.get(
                    "local_invalid_fallbacks", max(0, cache_hits - guided_pixels)
                ),
                "FSDR local-invalid fallbacks",
            ),
            "guided_top1_covered": covered,
            "guided_top1_missed": missed,
            "discrete_top1_available": discrete_top1_available,
            "full_depth_evaluations": _nonnegative_count(
                fsdr.get("full_depth_evaluations", 0),
                "FSDR full depth evaluations",
            )
            if depth_evaluations_available
            else 0,
            "executed_depth_evaluations": _nonnegative_count(
                fsdr.get("executed_depth_evaluations", 0),
                "FSDR executed depth evaluations",
            )
            if depth_evaluations_available
            else 0,
            "depth_evaluations_available": depth_evaluations_available,
            "feature_buffer_bytes_baseline": _nonnegative_count(
                fsdr.get("feature_buffer_bytes_baseline", 0),
                "FSDR baseline feature-buffer bytes",
            )
            if feature_buffer_bytes_available
            else 0,
            "feature_buffer_bytes_actual": _nonnegative_count(
                fsdr.get("feature_buffer_bytes_actual", 0),
                "FSDR actual feature-buffer bytes",
            )
            if feature_buffer_bytes_available
            else 0,
            "feature_buffer_bytes_available": feature_buffer_bytes_available,
            "source": "fsdr_path_event_counter",
        },
        "saes": {
            "total_tiles": total_tiles if tile_path_available else 0,
            "level0_tiles": level0_tiles if tile_path_available else 0,
            "level1_tiles": level1_tiles if tile_path_available else 0,
            "full_tiles": full_tiles if tile_path_available else 0,
            "tile_path_available": tile_path_available,
            "baseline_gaussians": (
                effective_gaussians + zeroed_gaussians
                if gaussian_counts_available
                else 0
            ),
            "actual_gaussians": effective_gaussians if gaussian_counts_available else 0,
            "gaussian_counts_available": gaussian_counts_available,
            "full_s2_evaluations": full_s2_evaluations,
            "executed_s2_evaluations": executed_s2_evaluations,
            "s2_evaluations_available": s2_evaluations_available,
            "l0_representatives": _nonnegative_count(
                saes.get("l0_representatives", level0_tiles * 4),
                "SAES L0 representatives",
            ),
            "l1_lightweight_anchors": _nonnegative_count(
                saes.get("l1_lightweight_anchors", level1_tiles * 8),
                "SAES L1 lightweight anchors",
            ),
            "full_stage3_gaussians": _nonnegative_count(
                saes.get("full_stage3_gaussians", full_tiles * 16),
                "SAES full Stage-3 Gaussians",
            ),
            "assignment_weight_sum_error_max": float(
                saes.get("assignment_weight_sum_error_max", 0.0)
            ),
            "opacity_transmittance_error_max": float(
                saes.get("opacity_transmittance_error_max", 0.0)
            ),
            "covariance_psd_violations": _nonnegative_count(
                saes.get("covariance_psd_violations", 0),
                "SAES covariance PSD violations",
            ),
            "hardware_accounting": (
                dict(hardware_accounting)
                if isinstance(hardware_accounting, Mapping)
                else None
            ),
            "execution_dependency": (
                dict(execution_dependency)
                if isinstance(execution_dependency, Mapping)
                else None
            ),
            "s2_s3_saving": {
                "s2": _unit_fraction(
                    requested_saving.get("s2", 0.0), "SAES S2 saving"
                ),
                "s3": _unit_fraction(
                    requested_saving.get("s3", 0.0), "SAES S3 saving"
                ),
            },
            "source": "runtime_summary",
        },
    }


def _copy_event_records(
    records: Mapping[str, Any] | None, fsdr_saes: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    if records is None:
        return _default_event_records(fsdr_saes)
    if set(records) != {"fsdr", "saes"}:
        raise ValueError("event records must contain exactly fsdr and saes")
    copied: dict[str, dict[str, Any]] = {}
    for namespace in ("fsdr", "saes"):
        value = records[namespace]
        if not isinstance(value, Mapping):
            raise ValueError(f"event record {namespace} must be an object")
        copied[namespace] = dict(value)
    return copied


def _upgrade_v21_events(records: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    fsdr = records["fsdr"]
    cache_hits = _nonnegative_count(fsdr.get("cache_hits", 0), "FSDR cache hits")
    guided = _nonnegative_count(fsdr.get("guided_pixels", 0), "FSDR guided pixels")
    hamming_hits = _nonnegative_count(
        fsdr.get("hamming_hits", cache_hits), "FSDR Hamming hits"
    )
    local_valid = _nonnegative_count(
        fsdr.get("local_valid_hits", guided), "FSDR local-valid hits"
    )
    local_fallbacks = _nonnegative_count(
        fsdr.get("local_invalid_fallbacks", hamming_hits - local_valid),
        "FSDR local-invalid fallbacks",
    )
    fsdr.update(
        {
            "hamming_hits": hamming_hits,
            "local_valid_hits": local_valid,
            "local_invalid_fallbacks": local_fallbacks,
        }
    )

    saes = records["saes"]
    level0_tiles = _nonnegative_count(saes.get("level0_tiles", 0), "SAES L0 tiles")
    level1_tiles = _nonnegative_count(saes.get("level1_tiles", 0), "SAES L1 tiles")
    full_tiles = _nonnegative_count(saes.get("full_tiles", 0), "SAES full tiles")
    saes.update(
        {
            "l0_representatives": _nonnegative_count(
                saes.get("l0_representatives", level0_tiles * 4),
                "SAES L0 representatives",
            ),
            "l1_lightweight_anchors": _nonnegative_count(
                saes.get("l1_lightweight_anchors", level1_tiles * 8),
                "SAES L1 lightweight anchors",
            ),
            "full_stage3_gaussians": _nonnegative_count(
                saes.get("full_stage3_gaussians", full_tiles * 16),
                "SAES full Stage-3 Gaussians",
            ),
            "assignment_weight_sum_error_max": float(
                saes.get("assignment_weight_sum_error_max", 0.0)
            ),
            "opacity_transmittance_error_max": float(
                saes.get("opacity_transmittance_error_max", 0.0)
            ),
            "covariance_psd_violations": _nonnegative_count(
                saes.get("covariance_psd_violations", 0),
                "SAES covariance PSD violations",
            ),
        }
    )
    requested_saving = saes.get("s2_s3_saving", {})
    if not isinstance(requested_saving, Mapping):
        raise ValueError("SAES S2/S3 saving must be an object")
    saes["s2_s3_saving"] = {
        "s2": _unit_fraction(requested_saving.get("s2", 0.0), "SAES S2 saving"),
        "s3": _unit_fraction(requested_saving.get("s3", 0.0), "SAES S3 saving"),
    }
    return records


def _bind_saes_execution_dependency(
    records: dict[str, dict[str, Any]], model: str
) -> None:
    """Bind v2.1 SAES cycle savings to the shipped per-model contract."""
    from saes.execution_dependency import resolve_s2_s3_execution_contract

    expected = resolve_s2_s3_execution_contract(model)
    observed = records["saes"].get("execution_dependency")
    if observed is not None:
        if not isinstance(observed, Mapping) or dict(observed) != expected:
            raise ValueError(
                "SAES execution dependency does not match the model contract"
            )
    records["saes"]["execution_dependency"] = expected
    saving = records["saes"]["s2_s3_saving"]
    if not expected["s2_s3_sparse_execution_verified"] and any(
        saving[stage] != 0.0 for stage in ("s2", "s3")
    ):
        raise ValueError(
            "unverified SAES execution dependency cannot claim S2/S3 savings"
        )


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
    source_tree = hashlib.sha256(
        json.dumps(
            {"git_commit": commit, "submodules": submodules, "files": files},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "git_commit": commit,
        "git_dirty": False,
        "submodules": dict(submodules),
        "source": "release_manifest",
        "source_tree_sha256": source_tree,
    }


def _worktree_source_sha256(
    root: Path, commit: str, submodules: Mapping[str, str]
) -> str:
    digest = hashlib.sha256()
    digest.update(commit.encode("ascii") + b"\0")
    digest.update(
        json.dumps(dict(submodules), sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    )
    diff = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--", "."],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    digest.update(b"\0tracked-diff\0" + diff)
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.split(b"\0")
    for raw_path in sorted(path for path in untracked if path):
        path = root / raw_path.decode("utf-8")
        if path.is_file():
            digest.update(b"\0untracked\0" + raw_path + b"\0")
            digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


@lru_cache(maxsize=2)
def source_identity(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    if (root / "release-manifest.json").is_file():
        return _archive_source_identity(root)
    try:
        commit = _git("rev-parse", "HEAD", root=root)
        submodules = _submodule_commits(root)
        return {
            "git_commit": commit,
            "git_dirty": bool(_git("status", "--porcelain", allow_empty=True, root=root)),
            "submodules": submodules,
            "source": "git",
            "source_tree_sha256": _worktree_source_sha256(
                root, commit, submodules
            ),
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
    paper_result_eligible: bool | None = None,
    run_class: str = "diagnostic",
    saes_materialization: str = REPRESENTATIVE_SAES_MATERIALIZATION,
    evidence_class: str = "deterministic_execution",
    stage_records: Mapping[str, Any] | None = None,
    event_records: Mapping[str, Any] | None = None,
    energy_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not dataset_representation:
        raise ValueError("dataset representation must be recorded")
    execution = build_execution_contract(
        run_class=run_class,
        saes_materialization=saes_materialization,
    )
    if (
        execution["saes_materialization"]
        == ASSIGNMENT_CONSENSUS_PSEUDO_DESCRIPTOR_MATERIALIZATION
    ):
        raise ValueError(
            "assignment-consensus pseudo descriptors cannot produce a normal "
            "quality/performance result record"
        )
    if (
        execution["run_class"] in {"claim", "functional"}
        and execution["saes_materialization"]
        != REPRESENTATIVE_SAES_MATERIALIZATION
    ):
        raise ValueError(
            "claim and functional result records require representative SAES "
            "materialization"
        )
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
    checked_stages = _copy_stage_records(stage_records, checked_cycles)
    checked_events = _upgrade_v21_events(
        _copy_event_records(event_records, fsdr_saes)
    )
    _bind_saes_execution_dependency(checked_events, model)
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
    if paper_result_eligible is None:
        paper_result_eligible = (
            not functional_fixture
            and execution["run_class"] == "claim"
            and execution["saes_materialization"]
            == REPRESENTATIVE_SAES_MATERIALIZATION
        )
    elif not isinstance(paper_result_eligible, bool):
        raise ValueError("paper_result_eligible must be a boolean")
    paper_result_eligible = paper_result_eligible and not functional_fixture
    if paper_result_eligible and (
        execution["run_class"] != "claim"
        or execution["saes_materialization"]
        != REPRESENTATIVE_SAES_MATERIALIZATION
    ):
        raise ValueError(
            "paper-result-eligible records require claim run class and "
            "representative SAES materialization"
        )
    _, mechanism = load_mechanism_config()
    if mechanism["status"] != "calibrated":
        paper_result_eligible = False
    elif (
        mechanism.get("protocol") != "dl3dv_train_holdout_v1"
        or mechanism.get("train_holdout_scene_disjoint") is not True
        or not isinstance(mechanism.get("train"), dict)
        or not isinstance(mechanism.get("holdout"), dict)
    ):
        raise RuntimeError(
            "calibrated mechanism provenance has no verified DL3DV holdout evidence"
        )
    calibration_provenance = {
        key: value
        for key, value in mechanism.items()
        if key != "mechanism_config_sha256"
    }
    return {
        "schema_version": "2.1",
        "evidence_class": evidence_class,
        "provenance": {
            "git_commit": source["git_commit"],
            "git_dirty": source["git_dirty"],
            "source_identity": source["source"],
            "source_tree_sha256": source["source_tree_sha256"],
            "mechanism_config_sha256": mechanism["mechanism_config_sha256"],
            "calibration_provenance": calibration_provenance,
            "submodules": source["submodules"],
            "command": portable_command(command),
            "runtime_assets": dict(runtime_assets),
            "execution_contract": execution,
            "seed": int(seed),
            "model": model,
            "environment": dict(environment),
            "device": dict(device),
            "dataset": {
                "name": dataset,
                "representation": dataset_representation,
                "functional_fixture": functional_fixture,
                "paper_result_eligible": paper_result_eligible,
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
            "stages": checked_stages,
        },
        "events": checked_events,
        "energy": dict(
            energy_record
            if energy_record is not None
            else {"available": False, "source": "not_measured"}
        ),
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
