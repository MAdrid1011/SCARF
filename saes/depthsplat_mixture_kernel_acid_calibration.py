"""Frozen ACID 24/8 calibration for the DepthSplat v3 kernel-risk guard.

The record is deliberately separate from the literal T=4 V16 LOO record.
It freezes only source-camera analytic mixture-kernel risks collected from
ACID context-only sidecars.  No target frame, renderer, decoder, or quality
metric is part of this module or its on-disk evidence.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping as MappingABC
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from saes.depthsplat_acid_disjoint_calibration import (
    DEFAULT_MATERIALIZATION_ROOT,
    DEFAULT_PLAN_PATH,
    canonical_sha256,
    resolve_depthsplat_acid_binding,
)
from saes.depthsplat_l0_l1_materializer import (
    DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_AGGREGATE_SCHEMA,
    DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE,
    DEPTHSPLAT_SOFT_MIXTURE_T4_MOMENT_CERTIFICATE,
    _mixture_kernel_closure_aggregate_from_trace,
)
from saes.depthsplat_mixture_kernel_guard import (
    KIND as KERNEL_CLOSURE_KIND,
    POLICY as KERNEL_CLOSURE_POLICY,
    SCHEMA_VERSION as KERNEL_CLOSURE_SCHEMA_VERSION,
)
from saes.depthsplat_soft_mixture_certificate import (
    KIND as SOFT_MIXTURE_CERTIFICATE_KIND,
    POLICY as SOFT_MIXTURE_CERTIFICATE_POLICY,
    SCHEMA_VERSION as SOFT_MIXTURE_CERTIFICATE_SCHEMA_VERSION,
)
from saes.probe_first_schedule import (
    BALANCED_L1_ANCHOR_SEMANTICS,
    DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_PLAN_CONTRACT,
    SOFT_MIXTURE_KERNEL_CLOSURE_GUARD_POLICY,
    SOFT_MIXTURE_KERNEL_CLOSURE_T4_L0_SECONDARY_PREFETCH_POLICY,
    soft_mixture_kernel_closure_t4_route_config_sha256,
)
from saes.progressive_saes import PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "1.0"
TRAIN_SPLIT = "calibration_train"
HOLDOUT_SPLIT = "calibration_holdout"
SPLITS = (TRAIN_SPLIT, HOLDOUT_SPLIT)

KERNEL_RISK_KIND = "depthsplat-soft-mixture-kernel-risk-acid-24-8-v1"
FROZEN_STATUS = "FROZEN_EVALUATION_DISJOINT_TARGET_FREE"
KERNEL_RISK_RECORD_NAME = "kernel-risk-v3.json"
KERNEL_RISK_PROFILE_ID = "depthsplat-soft-mixture-kernel-closure-t4-acid-risk-v1"
KERNEL_RISK_PROFILE_SCHEMA = "depthsplat-soft-mixture-kernel-risk-profile-v1"
KERNEL_RISK_METRIC = "maximum-world-or-source-analytic-l2-kernel-risk-v1"
KERNEL_RISK_THRESHOLD_RULE = "train-minimum-per-scene-q25"
KERNEL_RISK_GUARD_SCHEMA = "depthsplat-mixture-kernel-risk-frozen-acid-guard-v1"
KERNEL_RISK_TRACE_ARTIFACT_SCHEMA = "depthsplat-mixture-kernel-acid-trace-v1"
KERNEL_RISK_TRACE_ARTIFACT_KIND = "depthsplat-mixture-kernel-risk-trace"
KERNEL_RISK_COLLECTION_POLICY = "strict-numerical-abstention-observes-valid-candidates-v1"

_ACCESS_KEYS = (
    "target_mapping_present",
    "target_rgb_accessed",
    "target_camera_metadata_accessed",
    "target_index_accessed",
    "skipped_s3_attributes_accessed",
)
_SHA256_CHARS = frozenset("0123456789abcdef")
_COLLECTOR_SOURCE_CONTRACT = "depthsplat-mixture-kernel-acid-collector-source-v1"
_COLLECTOR_SOURCE_FILES = (
    "saes/depthsplat_mixture_kernel_acid_calibration.py",
    "saes/depthsplat_mixture_kernel_acid_collector.py",
    "scripts/saes_depthsplat_mixture_kernel_acid_calibration.py",
    "saes/depthsplat_literal_t4_acid_calibration.py",
    "saes/depthsplat_literal_t4_acid_collector.py",
    "saes/depthsplat_acid_disjoint_calibration.py",
    "saes/depthsplat_backend.py",
    "saes/depthsplat_l0_l1_materializer.py",
    "saes/depthsplat_mixture_kernel_guard.py",
    "saes/depthsplat_soft_mixture_certificate.py",
    "saes/depthsplat_selected_output.py",
    "saes/probe_first_schedule.py",
    "saes/probe_layout.py",
    "saes/progressive_saes.py",
    "saes/selected_output_replay.py",
    "integration/model_loader.py",
    "integration/acid_joint_context.py",
    "integration/acid_joint_model_context.py",
    "scripts/ae_config.py",
    "scripts/saes_selected_output_replay_audit.py",
)
_SOFT_MIXTURE_SOURCE_ONLY = {
    "source_camera_only": True,
    "target_mapping_present": False,
    "target_rgb_accessed": False,
    "target_camera_metadata_accessed": False,
    "target_index_accessed": False,
    "omitted_s3_attributes_accessed": False,
    "input_covariances_mutated": False,
    "fixed_covariance_scale": True,
    "boolean_owner_assignment_used": False,
    "projected_domain_guard_used": False,
}

_VERIFIED_KERNEL_RISK_GUARD_ISSUER = object()


@dataclass(frozen=True, eq=False, init=False)
class VerifiedMixtureKernelRiskGuard(MappingABC[str, Any]):
    """Immutable v3 threshold capability issued after a live record reload."""

    _projection: Mapping[str, Any] = field(repr=False)
    _issuer: object = field(repr=False, compare=False)

    def __init__(self, projection: Mapping[str, Any], *, _issuer: object | None = None) -> None:
        if _issuer is not _VERIFIED_KERNEL_RISK_GUARD_ISSUER:
            raise TypeError("kernel-risk guards must be issued by the verified loader")
        if not isinstance(projection, MappingABC):
            raise TypeError("kernel-risk guard projection must be a mapping")
        object.__setattr__(self, "_projection", MappingProxyType(dict(projection)))
        object.__setattr__(self, "_issuer", _issuer)

    def __getitem__(self, key: str) -> Any:
        return self._projection[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._projection)

    def __len__(self) -> int:
        return len(self._projection)

    def as_dict(self) -> dict[str, Any]:
        return dict(self._projection)


def _issue_verified_mixture_kernel_risk_guard(
    projection: Mapping[str, Any],
) -> VerifiedMixtureKernelRiskGuard:
    return VerifiedMixtureKernelRiskGuard(
        projection, _issuer=_VERIFIED_KERNEL_RISK_GUARD_ISSUER
    )


def verified_mixture_kernel_materializer_guard_projection(value: object) -> dict[str, Any]:
    """Return a verified projection and reject forged serialized mappings."""

    if not isinstance(value, VerifiedMixtureKernelRiskGuard):
        raise TypeError("v3 kernel-risk materialization requires an authenticated ACID guard")
    if getattr(value, "_issuer", None) is not _VERIFIED_KERNEL_RISK_GUARD_ISSUER:
        raise ValueError("kernel-risk guard authentication changed")
    return value.as_dict()


# The shorter name is retained for local callers written before the materializer
# integration name was fixed.  Both names enforce the same opaque capability.
verified_mixture_kernel_risk_guard_projection = (
    verified_mixture_kernel_materializer_guard_projection
)


def _access_record() -> dict[str, bool]:
    return {name: False for name in _ACCESS_KEYS}


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"kernel-risk calibration cannot read {Path(path).name}") from error
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unavailable or invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _finite_quantile(values: Sequence[float], fraction: float) -> float:
    if (
        not values
        or isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not 0.0 <= float(fraction) <= 1.0
    ):
        raise ValueError("kernel-risk calibration requires a nonempty valid quantile")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) and value >= 0.0 for value in ordered):
        raise ValueError("kernel-risk calibration values must be finite and nonnegative")
    return ordered[round((len(ordered) - 1) * float(fraction))]


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("kernel-risk calibration summary requires values")
    return {
        "candidate_count": len(values),
        "minimum": _finite_quantile(values, 0.0),
        "p10": _finite_quantile(values, 0.10),
        "p25": _finite_quantile(values, 0.25),
        "p50": _finite_quantile(values, 0.50),
        "p75": _finite_quantile(values, 0.75),
        "p90": _finite_quantile(values, 0.90),
        "p95": _finite_quantile(values, 0.95),
        "p99": _finite_quantile(values, 0.99),
        "maximum": _finite_quantile(values, 1.0),
    }


def _nonnegative_finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"kernel-risk {label} must be finite and nonnegative")
    return float(value)


def _kernel_route_events() -> dict[str, Any]:
    return {
        "contract_version": DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_PLAN_CONTRACT,
        "tile_size": 4,
        "feature_threshold": 0.20,
        "depth_threshold": 0.10,
        "decision_semantics": PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS,
        "feature_statistic": "normalized-probe-vector-standard-deviation",
        "assignment_feature_semantics": "unit-normalized-bilinear-s1-v1",
        "l1_anchor_semantics": BALANCED_L1_ANCHOR_SEMANTICS,
        "l0_anchor_count": 4,
        "l1_anchor_count": 12,
        "formal_paper_kp4": False,
        "depth_checked_after_l0_miss_only": False,
        "soft_mixture_kernel_closure_l0_secondary_prefetch_policy": (
            SOFT_MIXTURE_KERNEL_CLOSURE_T4_L0_SECONDARY_PREFETCH_POLICY
        ),
        "kernel_closure_guard_policy": SOFT_MIXTURE_KERNEL_CLOSURE_GUARD_POLICY,
    }


def mixture_kernel_risk_v3_profile() -> dict[str, Any]:
    """Return the only ACID profile that may issue a positive v3 threshold."""

    route_events = _kernel_route_events()
    return {
        "schema_version": KERNEL_RISK_PROFILE_SCHEMA,
        "id": KERNEL_RISK_PROFILE_ID,
        "materialization_profile": DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_MATERIALIZATION_PROFILE,
        "route_plan_contract": DEPTHSPLAT_SOFT_MIXTURE_KERNEL_CLOSURE_T4_PLAN_CONTRACT,
        "tile_size": 4,
        "feature_threshold": 0.20,
        "depth_threshold": 0.10,
        "decision_semantics": PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS,
        "feature_statistic": "normalized-probe-vector-standard-deviation",
        "assignment_feature_semantics": "unit-normalized-bilinear-s1-v1",
        "l1_anchor_semantics": BALANCED_L1_ANCHOR_SEMANTICS,
        "l0_anchor_count": 4,
        "l1_anchor_count": 12,
        "formal_paper_kp4": False,
        "depth_checked_after_l0_miss_only": False,
        "l0_secondary_prefetch_policy": SOFT_MIXTURE_KERNEL_CLOSURE_T4_L0_SECONDARY_PREFETCH_POLICY,
        "kernel_closure_guard_policy": SOFT_MIXTURE_KERNEL_CLOSURE_GUARD_POLICY,
        "kernel_closure_schema_version": KERNEL_CLOSURE_SCHEMA_VERSION,
        "kernel_closure_kind": KERNEL_CLOSURE_KIND,
        "kernel_closure_policy": KERNEL_CLOSURE_POLICY,
        "kernel_risk_collection_policy": KERNEL_RISK_COLLECTION_POLICY,
        "moment_merge_certificate": DEPTHSPLAT_SOFT_MIXTURE_T4_MOMENT_CERTIFICATE,
        "moment_covariance_scale": 1.0,
        "support_containment_guard": False,
        "projected_domain_guard": False,
        "alpha_union_used": False,
        "route_plan_config_sha256": soft_mixture_kernel_closure_t4_route_config_sha256(
            route_events
        ),
    }


def mixture_kernel_risk_v3_profile_sha256() -> str:
    return canonical_sha256(mixture_kernel_risk_v3_profile())


def _validate_profile(value: Any) -> dict[str, Any]:
    expected = mixture_kernel_risk_v3_profile()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ValueError("kernel-risk calibration profile changed")
    if value.get("route_plan_config_sha256") != soft_mixture_kernel_closure_t4_route_config_sha256(
        _kernel_route_events()
    ):
        raise ValueError("kernel-risk calibration route configuration changed")
    return expected


def _validate_acid_binding(value: Any) -> dict[str, Any]:
    required = {
        "schema_version",
        "kind",
        "dataset",
        "protocol_id",
        "plan_sha256",
        "plan_file_sha256",
        "materialization_record_sha256",
        "materialization_tree_sha256",
        "materialization_manifest_sha256",
        "splits",
        "access",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("kernel-risk ACID binding has an invalid schema")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != "saes-acid-context-only-calibration-binding"
        or value.get("dataset") != "acid"
        or not isinstance(value.get("protocol_id"), str)
        or not value["protocol_id"]
        or value.get("access") != _access_record()
    ):
        raise ValueError("kernel-risk ACID binding is not context-only")
    for field in (
        "plan_sha256",
        "plan_file_sha256",
        "materialization_record_sha256",
        "materialization_tree_sha256",
        "materialization_manifest_sha256",
    ):
        _require_sha256(value.get(field), f"kernel-risk ACID {field}")
    splits = value.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(SPLITS):
        raise ValueError("kernel-risk ACID splits are invalid")
    normalized: dict[str, dict[str, Any]] = {}
    for split, expected_count in ((TRAIN_SPLIT, 24), (HOLDOUT_SPLIT, 8)):
        entry = splits.get(split)
        required_split = {
            "scenes",
            "scene_count",
            "scene_set_sha256",
            "selection_sha256",
            "sidecar_tree_sha256",
            "sidecar_manifest_sha256",
            "input_provenance_sha256",
        }
        if not isinstance(entry, Mapping) or set(entry) != required_split:
            raise ValueError(f"kernel-risk ACID {split} binding is invalid")
        scenes = entry.get("scenes")
        if (
            not isinstance(scenes, list)
            or len(scenes) != expected_count
            or entry.get("scene_count") != expected_count
            or len(set(scenes)) != expected_count
            or any(not isinstance(scene, str) or not scene for scene in scenes)
            or canonical_sha256(sorted(scenes)) != entry.get("scene_set_sha256")
        ):
            raise ValueError(f"kernel-risk ACID {split} scenes are invalid")
        for field in required_split - {"scenes", "scene_count"}:
            _require_sha256(entry.get(field), f"kernel-risk ACID {split} {field}")
        normalized[split] = dict(entry)
    if set(normalized[TRAIN_SPLIT]["scenes"]) & set(normalized[HOLDOUT_SPLIT]["scenes"]):
        raise ValueError("kernel-risk ACID train and holdout scenes overlap")
    return {**dict(value), "splits": normalized, "access": _access_record()}


def _collector_source_identity(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    files: list[dict[str, str]] = []
    for relative_path in _COLLECTOR_SOURCE_FILES:
        path = (root / relative_path).resolve()
        if root not in path.parents or not path.is_file() or path.is_symlink():
            raise ValueError("kernel-risk collector source file is invalid")
        files.append({"path": relative_path, "sha256": _sha256_file(path)})
    return {
        "contract": _COLLECTOR_SOURCE_CONTRACT,
        "files": files,
        "tree_sha256": canonical_sha256(files),
    }


def _validate_application_shape(value: Any) -> dict[str, Any]:
    required = {"model", "dataset", "backend_identity", "collector_source"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("kernel-risk application has an invalid schema")
    if (
        value.get("model") != "depthsplat"
        or value.get("dataset") != "dl3dv"
        or not isinstance(value.get("backend_identity"), Mapping)
    ):
        raise ValueError("kernel-risk application is not DepthSplat/DL3DV")
    source = value.get("collector_source")
    if not isinstance(source, Mapping) or set(source) != {"contract", "files", "tree_sha256"}:
        raise ValueError("kernel-risk collector source has an invalid schema")
    if source.get("contract") != _COLLECTOR_SOURCE_CONTRACT:
        raise ValueError("kernel-risk collector source contract changed")
    files = source.get("files")
    if not isinstance(files, list) or len(files) != len(_COLLECTOR_SOURCE_FILES):
        raise ValueError("kernel-risk collector source files are invalid")
    for relative_path, entry in zip(_COLLECTOR_SOURCE_FILES, files):
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
            raise ValueError("kernel-risk collector source entry is invalid")
        if entry.get("path") != relative_path:
            raise ValueError("kernel-risk collector source ordering changed")
        _require_sha256(entry.get("sha256"), "kernel-risk collector source")
    if source.get("tree_sha256") != canonical_sha256(files):
        raise ValueError("kernel-risk collector source tree changed")
    return {
        "model": "depthsplat",
        "dataset": "dl3dv",
        "backend_identity": dict(value["backend_identity"]),
        "collector_source": {
            "contract": _COLLECTOR_SOURCE_CONTRACT,
            "files": [dict(entry) for entry in files],
            "tree_sha256": source["tree_sha256"],
        },
    }


def build_mixture_kernel_risk_application(
    *, backend_identity: Mapping[str, Any], root: Path = ROOT
) -> dict[str, Any]:
    if not isinstance(backend_identity, Mapping):
        raise TypeError("kernel-risk application requires a frozen backend identity")
    return _validate_application_shape(
        {
            "model": "depthsplat",
            "dataset": "dl3dv",
            "backend_identity": dict(backend_identity),
            "collector_source": _collector_source_identity(Path(root)),
        }
    )


def _validate_live_application(value: Any, *, root: Path) -> dict[str, Any]:
    normalized = _validate_application_shape(value)
    from saes.depthsplat_backend import (
        resolve_depthsplat_backend_contract,
        validate_frozen_depthsplat_backend_identity,
    )

    contract = resolve_depthsplat_backend_contract(Path(root))
    identity = validate_frozen_depthsplat_backend_identity(
        contract, normalized["backend_identity"]
    )
    expected = build_mixture_kernel_risk_application(
        backend_identity=identity, root=Path(root)
    )
    if normalized != expected:
        raise ValueError("kernel-risk application identity changed")
    return expected


def _source_only_kernel_flags(value: Any) -> dict[str, bool]:
    expected = {
        "source_camera_only": True,
        "target_mapping_present": False,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "target_index_accessed": False,
        "omitted_s3_attributes_accessed": False,
        "boolean_owner_assignment_used": False,
        "projected_domain_guard_used": False,
        "covariance_expansion_used": False,
        "alpha_union_used": False,
    }
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ValueError("kernel-risk closure crossed its source-only contract")
    return expected


def _certificate_passed(value: Any, sha256: Any, *, tile_key: list[int]) -> bool:
    if (
        not isinstance(value, Mapping)
        or _require_sha256(sha256, "kernel-risk S/R certificate") != canonical_sha256(value)
        or value.get("schema_version") != SOFT_MIXTURE_CERTIFICATE_SCHEMA_VERSION
        or value.get("kind") != SOFT_MIXTURE_CERTIFICATE_KIND
        or value.get("policy") != SOFT_MIXTURE_CERTIFICATE_POLICY
        or value.get("tile_key") != tile_key
        or not isinstance(value.get("passed"), bool)
        or value.get("source_only") != _SOFT_MIXTURE_SOURCE_ONLY
    ):
        raise ValueError("kernel-risk S/R certificate changed")
    if value["passed"] is False:
        return False
    summary = value.get("summary")
    if (
        not isinstance(summary, Mapping)
        or summary.get("input_valid") is not True
    ):
        raise ValueError("kernel-risk S/R certificate summary changed")
    return True


def _validate_closure(
    value: Any, sha256: Any, *, tile_key: list[int], level: str
) -> dict[str, float] | None:
    if (
        not isinstance(value, Mapping)
        or _require_sha256(sha256, "kernel-risk closure") != canonical_sha256(value)
        or value.get("schema_version") != KERNEL_CLOSURE_SCHEMA_VERSION
        or value.get("kind") != KERNEL_CLOSURE_KIND
        or value.get("policy") != KERNEL_CLOSURE_POLICY
        or value.get("tile_key") != tile_key
        or not isinstance(value.get("passed"), bool)
    ):
        raise ValueError("kernel-risk closure evidence changed")
    _source_only_kernel_flags(value.get("source_only"))
    summary = value.get("summary")
    if not isinstance(summary, Mapping) or not isinstance(summary.get("input_valid"), bool):
        raise ValueError("kernel-risk closure validity changed")
    input_valid = summary["input_valid"]
    if not input_valid:
        if (
            value.get("passed") is not False
            or value.get("binding") is not None
            or not isinstance(summary.get("reason"), str)
            or any(
                summary.get(name) is not None
                for name in (
                    "maximum_world_kernel_risk",
                    "maximum_source_kernel_risk",
                    "maximum_kernel_risk",
                    "maximum_source_log_depth_rms",
                )
            )
        ):
            raise ValueError("unscorable kernel-risk candidates must remain Full")
        _nonnegative_finite(
            summary.get("strict_maximum_relative_risk"), "collection strict threshold"
        )
        return None
    if not isinstance(value.get("binding"), Mapping):
        raise ValueError("kernel-risk closure binding is missing")
    fields = {
        name: _nonnegative_finite(summary.get(name), name)
        for name in (
            "maximum_world_kernel_risk",
            "maximum_source_kernel_risk",
            "maximum_kernel_risk",
            "maximum_source_log_depth_rms",
        )
    }
    threshold = _nonnegative_finite(
        summary.get("strict_maximum_relative_risk"), "collection strict threshold"
    )
    if (
        fields["maximum_kernel_risk"]
        != max(fields["maximum_world_kernel_risk"], fields["maximum_source_kernel_risk"])
        or value.get("passed") is not (fields["maximum_kernel_risk"] <= threshold)
        or level not in {"L0", "L1"}
    ):
        raise ValueError("kernel-risk closure decision changed")
    return fields


def kernel_risk_observations_from_preflight_trace(
    trace: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Extract exactly the finite, S/R-certified closure candidates in a trace."""

    if not isinstance(trace, Sequence) or isinstance(trace, (str, bytes)) or not trace:
        raise ValueError("kernel-risk preflight trace is invalid")
    observations: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, str, str]] = set()
    passed_certificate_sha256: set[str] = set()
    closure_certificate_sha256: set[str] = set()
    for entry in trace:
        if not isinstance(entry, Mapping):
            raise ValueError("kernel-risk preflight tile is invalid")
        tile_key = [entry.get("view"), entry.get("tile_y"), entry.get("tile_x")]
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in tile_key):
            raise ValueError("kernel-risk preflight tile identity changed")
        candidate_specs = (
            (
                "mixture_kernel_closure",
                "mixture_kernel_closure_sha256",
                "soft_mixture_certificate",
                "soft_mixture_certificate_sha256",
                entry.get("accepted_level"),
            ),
            (
                "l0_mixture_kernel_closure_failure",
                "l0_mixture_kernel_closure_failure_sha256",
                "l0_soft_mixture_certificate_before_kernel_closure",
                "l0_soft_mixture_certificate_before_kernel_closure_sha256",
                "L0",
            ),
            (
                "l1_mixture_kernel_closure_failure",
                "l1_mixture_kernel_closure_failure_sha256",
                "l1_soft_mixture_certificate_before_kernel_closure",
                "l1_soft_mixture_certificate_before_kernel_closure_sha256",
                "L1",
            ),
        )
        for closure_key, closure_sha_key, certificate_key, certificate_sha_key, level in candidate_specs:
            closure = entry.get(closure_key)
            closure_sha = entry.get(closure_sha_key)
            certificate = entry.get(certificate_key)
            certificate_sha = entry.get(certificate_sha_key)
            if closure is None and closure_sha is None:
                if certificate is None and certificate_sha is None:
                    continue
                if _certificate_passed(certificate, certificate_sha, tile_key=tile_key):
                    passed_certificate_sha256.add(str(certificate_sha))
                continue
            if level not in {"L0", "L1"}:
                raise ValueError("kernel-risk accepted closure has no compact level")
            fields = _validate_closure(
                closure, closure_sha, tile_key=tile_key, level=str(level)
            )
            closure_tile_key = closure.get("tile_key") if isinstance(closure, Mapping) else None
            if not _certificate_passed(
                certificate, certificate_sha, tile_key=closure_tile_key
            ):
                raise ValueError("kernel-risk candidate lacks a passed S/R certificate")
            passed_certificate_sha256.add(str(certificate_sha))
            closure_certificate_sha256.add(str(certificate_sha))
            if fields is None:
                if closure_key == "mixture_kernel_closure":
                    raise ValueError("unscorable kernel-risk candidate was accepted compact")
                continue
            key = (*tile_key, closure_key, str(level))
            if key in seen:
                raise ValueError("kernel-risk trace duplicates a candidate")
            seen.add(key)
            observations.append(
                {
                    "tile_key": tile_key,
                    "attempt_level": str(level),
                    "trace_key": closure_key,
                    "closure_sha256": str(closure_sha),
                    **fields,
                }
            )
    if passed_certificate_sha256 != closure_certificate_sha256:
        raise ValueError("kernel-risk trace omits an S/R-passed kernel attempt")
    return observations


def _validate_kernel_aggregate(
    trace: Sequence[Mapping[str, Any]], value: Any
) -> dict[str, Any]:
    expected = _mixture_kernel_closure_aggregate_from_trace(trace)
    if (
        not isinstance(value, Mapping)
        or dict(value) != expected
        or value.get("schema") != DEPTHSPLAT_MIXTURE_KERNEL_CLOSURE_AGGREGATE_SCHEMA
        or value.get("kind") != KERNEL_CLOSURE_KIND
        or value.get("policy") != KERNEL_CLOSURE_POLICY
    ):
        raise ValueError("kernel-risk aggregate does not match its trace")
    return expected


def build_kernel_risk_trace_artifact(
    *,
    scene: str,
    split: str,
    preflight_tile_trace: Sequence[Mapping[str, Any]],
    mixture_kernel_closure_aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(scene, str) or not scene or split not in SPLITS:
        raise ValueError("kernel-risk trace artifact scene identity is invalid")
    trace = [dict(entry) for entry in preflight_tile_trace]
    aggregate = _validate_kernel_aggregate(trace, mixture_kernel_closure_aggregate)
    observations = kernel_risk_observations_from_preflight_trace(trace)
    payload = {
        "schema_version": KERNEL_RISK_TRACE_ARTIFACT_SCHEMA,
        "kind": KERNEL_RISK_TRACE_ARTIFACT_KIND,
        "scene": scene,
        "split": split,
        "preflight_tile_trace": trace,
        "mixture_kernel_closure_aggregate": aggregate,
        "kernel_risk_observations": observations,
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def _trace_reference(value: Any) -> dict[str, str]:
    required = {
        "relative_path",
        "sha256",
        "preflight_tile_trace_sha256",
        "kernel_risk_observations_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("kernel-risk trace reference has an invalid schema")
    relative_path = value.get("relative_path")
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        raise ValueError("kernel-risk trace path is invalid")
    parsed = PurePosixPath(relative_path)
    if (
        parsed.is_absolute()
        or not parsed.parts
        or any(part in {"", ".", ".."} for part in parsed.parts)
        or parsed.as_posix() != relative_path
    ):
        raise ValueError("kernel-risk trace path is not safely relative")
    return {
        "relative_path": relative_path,
        "sha256": _require_sha256(value.get("sha256"), "kernel-risk trace artifact"),
        "preflight_tile_trace_sha256": _require_sha256(
            value.get("preflight_tile_trace_sha256"), "kernel-risk trace"
        ),
        "kernel_risk_observations_sha256": _require_sha256(
            value.get("kernel_risk_observations_sha256"), "kernel-risk observations"
        ),
    }


def _load_trace_artifact(
    reference: Mapping[str, Any],
    *,
    record_directory: Path,
    scene: str,
    split: str,
    observations: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_reference = _trace_reference(reference)
    record_directory = Path(record_directory).resolve()
    artifact_path = record_directory
    for component in PurePosixPath(normalized_reference["relative_path"]).parts:
        artifact_path = artifact_path / component
        if artifact_path.is_symlink():
            raise ValueError("kernel-risk trace artifact path contains a symlink")
    try:
        artifact_path = artifact_path.resolve(strict=True)
    except OSError as error:
        raise ValueError("kernel-risk trace artifact is unavailable") from error
    if not artifact_path.is_file() or record_directory not in artifact_path.parents:
        raise ValueError("kernel-risk trace artifact escapes its record directory")
    artifact = _read_json(artifact_path, "kernel-risk trace artifact")
    recorded_sha = artifact.pop("sha256", None)
    if (
        recorded_sha != canonical_sha256(artifact)
        or recorded_sha != normalized_reference["sha256"]
    ):
        raise ValueError("kernel-risk trace artifact SHA256 is invalid")
    required = {
        "schema_version",
        "kind",
        "scene",
        "split",
        "preflight_tile_trace",
        "mixture_kernel_closure_aggregate",
        "kernel_risk_observations",
    }
    if (
        set(artifact) != required
        or artifact.get("schema_version") != KERNEL_RISK_TRACE_ARTIFACT_SCHEMA
        or artifact.get("kind") != KERNEL_RISK_TRACE_ARTIFACT_KIND
        or artifact.get("scene") != scene
        or artifact.get("split") != split
    ):
        raise ValueError("kernel-risk trace artifact identity changed")
    trace = artifact.get("preflight_tile_trace")
    rebuilt_aggregate = _validate_kernel_aggregate(trace, artifact.get("mixture_kernel_closure_aggregate"))
    rebuilt_observations = kernel_risk_observations_from_preflight_trace(trace)
    if (
        rebuilt_aggregate != dict(aggregate)
        or rebuilt_observations != list(observations)
        or artifact.get("mixture_kernel_closure_aggregate") != dict(aggregate)
        or artifact.get("kernel_risk_observations") != list(observations)
        or normalized_reference["preflight_tile_trace_sha256"] != canonical_sha256(trace)
        or normalized_reference["kernel_risk_observations_sha256"]
        != canonical_sha256(rebuilt_observations)
    ):
        raise ValueError("kernel-risk trace artifact is not linked to its scene record")
    return {**artifact, "sha256": recorded_sha}


def _validate_scene_evidence(
    value: Any,
    *,
    profile_sha256: str,
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required = {
        "profile_sha256",
        "route_plan_config_sha256",
        "native_execution_sha256",
        "initial_attribute_binding_sha256",
        "final_selected_attribute_binding_sha256",
        "selected_head_replay_fallback_summary",
        "materialized_attribute_binding_sha256",
        "full_passthrough_mask_sha256",
        "full_attribute_binding_sha256",
        "full_attributes_bitwise_native",
        "coverage_certificate",
        "coverage_certificate_sha256",
        "accepted_update_slots_sha256",
        "accepted_update_slot_count",
        "mixture_kernel_closure_aggregate",
        "mixture_kernel_closure_aggregate_sha256",
        "mixture_kernel_closure_trace_sha256",
        "trace_artifact",
        "source_nonprobe_s3_attribute_reads",
        "nonzero_merge_applied",
        "renderer_executed",
        "quality_metrics_computed",
        "whole_pipeline_s2_s3_sparse_execution_verified",
        "global_s2_s3_savings_claimed",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("kernel-risk scene evidence has an invalid schema")
    if (
        value.get("profile_sha256") != profile_sha256
        or value.get("route_plan_config_sha256")
        != mixture_kernel_risk_v3_profile()["route_plan_config_sha256"]
        or value.get("coverage_certificate") != DEPTHSPLAT_SOFT_MIXTURE_T4_MOMENT_CERTIFICATE
        or value.get("full_attributes_bitwise_native") is not True
        or not isinstance(value.get("nonzero_merge_applied"), bool)
        or value.get("renderer_executed") is not False
        or value.get("quality_metrics_computed") is not False
        or value.get("whole_pipeline_s2_s3_sparse_execution_verified") is not False
        or value.get("global_s2_s3_savings_claimed") is not False
        or value.get("source_nonprobe_s3_attribute_reads") != 0
    ):
        raise ValueError("kernel-risk scene evidence crossed its source-only contract")
    for field in (
        "native_execution_sha256",
        "initial_attribute_binding_sha256",
        "final_selected_attribute_binding_sha256",
        "materialized_attribute_binding_sha256",
        "full_passthrough_mask_sha256",
        "full_attribute_binding_sha256",
        "coverage_certificate_sha256",
        "accepted_update_slots_sha256",
        "mixture_kernel_closure_aggregate_sha256",
        "mixture_kernel_closure_trace_sha256",
    ):
        _require_sha256(value.get(field), f"kernel-risk scene {field}")
    count = value.get("accepted_update_slot_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("kernel-risk scene accepted update count is invalid")
    aggregate = value.get("mixture_kernel_closure_aggregate")
    if not isinstance(aggregate, Mapping):
        raise ValueError("kernel-risk scene has no closure aggregate")
    if (
        value.get("mixture_kernel_closure_aggregate_sha256") != canonical_sha256(aggregate)
        or value.get("mixture_kernel_closure_trace_sha256")
        != canonical_sha256([dict(item) for item in observations])
    ):
        raise ValueError("kernel-risk scene aggregate binding changed")
    _trace_reference(value.get("trace_artifact"))
    return dict(value)


def _normalize_observations(
    value: Any, *, label: str, require_nonempty: bool
) -> list[dict[str, Any]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or (require_nonempty and not value)
    ):
        raise ValueError(f"kernel-risk {label} observations are invalid")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, str, str]] = set()
    for entry in value:
        required = {
            "tile_key",
            "attempt_level",
            "trace_key",
            "closure_sha256",
            "maximum_world_kernel_risk",
            "maximum_source_kernel_risk",
            "maximum_kernel_risk",
            "maximum_source_log_depth_rms",
        }
        if not isinstance(entry, Mapping) or set(entry) != required:
            raise ValueError(f"kernel-risk {label} observation schema changed")
        tile_key = entry.get("tile_key")
        level = entry.get("attempt_level")
        trace_key = entry.get("trace_key")
        if (
            not isinstance(tile_key, list)
            or len(tile_key) != 3
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in tile_key)
            or level not in {"L0", "L1"}
            or trace_key
            not in {
                "mixture_kernel_closure",
                "l0_mixture_kernel_closure_failure",
                "l1_mixture_kernel_closure_failure",
            }
        ):
            raise ValueError(f"kernel-risk {label} observation identity changed")
        if (trace_key == "l0_mixture_kernel_closure_failure" and level != "L0") or (
            trace_key == "l1_mixture_kernel_closure_failure" and level != "L1"
        ):
            raise ValueError(f"kernel-risk {label} observation level changed")
        fields = {
            name: _nonnegative_finite(entry.get(name), f"{label} {name}")
            for name in (
                "maximum_world_kernel_risk",
                "maximum_source_kernel_risk",
                "maximum_kernel_risk",
                "maximum_source_log_depth_rms",
            )
        }
        if fields["maximum_kernel_risk"] != max(
            fields["maximum_world_kernel_risk"], fields["maximum_source_kernel_risk"]
        ):
            raise ValueError(f"kernel-risk {label} maximum risk changed")
        key = (*tile_key, trace_key, level)
        if key in seen:
            raise ValueError(f"kernel-risk {label} observations are duplicated")
        seen.add(key)
        normalized.append(
            {
                "tile_key": [int(item) for item in tile_key],
                "attempt_level": level,
                "trace_key": trace_key,
                "closure_sha256": _require_sha256(
                    entry.get("closure_sha256"), f"kernel-risk {label} closure"
                ),
                **fields,
            }
        )
    return normalized


def _normalize_scene_records(
    records: Sequence[Mapping[str, Any]],
    *,
    binding: Mapping[str, Any],
    split: str,
    profile_sha256: str,
    record_directory: Path | None = None,
    seen_trace_artifacts: set[str] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError(f"kernel-risk {split} records are invalid")
    expected_scenes = list(binding["splits"][split]["scenes"])
    if len(records) != len(expected_scenes):
        raise ValueError(f"kernel-risk {split} scene count changed")
    normalized: list[dict[str, Any]] = []
    for index, (scene, record) in enumerate(zip(expected_scenes, records)):
        if (
            not isinstance(record, Mapping)
            or set(record) != {"scene", "kernel_risk_observations", "evidence", "access"}
            or record.get("scene") != scene
            or record.get("access") != _access_record()
        ):
            raise ValueError(f"kernel-risk {split} scene record is invalid at {index}")
        observations = _normalize_observations(
            record.get("kernel_risk_observations"),
            label=split,
            require_nonempty=(split == TRAIN_SPLIT),
        )
        evidence = _validate_scene_evidence(
            record.get("evidence"), profile_sha256=profile_sha256, observations=observations
        )
        reference = _trace_reference(evidence["trace_artifact"])
        if seen_trace_artifacts is not None:
            if reference["relative_path"] in seen_trace_artifacts:
                raise ValueError("kernel-risk scene records reuse a trace artifact")
            seen_trace_artifacts.add(reference["relative_path"])
        if record_directory is not None:
            _load_trace_artifact(
                reference,
                record_directory=record_directory,
                scene=scene,
                split=split,
                observations=observations,
                aggregate=evidence["mixture_kernel_closure_aggregate"],
            )
        normalized.append(
            {
                "scene": scene,
                "kernel_risk_observations": observations,
                "evidence": evidence,
                "access": _access_record(),
            }
        )
    return normalized


def _scene_values(record: Mapping[str, Any]) -> list[float]:
    return [
        float(observation["maximum_kernel_risk"])
        for observation in record["kernel_risk_observations"]
    ]


def _split_verification(records: Sequence[Mapping[str, Any]], *, threshold: float) -> dict[str, Any]:
    values = [value for record in records for value in _scene_values(record)]
    retained = sum(value <= threshold for value in values)
    return {
        "threshold_value": threshold,
        "candidate_count": len(values),
        "retained_count": retained,
        "promoted_count": len(values) - retained,
        "summary": _summary(values) if values else None,
        "threshold_updated": False,
    }


def build_kernel_risk_record(
    *,
    binding: Mapping[str, Any],
    application: Mapping[str, Any],
    profile: Mapping[str, Any],
    train_scene_records: Sequence[Mapping[str, Any]],
    holdout_scene_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze a strictly positive train-only minimum-per-scene q25 threshold."""

    validated_binding = _validate_acid_binding(binding)
    validated_application = _validate_application_shape(application)
    validated_profile = _validate_profile(profile)
    profile_sha = mixture_kernel_risk_v3_profile_sha256()
    train = _normalize_scene_records(
        train_scene_records,
        binding=validated_binding,
        split=TRAIN_SPLIT,
        profile_sha256=profile_sha,
    )
    holdout = _normalize_scene_records(
        holdout_scene_records,
        binding=validated_binding,
        split=HOLDOUT_SPLIT,
        profile_sha256=profile_sha,
    )
    per_scene_q25 = [_finite_quantile(_scene_values(record), 0.25) for record in train]
    threshold = min(per_scene_q25)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("kernel-risk calibration cannot freeze a nonpositive threshold")
    train_values = [value for record in train for value in _scene_values(record)]
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": KERNEL_RISK_KIND,
        "status": FROZEN_STATUS,
        "paper_result_eligible": False,
        "application": validated_application,
        "acid_binding": validated_binding,
        "profile": validated_profile,
        "profile_sha256": profile_sha,
        "signal": {
            "name": KERNEL_RISK_METRIC,
            "source": "target-free-v3-actual-anchor-virtual-analytic-l2-kernel-closure",
            "smaller_is_safer": True,
        },
        "split_policy": {
            "train_split": TRAIN_SPLIT,
            "holdout_split": HOLDOUT_SPLIT,
            "threshold_source": "train_only",
            "holdout_threshold_update_allowed": False,
        },
        "access": _access_record(),
        "train_scene_records": train,
        "holdout_scene_records": holdout,
        "train_summary": _summary(train_values),
        "holdout_verification": _split_verification(holdout, threshold=threshold),
        "threshold": {
            "risk_metric": KERNEL_RISK_METRIC,
            "value": threshold,
            "promote_when": "maximum_kernel_risk_gt_value_or_unscorable",
            "rule": KERNEL_RISK_THRESHOLD_RULE,
            "per_scene_q25": per_scene_q25,
            "positive_value_required": True,
        },
    }
    return {**record, "sha256": canonical_sha256(record)}


def _read_record(path: Path) -> dict[str, Any]:
    record = _read_json(Path(path), "kernel-risk frozen record")
    recorded_sha = record.pop("sha256", None)
    if recorded_sha != canonical_sha256(record):
        raise ValueError("kernel-risk frozen record SHA256 is invalid")
    if record.get("kind") != KERNEL_RISK_KIND or record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("kernel-risk frozen record kind or schema is invalid")
    return {**record, "sha256": recorded_sha}


def _validate_frozen_record(
    record: Mapping[str, Any],
    *,
    root: Path,
    plan_path: Path,
    materialization_root: Path,
    record_directory: Path,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "kind",
        "status",
        "paper_result_eligible",
        "application",
        "acid_binding",
        "profile",
        "profile_sha256",
        "signal",
        "split_policy",
        "access",
        "train_scene_records",
        "holdout_scene_records",
        "train_summary",
        "holdout_verification",
        "threshold",
        "sha256",
    }
    if not isinstance(record, Mapping) or set(record) != required:
        raise ValueError("kernel-risk frozen record has unexpected fields")
    if (
        record.get("kind") != KERNEL_RISK_KIND
        or record.get("schema_version") != SCHEMA_VERSION
        or record.get("status") != FROZEN_STATUS
        or record.get("paper_result_eligible") is not False
        or record.get("access") != _access_record()
        or record.get("profile_sha256") != mixture_kernel_risk_v3_profile_sha256()
        or record.get("signal")
        != {
            "name": KERNEL_RISK_METRIC,
            "source": "target-free-v3-actual-anchor-virtual-analytic-l2-kernel-closure",
            "smaller_is_safer": True,
        }
        or record.get("split_policy")
        != {
            "train_split": TRAIN_SPLIT,
            "holdout_split": HOLDOUT_SPLIT,
            "threshold_source": "train_only",
            "holdout_threshold_update_allowed": False,
        }
    ):
        raise ValueError("kernel-risk record is not a frozen target-free calibration")
    profile = _validate_profile(record.get("profile"))
    application = _validate_live_application(record.get("application"), root=Path(root))
    if record.get("application") != application:
        raise ValueError("kernel-risk application identity changed")
    live_binding = _validate_acid_binding(
        resolve_depthsplat_acid_binding(
            plan_path=Path(plan_path), materialization_root=Path(materialization_root)
        )
    )
    if record.get("acid_binding") != live_binding:
        raise ValueError("kernel-risk ACID 24/8 binding changed")
    seen_trace_artifacts: set[str] = set()
    train = _normalize_scene_records(
        record.get("train_scene_records"),
        binding=live_binding,
        split=TRAIN_SPLIT,
        profile_sha256=mixture_kernel_risk_v3_profile_sha256(),
        record_directory=record_directory,
        seen_trace_artifacts=seen_trace_artifacts,
    )
    holdout = _normalize_scene_records(
        record.get("holdout_scene_records"),
        binding=live_binding,
        split=HOLDOUT_SPLIT,
        profile_sha256=mixture_kernel_risk_v3_profile_sha256(),
        record_directory=record_directory,
        seen_trace_artifacts=seen_trace_artifacts,
    )
    per_scene_q25 = [_finite_quantile(_scene_values(scene), 0.25) for scene in train]
    threshold = min(per_scene_q25)
    expected_threshold = {
        "risk_metric": KERNEL_RISK_METRIC,
        "value": threshold,
        "promote_when": "maximum_kernel_risk_gt_value_or_unscorable",
        "rule": KERNEL_RISK_THRESHOLD_RULE,
        "per_scene_q25": per_scene_q25,
        "positive_value_required": True,
    }
    if threshold <= 0.0 or record.get("threshold") != expected_threshold:
        raise ValueError("kernel-risk threshold is not reproducible from positive train-only risks")
    train_values = [value for scene in train for value in _scene_values(scene)]
    if record.get("train_summary") != _summary(train_values):
        raise ValueError("kernel-risk train summary changed")
    if record.get("holdout_verification") != _split_verification(holdout, threshold=threshold):
        raise ValueError("kernel-risk holdout verification changed")
    return {
        **dict(record),
        "profile": profile,
        "application": application,
        "acid_binding": live_binding,
        "threshold_value": float(threshold),
    }


def load_frozen_kernel_risk_threshold(
    path: Path,
    *,
    root: Path = ROOT,
    plan_path: Path = DEFAULT_PLAN_PATH,
    materialization_root: Path = DEFAULT_MATERIALIZATION_ROOT,
) -> dict[str, Any]:
    """Live-reload a frozen positive threshold against ACID and source identities."""

    record_path = Path(path)
    if record_path.is_symlink():
        raise ValueError("kernel-risk frozen record must not be a symlink")
    try:
        resolved = record_path.resolve(strict=True)
    except OSError as error:
        raise ValueError("kernel-risk frozen record is unavailable") from error
    if not resolved.is_file():
        raise ValueError("kernel-risk frozen record is invalid")
    return _validate_frozen_record(
        _read_record(resolved),
        root=Path(root),
        plan_path=Path(plan_path),
        materialization_root=Path(materialization_root),
        record_directory=resolved.parent,
    )


def _to_materializer_guard(record: Mapping[str, Any]) -> VerifiedMixtureKernelRiskGuard:
    required = {
        "schema_version",
        "kind",
        "status",
        "paper_result_eligible",
        "application",
        "acid_binding",
        "profile",
        "profile_sha256",
        "signal",
        "split_policy",
        "access",
        "train_scene_records",
        "holdout_scene_records",
        "train_summary",
        "holdout_verification",
        "threshold",
        "sha256",
    }
    if not isinstance(record, Mapping) or set(record) not in (required, required | {"threshold_value"}):
        raise ValueError("kernel-risk materializer guard record has unexpected fields")
    payload = {key: value for key, value in record.items() if key not in {"sha256", "threshold_value"}}
    if record.get("sha256") != canonical_sha256(payload):
        raise ValueError("kernel-risk materializer guard record SHA256 is invalid")
    profile = _validate_profile(record.get("profile"))
    threshold = record.get("threshold")
    if not isinstance(threshold, Mapping):
        raise ValueError("kernel-risk materializer guard threshold is invalid")
    value = _nonnegative_finite(threshold.get("value"), "frozen threshold")
    if value <= 0.0:
        raise ValueError("kernel-risk materializer guard threshold must be positive")
    projection = {
        "schema_version": KERNEL_RISK_GUARD_SCHEMA,
        "frozen_record_kind": KERNEL_RISK_KIND,
        "frozen_record_sha256": record["sha256"],
        "threshold_value": value,
        "threshold_rule": KERNEL_RISK_THRESHOLD_RULE,
        "risk_metric": KERNEL_RISK_METRIC,
        "materialization_profile": profile["materialization_profile"],
        "route_plan_contract": profile["route_plan_contract"],
        "route_plan_config_sha256": profile["route_plan_config_sha256"],
        "kernel_closure_schema_version": KERNEL_CLOSURE_SCHEMA_VERSION,
        "kernel_closure_kind": KERNEL_CLOSURE_KIND,
        "kernel_closure_policy": KERNEL_CLOSURE_POLICY,
        "acid_binding_sha256": canonical_sha256(record["acid_binding"]),
        "application_sha256": canonical_sha256(record["application"]),
    }
    return _issue_verified_mixture_kernel_risk_guard(projection)


def to_materializer_guard(
    path: Path,
    *,
    root: Path = ROOT,
    plan_path: Path = DEFAULT_PLAN_PATH,
    materialization_root: Path = DEFAULT_MATERIALIZATION_ROOT,
) -> VerifiedMixtureKernelRiskGuard:
    """Load, live-revalidate, and project the only admissible v3 threshold."""

    if isinstance(path, Mapping) or not isinstance(path, (str, Path)):
        raise TypeError("kernel-risk materializer guard requires a record path")
    return _to_materializer_guard(
        load_frozen_kernel_risk_threshold(
            Path(path),
            root=Path(root),
            plan_path=Path(plan_path),
            materialization_root=Path(materialization_root),
        )
    )


__all__ = [
    "DEFAULT_MATERIALIZATION_ROOT",
    "DEFAULT_PLAN_PATH",
    "FROZEN_STATUS",
    "HOLDOUT_SPLIT",
    "KERNEL_RISK_GUARD_SCHEMA",
    "KERNEL_RISK_KIND",
    "KERNEL_RISK_METRIC",
    "KERNEL_RISK_PROFILE_ID",
    "KERNEL_RISK_RECORD_NAME",
    "KERNEL_RISK_THRESHOLD_RULE",
    "TRAIN_SPLIT",
    "VerifiedMixtureKernelRiskGuard",
    "build_kernel_risk_trace_artifact",
    "build_kernel_risk_record",
    "build_mixture_kernel_risk_application",
    "kernel_risk_observations_from_preflight_trace",
    "load_frozen_kernel_risk_threshold",
    "mixture_kernel_risk_v3_profile",
    "mixture_kernel_risk_v3_profile_sha256",
    "to_materializer_guard",
    "verified_mixture_kernel_materializer_guard_projection",
    "verified_mixture_kernel_risk_guard_projection",
]
