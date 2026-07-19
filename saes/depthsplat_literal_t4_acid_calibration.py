"""Frozen ACID-disjoint calibration for the literal DepthSplat T=4 route.

This module is intentionally separate from the adaptive 15-anchor V15D/V16D
records.  The literal paper profile uses four corner probes at both compact
levels, raw probe-feature variance, and conditional depth probing.  Its risk
threshold is therefore collected and frozen from a new ACID 24/8 run rather
than inherited from an engineering profile.

The module is CPU-side only: it validates and freezes records emitted by a
native collector, but never loads a model, opens targets, invokes a renderer,
or starts CUDA work.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping as MappingABC
from dataclasses import dataclass, field
import hashlib
import json
import math
import struct
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from saes.depthsplat_acid_disjoint_calibration import (
    DEFAULT_MATERIALIZATION_ROOT,
    DEFAULT_PLAN_PATH,
    canonical_sha256,
    resolve_depthsplat_acid_binding,
)
from saes.probe_first_schedule import (
    LITERAL_PAPER_T4_PLAN_CONTRACT,
    PAPER_KP_ANCHOR_SEMANTICS,
    literal_paper_t4_route_config_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "1.0"
TRAIN_SPLIT = "calibration_train"
HOLDOUT_SPLIT = "calibration_holdout"
SPLITS = (TRAIN_SPLIT, HOLDOUT_SPLIT)

V16T4_KIND = "depthsplat-nonzero-l0-l1-acid-disjoint-v16l-t4"
FROZEN_STATUS = "FROZEN_EVALUATION_DISJOINT_TARGET_FREE"

LITERAL_T4_PROFILE_ID = "depthsplat-literal-paper-t4-probe-only-v1"
LITERAL_T4_MATERIALIZATION_PROFILE = (
    "depthsplat-literal-paper-t4-selected-probe-moment-v1"
)
LITERAL_T4_RISK_METRIC = "maximum-held-out-anchor-risk-v1"
LITERAL_T4_THRESHOLD_RULE = "train-minimum-per-scene-q25"
LITERAL_T4_GUARD_SCHEMA = "depthsplat-selected-anchor-attribute-loo-frozen-v16-guard-v1"
LITERAL_T4_PROFILE_SCHEMA = "depthsplat-literal-paper-t4-v16-profile-v1"
LITERAL_T4_COVERAGE_CERTIFICATE = (
    "depthsplat-selected-z-depth-3d-2sigma-ellipsoid-support-v2"
)
LITERAL_T4_LOO_CERTIFICATE = "depthsplat-selected-rgb-sh-opacity-all-anchor-loo-v1"
LITERAL_T4_LOO_POLICY = "all-retained-l0-l1-anchors-selected-labels-only-v1"
LITERAL_T4_LOO_AGGREGATE_SCHEMA = "depthsplat-selected-anchor-attribute-loo-aggregate-v1"
LITERAL_T4_TRACE_ARTIFACT_SCHEMA = "depthsplat-literal-t4-preflight-trace-artifact-v1"
LITERAL_T4_TRACE_ARTIFACT_KIND = "depthsplat-literal-t4-selected-anchor-loo-trace"

_ACCESS_KEYS = (
    "target_mapping_present",
    "target_rgb_accessed",
    "target_camera_metadata_accessed",
    "target_index_accessed",
    "skipped_s3_attributes_accessed",
)
_SHA256_CHARS = frozenset("0123456789abcdef")
# Native trace scalars are float32 ``.item()`` values serialized through JSON.
# Two ULPs cover round-trip representation and the float32 q75 interpolation,
# without admitting a distinct calibration risk.
_FLOAT32_SERIALIZATION_ULPS = 2.0
_FLOAT32_RELATIVE_TOLERANCE = _FLOAT32_SERIALIZATION_ULPS * 2.0**-23
_FLOAT32_ABSOLUTE_TOLERANCE = _FLOAT32_SERIALIZATION_ULPS * 2.0**-149
_COLLECTOR_SOURCE_CONTRACT = "depthsplat-literal-t4-acid-v16-collector-source-v1"
_COLLECTOR_SOURCE_FILES = (
    "saes/depthsplat_literal_t4_acid_calibration.py",
    "saes/depthsplat_literal_t4_acid_collector.py",
    "scripts/saes_depthsplat_literal_t4_acid_v16_calibration.py",
    "saes/depthsplat_acid_disjoint_calibration.py",
    "saes/depthsplat_backend.py",
    "saes/depthsplat_l0_l1_materializer.py",
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


# A plain projection is deliberately insufficient to authorize a runtime
# threshold.  Only this module can issue the opaque guard after its live
# record reload has completed.
_VERIFIED_LITERAL_T4_GUARD_ISSUER = object()


@dataclass(frozen=True, eq=False, init=False)
class VerifiedLiteralT4MaterializerGuard(MappingABC[str, Any]):
    """Immutable runtime projection issued after a verified V16T4 reload."""

    _projection: Mapping[str, Any] = field(repr=False)
    _issuer: object = field(repr=False, compare=False)

    def __init__(
        self,
        projection: Mapping[str, Any],
        *,
        _issuer: object | None = None,
    ) -> None:
        if _issuer is not _VERIFIED_LITERAL_T4_GUARD_ISSUER:
            raise TypeError(
                "literal T=4 materializer guards must be issued by the verified loader"
            )
        if not isinstance(projection, MappingABC):
            raise TypeError("literal T=4 materializer guard projection must be a mapping")
        object.__setattr__(self, "_projection", MappingProxyType(dict(projection)))
        object.__setattr__(self, "_issuer", _issuer)

    def __getitem__(self, key: str) -> Any:
        return self._projection[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._projection)

    def __len__(self) -> int:
        return len(self._projection)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe copy for evidence serialization."""

        return dict(self._projection)


def _issue_verified_literal_t4_materializer_guard(
    projection: Mapping[str, Any],
) -> VerifiedLiteralT4MaterializerGuard:
    """Issue the internal capability after a full record validation."""

    return VerifiedLiteralT4MaterializerGuard(
        projection,
        _issuer=_VERIFIED_LITERAL_T4_GUARD_ISSUER,
    )


def verified_literal_t4_materializer_guard_projection(
    value: object,
) -> dict[str, Any]:
    """Return an issued guard projection; reject plain or forged mappings."""

    if not isinstance(value, VerifiedLiteralT4MaterializerGuard):
        raise TypeError(
            "literal T=4 materializer requires an authenticated V16T4 guard from "
            "to_materializer_guard"
        )
    if getattr(value, "_issuer", None) is not _VERIFIED_LITERAL_T4_GUARD_ISSUER:
        raise ValueError("literal T=4 materializer guard authentication changed")
    return value.as_dict()


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
        raise ValueError(f"literal T=4 calibration cannot read {Path(path).name}") from error
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
        raise ValueError("literal T=4 calibration requires a nonempty valid quantile")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) and value >= 0.0 for value in ordered):
        raise ValueError("literal T=4 calibration values must be finite and nonnegative")
    return ordered[round((len(ordered) - 1) * float(fraction))]


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("literal T=4 calibration summary requires values")
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


def _literal_route_events() -> dict[str, Any]:
    """Return the immutable routing configuration accepted by literal T=4."""

    return {
        "contract_version": LITERAL_PAPER_T4_PLAN_CONTRACT,
        "tile_size": 4,
        "feature_threshold": 0.2,
        "depth_threshold": 0.1,
        "decision_semantics": "paper-probe-feature-variance-first-hit",
        "feature_statistic": "raw-probe-mean-channel-variance",
        "l1_anchor_semantics": PAPER_KP_ANCHOR_SEMANTICS,
        "l0_anchor_count": 4,
        "l1_anchor_count": 4,
        "formal_paper_kp4": True,
        "depth_checked_after_l0_miss_only": True,
    }


def literal_t4_profile() -> dict[str, Any]:
    """Return the sole source-faithful profile accepted by V16T4 records."""

    route_events = _literal_route_events()
    return {
        "schema_version": LITERAL_T4_PROFILE_SCHEMA,
        "id": LITERAL_T4_PROFILE_ID,
        "route_plan_contract": LITERAL_PAPER_T4_PLAN_CONTRACT,
        "materialization_profile": LITERAL_T4_MATERIALIZATION_PROFILE,
        "tile_size": 4,
        "decision_semantics": "paper-probe-feature-variance-first-hit",
        "feature_statistic": "raw-probe-mean-channel-variance",
        "feature_threshold": 0.2,
        "depth_threshold": 0.1,
        "l0_anchor_semantics": PAPER_KP_ANCHOR_SEMANTICS,
        "l1_anchor_semantics": PAPER_KP_ANCHOR_SEMANTICS,
        "l0_anchor_count": 4,
        "l1_anchor_count": 4,
        "secondary_mask_required_empty": True,
        "depth_checked_after_l0_miss_only": True,
        "route_plan_config_sha256": literal_paper_t4_route_config_sha256(route_events),
    }


def literal_t4_profile_sha256() -> str:
    """Hash the complete immutable literal-profile payload."""

    return canonical_sha256(literal_t4_profile())


def _validate_literal_t4_profile(value: Any) -> dict[str, Any]:
    expected = literal_t4_profile()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ValueError("literal T=4 calibration profile changed")
    expected_route_hash = literal_paper_t4_route_config_sha256(_literal_route_events())
    if value.get("route_plan_config_sha256") != expected_route_hash:
        raise ValueError("literal T=4 calibration route configuration changed")
    return expected


def _validate_acid_binding(binding: Any) -> dict[str, Any]:
    """Validate the public ACID 24/8 binding shape without V15D/V16D state."""

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
    if not isinstance(binding, Mapping) or set(binding) != required:
        raise ValueError("literal T=4 ACID binding has an invalid schema")
    if (
        binding.get("schema_version") != SCHEMA_VERSION
        or binding.get("kind") != "saes-acid-context-only-calibration-binding"
        or binding.get("dataset") != "acid"
        or not isinstance(binding.get("protocol_id"), str)
        or not binding["protocol_id"]
        or binding.get("access") != _access_record()
    ):
        raise ValueError("literal T=4 ACID binding is not context-only")
    for field in (
        "plan_sha256",
        "plan_file_sha256",
        "materialization_record_sha256",
        "materialization_tree_sha256",
        "materialization_manifest_sha256",
    ):
        _require_sha256(binding.get(field), f"literal T=4 ACID {field}")
    splits = binding.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(SPLITS):
        raise ValueError("literal T=4 ACID splits are invalid")
    normalized_splits: dict[str, dict[str, Any]] = {}
    for split, expected_count in ((TRAIN_SPLIT, 24), (HOLDOUT_SPLIT, 8)):
        value = splits.get(split)
        required_split = {
            "scenes",
            "scene_count",
            "scene_set_sha256",
            "selection_sha256",
            "sidecar_tree_sha256",
            "sidecar_manifest_sha256",
            "input_provenance_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != required_split:
            raise ValueError(f"literal T=4 ACID {split} binding is invalid")
        scenes = value.get("scenes")
        if (
            not isinstance(scenes, list)
            or len(scenes) != expected_count
            or value.get("scene_count") != expected_count
            or len(set(scenes)) != expected_count
            or any(not isinstance(scene, str) or not scene for scene in scenes)
            or canonical_sha256(sorted(scenes)) != value.get("scene_set_sha256")
        ):
            raise ValueError(f"literal T=4 ACID {split} scenes are invalid")
        for field in required_split - {"scenes", "scene_count"}:
            _require_sha256(value.get(field), f"literal T=4 ACID {split} {field}")
        normalized_splits[split] = dict(value)
    if set(normalized_splits[TRAIN_SPLIT]["scenes"]) & set(
        normalized_splits[HOLDOUT_SPLIT]["scenes"]
    ):
        raise ValueError("literal T=4 ACID train and holdout scenes overlap")
    return {**dict(binding), "splits": normalized_splits, "access": _access_record()}


def _collector_source_identity(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    files: list[dict[str, str]] = []
    for relative_path in _COLLECTOR_SOURCE_FILES:
        path = (root / relative_path).resolve()
        if root not in path.parents or not path.is_file() or path.is_symlink():
            raise ValueError("literal T=4 collector source file is invalid")
        files.append({"path": relative_path, "sha256": _sha256_file(path)})
    return {
        "contract": _COLLECTOR_SOURCE_CONTRACT,
        "files": files,
        "tree_sha256": canonical_sha256(files),
    }


def _validate_application_shape(value: Any) -> dict[str, Any]:
    required = {"model", "dataset", "backend_identity", "collector_source"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("literal T=4 application has an invalid schema")
    if (
        value.get("model") != "depthsplat"
        or value.get("dataset") != "dl3dv"
        or not isinstance(value.get("backend_identity"), Mapping)
    ):
        raise ValueError("literal T=4 application is not DepthSplat/DL3DV")
    source = value.get("collector_source")
    if not isinstance(source, Mapping) or set(source) != {"contract", "files", "tree_sha256"}:
        raise ValueError("literal T=4 collector source has an invalid schema")
    if source.get("contract") != _COLLECTOR_SOURCE_CONTRACT:
        raise ValueError("literal T=4 collector source contract changed")
    files = source.get("files")
    if not isinstance(files, list) or len(files) != len(_COLLECTOR_SOURCE_FILES):
        raise ValueError("literal T=4 collector source files are invalid")
    for relative_path, entry in zip(_COLLECTOR_SOURCE_FILES, files):
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
            raise ValueError("literal T=4 collector source file is invalid")
        if entry.get("path") != relative_path:
            raise ValueError("literal T=4 collector source ordering changed")
        _require_sha256(entry.get("sha256"), "literal T=4 collector source")
    if source.get("tree_sha256") != canonical_sha256(files):
        raise ValueError("literal T=4 collector source tree changed")
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


def build_literal_t4_application(
    *, backend_identity: Mapping[str, Any], root: Path = ROOT
) -> dict[str, Any]:
    """Bind a V16T4 record to native DepthSplat and its collector sources."""

    if not isinstance(backend_identity, Mapping):
        raise TypeError("literal T=4 application requires a frozen backend identity")
    return _validate_application_shape(
        {
            "model": "depthsplat",
            "dataset": "dl3dv",
            "backend_identity": dict(backend_identity),
            "collector_source": _collector_source_identity(Path(root)),
        }
    )


def _validate_live_application(application: Any, *, root: Path) -> dict[str, Any]:
    """Recompute native code identity before exposing a frozen threshold."""

    normalized = _validate_application_shape(application)
    from saes.depthsplat_backend import (
        resolve_depthsplat_backend_contract,
        validate_frozen_depthsplat_backend_identity,
    )

    contract = resolve_depthsplat_backend_contract(Path(root))
    backend_identity = validate_frozen_depthsplat_backend_identity(
        contract, normalized["backend_identity"]
    )
    expected = build_literal_t4_application(
        backend_identity=backend_identity, root=Path(root)
    )
    if normalized != expected:
        raise ValueError("literal T=4 application identity changed")
    return expected


def _normalise_risks(value: Any, *, label: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise ValueError(f"literal T=4 {label} risks are invalid")
    normalized = [float(item) for item in value]
    if not all(math.isfinite(item) and item >= 0.0 for item in normalized):
        raise ValueError(f"literal T=4 {label} risks must be finite and nonnegative")
    return normalized


def _loo_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    normalized = _normalise_risks(values, label="LOO aggregate")
    ordered = sorted(normalized)
    if not ordered:
        return {
            "count": 0,
            "minimum": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "maximum": None,
        }

    def quantile(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "count": len(ordered),
        "minimum": quantile(0.0),
        "p25": quantile(0.25),
        "p50": quantile(0.50),
        "p75": quantile(0.75),
        "maximum": quantile(1.0),
    }


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"literal T=4 {label} is invalid")
    return int(value)


def _validate_selected_anchor_loo_aggregate(value: Any) -> dict[str, Any]:
    """Validate the complete native preflight aggregate used to freeze risks."""

    required = {
        "schema_version",
        "certificate",
        "policy",
        "risk_metric",
        "mode",
        "frozen_guard",
        "preflight_tile_trace_sha256",
        "candidate_tile_count",
        "recorded_tile_count",
        "tile_records_sha256",
        "scorable_tile_count",
        "unscorable_promoted_full_tile_count",
        "accepted_compact_tile_count",
        "guard_promoted_full_tile_count",
        "maximum_held_out_risks",
        "maximum_held_out_risk_summary",
        "q75_risks",
        "q75_risk_summary",
        "selected_anchor_native_attribute_label_reads",
        "selected_anchor_native_attribute_endpoint_reads",
        "source_nonprobe_s3_attribute_reads",
        "nonzero_direct_deletion",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("literal T=4 selected-anchor LOO aggregate has an invalid schema")
    if (
        value.get("schema_version") != LITERAL_T4_LOO_AGGREGATE_SCHEMA
        or value.get("certificate") != LITERAL_T4_LOO_CERTIFICATE
        or value.get("policy") != LITERAL_T4_LOO_POLICY
        or value.get("risk_metric") != LITERAL_T4_RISK_METRIC
        or value.get("mode") != "collect_only"
        or value.get("frozen_guard") is not None
        or value.get("source_nonprobe_s3_attribute_reads") != 0
        or value.get("nonzero_direct_deletion") is not False
    ):
        raise ValueError("literal T=4 selected-anchor LOO aggregate crossed its collection contract")
    for field in ("preflight_tile_trace_sha256", "tile_records_sha256"):
        _require_sha256(value.get(field), f"literal T=4 LOO {field}")
    counts = {
        name: _nonnegative_int(value.get(name), label=f"LOO {name}")
        for name in (
            "candidate_tile_count",
            "recorded_tile_count",
            "scorable_tile_count",
            "unscorable_promoted_full_tile_count",
            "accepted_compact_tile_count",
            "guard_promoted_full_tile_count",
            "selected_anchor_native_attribute_label_reads",
            "selected_anchor_native_attribute_endpoint_reads",
        )
    }
    maximum_risks = _normalise_risks(
        value.get("maximum_held_out_risks"), label="LOO maximum-held-out"
    )
    q75_risks = _normalise_risks(value.get("q75_risks"), label="LOO q75")
    if (
        counts["candidate_tile_count"] != counts["recorded_tile_count"]
        or counts["scorable_tile_count"] + counts["unscorable_promoted_full_tile_count"]
        != counts["recorded_tile_count"]
        or len(maximum_risks) != counts["scorable_tile_count"]
        or len(q75_risks) != counts["scorable_tile_count"]
        or counts["scorable_tile_count"] < 1
        or counts["accepted_compact_tile_count"] < 1
        or counts["accepted_compact_tile_count"] > counts["candidate_tile_count"]
        or counts["guard_promoted_full_tile_count"] != 0
        or counts["selected_anchor_native_attribute_label_reads"]
        < counts["scorable_tile_count"]
        or counts["selected_anchor_native_attribute_endpoint_reads"]
        < counts["unscorable_promoted_full_tile_count"]
        or any(q75 > maximum for q75, maximum in zip(q75_risks, maximum_risks))
        or value.get("maximum_held_out_risk_summary") != _loo_summary(maximum_risks)
        or value.get("q75_risk_summary") != _loo_summary(q75_risks)
    ):
        raise ValueError("literal T=4 selected-anchor LOO aggregate counts or risks changed")
    return dict(value)


def _trace_reference(value: Any) -> dict[str, str]:
    """Validate a portable record-relative trace artifact reference."""

    if not isinstance(value, Mapping) or set(value) != {
        "relative_path",
        "sha256",
        "preflight_tile_trace_sha256",
        "tile_records_sha256",
    }:
        raise ValueError("literal T=4 trace artifact reference has an invalid schema")
    relative_path = value.get("relative_path")
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        raise ValueError("literal T=4 trace artifact path is invalid")
    parsed = PurePosixPath(relative_path)
    if (
        parsed.is_absolute()
        or not parsed.parts
        or any(part in {"", ".", ".."} for part in parsed.parts)
        or parsed.as_posix() != relative_path
    ):
        raise ValueError("literal T=4 trace artifact path is not safely relative")
    return {
        "relative_path": relative_path,
        "sha256": _require_sha256(value.get("sha256"), "literal T=4 trace artifact"),
        "preflight_tile_trace_sha256": _require_sha256(
            value.get("preflight_tile_trace_sha256"),
            "literal T=4 trace artifact preflight tile trace",
        ),
        "tile_records_sha256": _require_sha256(
            value.get("tile_records_sha256"),
            "literal T=4 trace artifact tile records",
        ),
    }


def _float32(value: float, *, label: str) -> float:
    """Round one serialized scalar as the native float32 trace did."""

    try:
        rounded = struct.unpack("!f", struct.pack("!f", float(value)))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError(f"literal T=4 {label} is not representable as float32") from error
    if not math.isfinite(rounded) or rounded < 0.0:
        raise ValueError(f"literal T=4 {label} is not a finite nonnegative float32")
    return float(rounded)


def _float32_close(actual: float, expected: float) -> bool:
    """Allow only a small documented number of float32 serialization ULPs."""

    return math.isclose(
        float(actual),
        float(expected),
        rel_tol=_FLOAT32_RELATIVE_TOLERANCE,
        abs_tol=_FLOAT32_ABSOLUTE_TOLERANCE,
    )


def _materializer_q75_float32(values: Sequence[float]) -> float:
    """Mirror ``torch.quantile(float32, 0.75)`` for four held-out risks."""

    if len(values) != 4:
        raise ValueError("literal T=4 q75 requires exactly four held-out risks")
    ordered = sorted(_float32(value, label="held-out risk") for value in values)
    lower = ordered[2]
    upper = ordered[3]
    # Torch's default linear interpolation is evaluated in the input dtype.
    delta = _float32(upper - lower, label="q75 interpolation delta")
    quarter_delta = _float32(delta * 0.25, label="q75 interpolation product")
    return _float32(lower + quarter_delta, label="q75 interpolation result")


def _trace_loo_observation(value: Any, *, planned_route: str) -> tuple[str, float | None, float | None, int, int]:
    """Validate one selected-anchor observation in a persisted preflight trace."""

    required = {
        "checked",
        "scorable",
        "status",
        "reason",
        "certificate",
        "policy",
        "risk_metric",
        "level",
        "anchor_count",
        "held_out_anchor_count",
        "held_out_anchor_records",
        "q75_risk",
        "maximum_held_out_risk",
        "selected_anchor_native_attribute_label_reads",
        "selected_anchor_native_attribute_endpoint_reads",
        "source_nonprobe_s3_attribute_reads",
        "nonzero_direct_deletion",
        "maximum_allowed_risk",
        "passed",
        "action",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("literal T=4 trace selected-anchor observation has an invalid schema")
    if (
        value.get("certificate") != LITERAL_T4_LOO_CERTIFICATE
        or value.get("policy") != LITERAL_T4_LOO_POLICY
        or value.get("risk_metric") != LITERAL_T4_RISK_METRIC
        or value.get("level") != planned_route
        or value.get("anchor_count") != 4
        or value.get("source_nonprobe_s3_attribute_reads") != 0
        or value.get("nonzero_direct_deletion") is not False
        or value.get("maximum_allowed_risk") is not None
        or value.get("passed") is not None
    ):
        raise ValueError("literal T=4 trace selected-anchor observation changed")
    if value.get("scorable") is True:
        records = value.get("held_out_anchor_records")
        count = value.get("held_out_anchor_count")
        if (
            value.get("checked") is not True
            or value.get("status") != "scored"
            or value.get("reason") is not None
            or value.get("action") != "observed_only"
            or count != 4
            or not isinstance(records, list)
            or len(records) != count
            or value.get("selected_anchor_native_attribute_label_reads") != count
            or value.get("selected_anchor_native_attribute_endpoint_reads") != 0
        ):
            raise ValueError("literal T=4 trace scored selected-anchor observation is invalid")
        held_out_risks: list[float] = []
        positions: set[tuple[int, int]] = set()
        for record in records:
            if not isinstance(record, Mapping) or set(record) != {
                "held_out_local_position",
                "harmonic_relative_error",
                "opacity_logit_relative_error",
                "risk",
            }:
                raise ValueError("literal T=4 trace held-out anchor record is invalid")
            position = record.get("held_out_local_position")
            if (
                not isinstance(position, list)
                or len(position) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) for item in position)
                or any(item < 0 or item >= 4 for item in position)
            ):
                raise ValueError("literal T=4 trace held-out anchor position is invalid")
            positions.add((position[0], position[1]))
            values = _normalise_risks(
                [
                    record.get("harmonic_relative_error"),
                    record.get("opacity_logit_relative_error"),
                    record.get("risk"),
                ],
                label="trace held-out anchor",
            )
            harmonic_error = _float32(values[0], label="held-out harmonic error")
            opacity_error = _float32(values[1], label="held-out opacity error")
            expected_risk = max(harmonic_error, opacity_error)
            if not _float32_close(values[2], expected_risk):
                raise ValueError(
                    "literal T=4 trace held-out risk is not the maximum attribute error"
                )
            held_out_risks.append(expected_risk)
        if positions != {(0, 0), (0, 3), (3, 0), (3, 3)}:
            raise ValueError("literal T=4 trace held-out anchors are not the four corners")
        maximum = max(held_out_risks)
        expected_q75 = _materializer_q75_float32(held_out_risks)
        q75 = value.get("q75_risk")
        reported_maximum = value.get("maximum_held_out_risk")
        if (
            isinstance(q75, bool)
            or not isinstance(q75, (int, float))
            or not math.isfinite(float(q75))
            or float(q75) < 0.0
            or isinstance(reported_maximum, bool)
            or not isinstance(reported_maximum, (int, float))
            or not _float32_close(float(reported_maximum), maximum)
            or not _float32_close(float(q75), expected_q75)
        ):
            raise ValueError("literal T=4 trace selected-anchor risks changed")
        # The persisted values may differ by a JSON float round-trip within the
        # accepted tolerance.  Freeze and reload only the recomputed float32
        # values so that a tolerated serialization difference cannot alter a
        # train-only threshold.
        return "scorable", maximum, expected_q75, int(count), 0

    endpoint_reads = value.get("selected_anchor_native_attribute_endpoint_reads")
    if (
        value.get("scorable") is not False
        or value.get("checked") is not False
        or value.get("status") != "native-opacity-endpoint-promoted-full-v1"
        or value.get("reason") != "native_opacity_endpoint_requires_full"
        or value.get("action") != "promote_full_unscorable"
        or value.get("held_out_anchor_count") != 0
        or value.get("held_out_anchor_records") != []
        or value.get("q75_risk") is not None
        or value.get("maximum_held_out_risk") is not None
        or value.get("selected_anchor_native_attribute_label_reads") != 0
        or isinstance(endpoint_reads, bool)
        or not isinstance(endpoint_reads, int)
        or endpoint_reads < 1
    ):
        raise ValueError("literal T=4 trace unscorable selected-anchor observation is invalid")
    return "unscorable", None, None, 0, int(endpoint_reads)


def _aggregate_from_preflight_tile_trace(value: Any) -> dict[str, Any]:
    """Rebuild the native collect-only aggregate from its full tile trace."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("literal T=4 preflight tile trace is invalid")
    trace: list[dict[str, Any]] = []
    tile_records: list[dict[str, Any]] = []
    tile_ids: set[tuple[int, int, int]] = set()
    maximum_risks: list[float] = []
    q75_risks: list[float] = []
    label_reads = 0
    endpoint_reads = 0
    accepted_count = 0
    unscorable_count = 0
    for index, raw_entry in enumerate(value):
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"literal T=4 preflight tile {index} is invalid")
        entry = dict(raw_entry)
        required = {
            "view",
            "tile_y",
            "tile_x",
            "planned_route",
            "attempted",
            "accepted",
            "reason",
            "source_nonprobe_s3_attribute_reads",
            "selected_anchor_attribute_loo",
        }
        if not required.issubset(entry):
            raise ValueError("literal T=4 preflight tile has no LOO provenance")
        coordinates = (entry.get("view"), entry.get("tile_y"), entry.get("tile_x"))
        if (
            any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in coordinates)
            or coordinates in tile_ids
            or entry.get("planned_route") not in {"L0", "L1", "Full"}
            or not isinstance(entry.get("attempted"), bool)
            or entry["attempted"] != (entry["planned_route"] in {"L0", "L1"})
            or not isinstance(entry.get("accepted"), bool)
            or not isinstance(entry.get("reason"), str)
            or entry.get("source_nonprobe_s3_attribute_reads") != 0
        ):
            raise ValueError("literal T=4 preflight tile identity changed")
        tile_ids.add((int(coordinates[0]), int(coordinates[1]), int(coordinates[2])))
        if not entry["attempted"]:
            if entry["selected_anchor_attribute_loo"] is not None:
                raise ValueError("literal T=4 Full tile unexpectedly has LOO evidence")
            trace.append(entry)
            continue
        loo = entry["selected_anchor_attribute_loo"]
        kind, maximum, q75, labels, endpoints = _trace_loo_observation(
            loo, planned_route=str(entry["planned_route"])
        )
        record = {
            "view": int(coordinates[0]),
            "tile_y": int(coordinates[1]),
            "tile_x": int(coordinates[2]),
            "planned_route": entry["planned_route"],
            "accepted": entry["accepted"],
            "reason": entry["reason"],
            "selected_anchor_attribute_loo": dict(loo),
        }
        tile_records.append(record)
        accepted_count += int(entry["accepted"])
        label_reads += labels
        endpoint_reads += endpoints
        if kind == "scorable":
            if maximum is None or q75 is None:
                raise RuntimeError("literal T=4 trace scorer returned no risk")
            maximum_risks.append(maximum)
            q75_risks.append(q75)
        else:
            unscorable_count += 1
        trace.append(entry)
    aggregate = {
        "schema_version": LITERAL_T4_LOO_AGGREGATE_SCHEMA,
        "certificate": LITERAL_T4_LOO_CERTIFICATE,
        "policy": LITERAL_T4_LOO_POLICY,
        "risk_metric": LITERAL_T4_RISK_METRIC,
        "mode": "collect_only",
        "frozen_guard": None,
        "preflight_tile_trace_sha256": canonical_sha256(trace),
        "candidate_tile_count": len(tile_records),
        "recorded_tile_count": len(tile_records),
        "tile_records_sha256": canonical_sha256(tile_records),
        "scorable_tile_count": len(maximum_risks),
        "unscorable_promoted_full_tile_count": unscorable_count,
        "accepted_compact_tile_count": accepted_count,
        "guard_promoted_full_tile_count": 0,
        "maximum_held_out_risks": maximum_risks,
        "maximum_held_out_risk_summary": _loo_summary(maximum_risks),
        "q75_risks": q75_risks,
        "q75_risk_summary": _loo_summary(q75_risks),
        "selected_anchor_native_attribute_label_reads": label_reads,
        "selected_anchor_native_attribute_endpoint_reads": endpoint_reads,
        "source_nonprobe_s3_attribute_reads": 0,
        "nonzero_direct_deletion": False,
    }
    return _validate_selected_anchor_loo_aggregate(aggregate)


def build_v16t4_trace_artifact(
    *,
    scene: str,
    split: str,
    preflight_tile_trace: Sequence[Mapping[str, Any]],
    selected_anchor_loo_aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a self-hashed, target-free trace artifact for one ACID scene."""

    if not isinstance(scene, str) or not scene or split not in SPLITS:
        raise ValueError("literal T=4 trace artifact scene identity is invalid")
    rebuilt = _aggregate_from_preflight_tile_trace(preflight_tile_trace)
    aggregate = _validate_selected_anchor_loo_aggregate(selected_anchor_loo_aggregate)
    if aggregate != rebuilt:
        raise ValueError("literal T=4 trace artifact aggregate does not match its tile trace")
    payload = {
        "schema_version": LITERAL_T4_TRACE_ARTIFACT_SCHEMA,
        "kind": LITERAL_T4_TRACE_ARTIFACT_KIND,
        "scene": scene,
        "split": split,
        "preflight_tile_trace": [dict(entry) for entry in preflight_tile_trace],
        "selected_anchor_loo_aggregate": aggregate,
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def _load_trace_artifact(
    reference: Mapping[str, Any],
    *,
    record_directory: Path,
    scene: str,
    split: str,
    evidence: Mapping[str, Any],
    risks: Sequence[float],
) -> dict[str, Any]:
    """Reopen and verify the persisted tile trace before a freeze is accepted."""

    normalized_reference = _trace_reference(reference)
    record_directory = Path(record_directory).resolve()
    artifact_path = record_directory
    for component in PurePosixPath(normalized_reference["relative_path"]).parts:
        artifact_path = artifact_path / component
        if artifact_path.is_symlink():
            raise ValueError("literal T=4 trace artifact path contains a symlink")
    try:
        resolved_artifact = artifact_path.resolve(strict=True)
    except OSError as error:
        raise ValueError("literal T=4 trace artifact is unavailable") from error
    if (
        not resolved_artifact.is_file()
        or record_directory not in resolved_artifact.parents
    ):
        raise ValueError("literal T=4 trace artifact escapes its record directory")
    artifact = _read_json(resolved_artifact, "literal T=4 trace artifact")
    recorded_sha256 = artifact.pop("sha256", None)
    if (
        recorded_sha256 != canonical_sha256(artifact)
        or recorded_sha256 != normalized_reference["sha256"]
    ):
        raise ValueError("literal T=4 trace artifact SHA256 is invalid")
    required = {
        "schema_version",
        "kind",
        "scene",
        "split",
        "preflight_tile_trace",
        "selected_anchor_loo_aggregate",
    }
    if set(artifact) != required:
        raise ValueError("literal T=4 trace artifact has unexpected fields")
    if (
        artifact.get("schema_version") != LITERAL_T4_TRACE_ARTIFACT_SCHEMA
        or artifact.get("kind") != LITERAL_T4_TRACE_ARTIFACT_KIND
        or artifact.get("scene") != scene
        or artifact.get("split") != split
    ):
        raise ValueError("literal T=4 trace artifact identity changed")
    rebuilt = _aggregate_from_preflight_tile_trace(artifact.get("preflight_tile_trace"))
    aggregate = _validate_selected_anchor_loo_aggregate(
        artifact.get("selected_anchor_loo_aggregate")
    )
    if (
        aggregate != rebuilt
        or aggregate != evidence.get("selected_anchor_loo_aggregate")
        or evidence.get("selected_anchor_loo_aggregate_sha256")
        != canonical_sha256(aggregate)
        or evidence.get("selected_anchor_loo_trace_sha256")
        != aggregate["preflight_tile_trace_sha256"]
        or normalized_reference["preflight_tile_trace_sha256"]
        != aggregate["preflight_tile_trace_sha256"]
        or normalized_reference["tile_records_sha256"]
        != aggregate["tile_records_sha256"]
        or list(risks) != aggregate["maximum_held_out_risks"]
    ):
        raise ValueError("literal T=4 trace artifact is not linked to its LOO aggregate")
    return {**artifact, "sha256": recorded_sha256}


def _validate_scene_evidence(
    value: Any, *, profile_sha256: str, risks: Sequence[float]
) -> dict[str, Any]:
    """Require source-bound, target-free evidence for one native collection."""

    required = {
        "profile_sha256",
        "route_plan_config_sha256",
        "native_execution_sha256",
        "initial_attribute_binding_sha256",
        "final_selected_attribute_binding_sha256",
        "materialized_attribute_binding_sha256",
        "full_passthrough_mask_sha256",
        "full_attribute_binding_sha256",
        "full_attributes_bitwise_native",
        "coverage_certificate",
        "coverage_certificate_sha256",
        "accepted_update_slots_sha256",
        "accepted_update_slot_count",
        "selected_anchor_loo_aggregate",
        "selected_anchor_loo_aggregate_sha256",
        "selected_anchor_loo_trace_sha256",
        "trace_artifact",
        "source_nonprobe_s3_attribute_reads",
        "nonzero_merge_applied",
        "renderer_executed",
        "quality_metrics_computed",
        "whole_pipeline_s2_s3_sparse_execution_verified",
        "global_s2_s3_savings_claimed",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("literal T=4 scene evidence has an invalid schema")
    if (
        value.get("profile_sha256") != profile_sha256
        or value.get("route_plan_config_sha256")
        != literal_t4_profile()["route_plan_config_sha256"]
        or value.get("coverage_certificate") != LITERAL_T4_COVERAGE_CERTIFICATE
        or value.get("full_attributes_bitwise_native") is not True
        or value.get("nonzero_merge_applied") is not True
        or value.get("renderer_executed") is not False
        or value.get("quality_metrics_computed") is not False
        or value.get("whole_pipeline_s2_s3_sparse_execution_verified") is not False
        or value.get("global_s2_s3_savings_claimed") is not False
        or value.get("source_nonprobe_s3_attribute_reads") != 0
    ):
        raise ValueError("literal T=4 scene evidence crossed its source-only contract")
    for field in (
        "native_execution_sha256",
        "initial_attribute_binding_sha256",
        "final_selected_attribute_binding_sha256",
        "materialized_attribute_binding_sha256",
        "full_passthrough_mask_sha256",
        "full_attribute_binding_sha256",
        "coverage_certificate_sha256",
        "accepted_update_slots_sha256",
        "selected_anchor_loo_aggregate_sha256",
        "selected_anchor_loo_trace_sha256",
    ):
        _require_sha256(value.get(field), f"literal T=4 scene {field}")
    for field, minimum in (("accepted_update_slot_count", 1),):
        raw = value.get(field)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < minimum:
            raise ValueError(f"literal T=4 scene {field} is invalid")
    aggregate = _validate_selected_anchor_loo_aggregate(
        value.get("selected_anchor_loo_aggregate")
    )
    trace_artifact = _trace_reference(value.get("trace_artifact"))
    if (
        value.get("selected_anchor_loo_aggregate_sha256") != canonical_sha256(aggregate)
        or value.get("selected_anchor_loo_trace_sha256")
        != aggregate["preflight_tile_trace_sha256"]
        or trace_artifact["preflight_tile_trace_sha256"]
        != aggregate["preflight_tile_trace_sha256"]
        or trace_artifact["tile_records_sha256"] != aggregate["tile_records_sha256"]
        or list(risks) != aggregate["maximum_held_out_risks"]
        or value.get("source_nonprobe_s3_attribute_reads")
        != aggregate["source_nonprobe_s3_attribute_reads"]
    ):
        raise ValueError("literal T=4 scene risks are not bound to its LOO aggregate")
    return dict(value)


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
        raise ValueError(f"literal T=4 {split} scene records are invalid")
    expected_scenes = list(binding["splits"][split]["scenes"])
    if len(records) != len(expected_scenes):
        raise ValueError(f"literal T=4 {split} scene count changed")
    normalized: list[dict[str, Any]] = []
    for position, (scene, record) in enumerate(zip(expected_scenes, records)):
        if (
            not isinstance(record, Mapping)
            or set(record) != {"scene", "maximum_held_out_risks", "evidence", "access"}
            or record.get("scene") != scene
            or record.get("access") != _access_record()
        ):
            raise ValueError(f"literal T=4 {split} scene record is invalid at {position}")
        risks = _normalise_risks(record.get("maximum_held_out_risks"), label=split)
        evidence = _validate_scene_evidence(
            record.get("evidence"), profile_sha256=profile_sha256, risks=risks
        )
        reference = _trace_reference(evidence["trace_artifact"])
        if seen_trace_artifacts is not None:
            if reference["relative_path"] in seen_trace_artifacts:
                raise ValueError("literal T=4 scene records reuse a trace artifact")
            seen_trace_artifacts.add(reference["relative_path"])
        if record_directory is not None:
            _load_trace_artifact(
                reference,
                record_directory=record_directory,
                scene=scene,
                split=split,
                evidence=evidence,
                risks=risks,
            )
        normalized.append(
            {
                "scene": scene,
                "maximum_held_out_risks": risks,
                "evidence": evidence,
                "access": _access_record(),
            }
        )
    return normalized


def _split_verification(
    records: Sequence[Mapping[str, Any]], *, threshold: float
) -> dict[str, Any]:
    values = [
        value
        for record in records
        for value in record["maximum_held_out_risks"]
    ]
    retained = sum(value <= threshold for value in values)
    return {
        "threshold_value": threshold,
        "candidate_count": len(values),
        "retained_count": retained,
        "promoted_count": len(values) - retained,
        "summary": _summary(values),
        "threshold_updated": False,
    }


def build_v16t4_record(
    *,
    binding: Mapping[str, Any],
    application: Mapping[str, Any],
    profile: Mapping[str, Any],
    train_scene_records: Sequence[Mapping[str, Any]],
    holdout_scene_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze the literal T=4 risk threshold from ACID train scenes only."""

    validated_binding = _validate_acid_binding(binding)
    validated_application = _validate_application_shape(application)
    validated_profile = _validate_literal_t4_profile(profile)
    profile_sha = literal_t4_profile_sha256()
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
    per_scene_q25 = [
        _finite_quantile(record["maximum_held_out_risks"], 0.25) for record in train
    ]
    threshold = min(per_scene_q25)
    train_values = [
        value for record in train for value in record["maximum_held_out_risks"]
    ]
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": V16T4_KIND,
        "status": FROZEN_STATUS,
        "paper_result_eligible": False,
        "application": validated_application,
        "acid_binding": validated_binding,
        "profile": validated_profile,
        "profile_sha256": profile_sha,
        "signal": {
            "name": LITERAL_T4_RISK_METRIC,
            "source": "target-free-selected-anchor-rgb-sh-opacity-leave-one-out",
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
            "risk_metric": LITERAL_T4_RISK_METRIC,
            "value": threshold,
            "promote_when": "maximum_held_out_risk_gt_value",
            "rule": LITERAL_T4_THRESHOLD_RULE,
            "per_scene_q25": per_scene_q25,
        },
    }
    return {**record, "sha256": canonical_sha256(record)}


def _read_record(path: Path) -> dict[str, Any]:
    record = _read_json(Path(path), "literal T=4 V16 record")
    recorded_sha = record.pop("sha256", None)
    if recorded_sha != canonical_sha256(record):
        raise ValueError("literal T=4 V16 record SHA256 is invalid")
    if record.get("kind") != V16T4_KIND or record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("literal T=4 V16 record kind or schema is invalid")
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
        raise ValueError("literal T=4 V16 record has unexpected fields")
    if (
        record.get("kind") != V16T4_KIND
        or record.get("schema_version") != SCHEMA_VERSION
        or record.get("status") != FROZEN_STATUS
        or record.get("paper_result_eligible") is not False
        or record.get("access") != _access_record()
        or record.get("profile_sha256") != literal_t4_profile_sha256()
        or record.get("signal")
        != {
            "name": LITERAL_T4_RISK_METRIC,
            "source": "target-free-selected-anchor-rgb-sh-opacity-leave-one-out",
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
        raise ValueError("literal T=4 V16 record is not a frozen target-free calibration")
    profile = _validate_literal_t4_profile(record.get("profile"))
    application = _validate_live_application(record.get("application"), root=Path(root))
    if record.get("application") != application:
        raise ValueError("literal T=4 V16 application identity changed")
    live_binding = _validate_acid_binding(
        resolve_depthsplat_acid_binding(
            plan_path=Path(plan_path), materialization_root=Path(materialization_root)
        )
    )
    if record.get("acid_binding") != live_binding:
        raise ValueError("literal T=4 V16 ACID 24/8 binding changed")
    seen_trace_artifacts: set[str] = set()
    train = _normalize_scene_records(
        record.get("train_scene_records"),
        binding=live_binding,
        split=TRAIN_SPLIT,
        profile_sha256=literal_t4_profile_sha256(),
        record_directory=record_directory,
        seen_trace_artifacts=seen_trace_artifacts,
    )
    holdout = _normalize_scene_records(
        record.get("holdout_scene_records"),
        binding=live_binding,
        split=HOLDOUT_SPLIT,
        profile_sha256=literal_t4_profile_sha256(),
        record_directory=record_directory,
        seen_trace_artifacts=seen_trace_artifacts,
    )
    per_scene_q25 = [
        _finite_quantile(scene["maximum_held_out_risks"], 0.25) for scene in train
    ]
    expected_threshold = min(per_scene_q25)
    expected_threshold_record = {
        "risk_metric": LITERAL_T4_RISK_METRIC,
        "value": expected_threshold,
        "promote_when": "maximum_held_out_risk_gt_value",
        "rule": LITERAL_T4_THRESHOLD_RULE,
        "per_scene_q25": per_scene_q25,
    }
    if record.get("threshold") != expected_threshold_record:
        raise ValueError("literal T=4 V16 threshold is not reproducible from train-only risks")
    train_values = [
        value for scene in train for value in scene["maximum_held_out_risks"]
    ]
    if record.get("train_summary") != _summary(train_values):
        raise ValueError("literal T=4 V16 train summary changed")
    if record.get("holdout_verification") != _split_verification(
        holdout, threshold=expected_threshold
    ):
        raise ValueError("literal T=4 V16 holdout verification changed")
    return {
        **dict(record),
        "profile": profile,
        "application": application,
        "acid_binding": live_binding,
        "threshold_value": float(expected_threshold),
    }


def load_frozen_v16t4_threshold(
    path: Path,
    *,
    root: Path = ROOT,
    plan_path: Path = DEFAULT_PLAN_PATH,
    materialization_root: Path = DEFAULT_MATERIALIZATION_ROOT,
) -> dict[str, Any]:
    """Load V16T4 only when its live native and ACID identities still match."""

    record_path = Path(path)
    if record_path.is_symlink():
        raise ValueError("literal T=4 V16 record must not be a symlink")
    try:
        resolved_record_path = record_path.resolve(strict=True)
    except OSError as error:
        raise ValueError("literal T=4 V16 record is unavailable") from error
    if not resolved_record_path.is_file():
        raise ValueError("literal T=4 V16 record is invalid")
    return _validate_frozen_record(
        _read_record(resolved_record_path),
        root=Path(root),
        plan_path=Path(plan_path),
        materialization_root=Path(materialization_root),
        record_directory=resolved_record_path.parent,
    )


def _to_materializer_guard(
    record: Mapping[str, Any],
) -> VerifiedLiteralT4MaterializerGuard:
    """Project a record already verified by the path-based live loader."""

    if not isinstance(record, Mapping):
        raise TypeError("literal T=4 materializer guard requires a verified record")
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
    if set(record) not in (required, required | {"threshold_value"}):
        raise ValueError("literal T=4 materializer guard record has unexpected fields")
    payload = {
        key: value
        for key, value in record.items()
        if key not in {"sha256", "threshold_value"}
    }
    if record.get("sha256") != canonical_sha256(payload):
        raise ValueError("literal T=4 materializer guard record SHA256 is invalid")
    if (
        record.get("kind") != V16T4_KIND
        or record.get("status") != FROZEN_STATUS
        or record.get("paper_result_eligible") is not False
        or record.get("access") != _access_record()
        or record.get("profile_sha256") != literal_t4_profile_sha256()
        or record.get("signal")
        != {
            "name": LITERAL_T4_RISK_METRIC,
            "source": "target-free-selected-anchor-rgb-sh-opacity-leave-one-out",
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
        raise ValueError("literal T=4 materializer guard record identity changed")
    profile = _validate_literal_t4_profile(record.get("profile"))
    threshold = record.get("threshold")
    if not isinstance(threshold, Mapping) or set(threshold) != {
        "risk_metric",
        "value",
        "promote_when",
        "rule",
        "per_scene_q25",
    }:
        raise ValueError("literal T=4 materializer guard threshold is invalid")
    value = threshold.get("value")
    if (
        threshold.get("risk_metric") != LITERAL_T4_RISK_METRIC
        or threshold.get("rule") != LITERAL_T4_THRESHOLD_RULE
        or threshold.get("promote_when") != "maximum_held_out_risk_gt_value"
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError("literal T=4 materializer guard threshold changed")
    application = _validate_application_shape(record.get("application"))
    binding = _validate_acid_binding(record.get("acid_binding"))
    train = _normalize_scene_records(
        record.get("train_scene_records"),
        binding=binding,
        split=TRAIN_SPLIT,
        profile_sha256=literal_t4_profile_sha256(),
    )
    holdout = _normalize_scene_records(
        record.get("holdout_scene_records"),
        binding=binding,
        split=HOLDOUT_SPLIT,
        profile_sha256=literal_t4_profile_sha256(),
    )
    per_scene_q25 = [
        _finite_quantile(scene["maximum_held_out_risks"], 0.25) for scene in train
    ]
    expected_threshold = min(per_scene_q25)
    if threshold != {
        "risk_metric": LITERAL_T4_RISK_METRIC,
        "value": expected_threshold,
        "promote_when": "maximum_held_out_risk_gt_value",
        "rule": LITERAL_T4_THRESHOLD_RULE,
        "per_scene_q25": per_scene_q25,
    }:
        raise ValueError("literal T=4 materializer guard threshold is not train-only")
    train_values = [
        value for scene in train for value in scene["maximum_held_out_risks"]
    ]
    if record.get("train_summary") != _summary(train_values):
        raise ValueError("literal T=4 materializer guard train summary changed")
    if record.get("holdout_verification") != _split_verification(
        holdout, threshold=expected_threshold
    ):
        raise ValueError("literal T=4 materializer guard holdout verification changed")
    if "threshold_value" in record and record["threshold_value"] != float(expected_threshold):
        raise ValueError("literal T=4 materializer guard loaded threshold changed")
    _require_sha256(record.get("sha256"), "literal T=4 frozen record")
    projection = {
        "schema_version": LITERAL_T4_GUARD_SCHEMA,
        "frozen_record_kind": V16T4_KIND,
        "frozen_record_sha256": record["sha256"],
        "threshold_value": float(value),
        "threshold_rule": LITERAL_T4_THRESHOLD_RULE,
        "risk_metric": LITERAL_T4_RISK_METRIC,
        "materialization_profile": profile["materialization_profile"],
        "route_plan_contract": profile["route_plan_contract"],
        "route_plan_config_sha256": profile["route_plan_config_sha256"],
        "acid_binding_sha256": canonical_sha256(binding),
        "application_sha256": canonical_sha256(application),
    }
    return _issue_verified_literal_t4_materializer_guard(projection)


def to_materializer_guard(
    path: Path,
    *,
    root: Path = ROOT,
    plan_path: Path = DEFAULT_PLAN_PATH,
    materialization_root: Path = DEFAULT_MATERIALIZATION_ROOT,
) -> VerifiedLiteralT4MaterializerGuard:
    """Load, fully revalidate, and project a frozen V16T4 record by path.

    A guard is deliberately not projected from an arbitrary mapping.  This
    keeps the record's self hash, live application/ACID bindings, and the
    persisted per-scene trace artifacts on the only public loading path.
    """

    if isinstance(path, Mapping) or not isinstance(path, (str, Path)):
        raise TypeError("literal T=4 materializer guard requires a record path")
    record = load_frozen_v16t4_threshold(
        Path(path),
        root=Path(root),
        plan_path=Path(plan_path),
        materialization_root=Path(materialization_root),
    )
    return _to_materializer_guard(record)


__all__ = [
    "DEFAULT_MATERIALIZATION_ROOT",
    "DEFAULT_PLAN_PATH",
    "FROZEN_STATUS",
    "HOLDOUT_SPLIT",
    "LITERAL_T4_GUARD_SCHEMA",
    "LITERAL_T4_MATERIALIZATION_PROFILE",
    "LITERAL_T4_PROFILE_ID",
    "LITERAL_T4_PROFILE_SCHEMA",
    "LITERAL_T4_RISK_METRIC",
    "LITERAL_T4_THRESHOLD_RULE",
    "LITERAL_T4_TRACE_ARTIFACT_KIND",
    "LITERAL_T4_TRACE_ARTIFACT_SCHEMA",
    "TRAIN_SPLIT",
    "V16T4_KIND",
    "VerifiedLiteralT4MaterializerGuard",
    "build_literal_t4_application",
    "build_v16t4_trace_artifact",
    "build_v16t4_record",
    "literal_t4_profile",
    "literal_t4_profile_sha256",
    "load_frozen_v16t4_threshold",
    "to_materializer_guard",
    "verified_literal_t4_materializer_guard_projection",
]
