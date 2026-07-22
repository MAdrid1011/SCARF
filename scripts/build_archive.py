#!/usr/bin/env python3
"""Build and verify a deterministic SCARF AE source archive."""

from __future__ import annotations

import argparse
import contextlib
import copy
import gzip
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MANIFEST_NAME = "release-manifest.json"
VERSION_RE = re.compile(r"v[0-9]+\.[0-9]+\.[0-9]+")
SHA256_RE = re.compile(r"[0-9a-f]{64}")

# Source archives are intentionally a closed set.  New top-level material must
# be reviewed before it can be published, rather than being included merely
# because it happened to be tracked in the working tree.
SOURCE_RELEASE_ROOT_FILES = frozenset(
    {
        ".gitignore",
        ".gitmodules",
        ".dockerignore",
        "ARTIFACT_EVALUATION.md",
        "Dockerfile",
        "LICENSE",
        "README.md",
        "THIRD_PARTY.md",
        "install.sh",
        "pytest.ini",
        "requirements.txt",
    }
)
SOURCE_RELEASE_DIRECTORIES = frozenset(
    {
        "adapters",
        "chisel",
        "data",
        "depth_predictor",
        "depthsplat",
        "docker",
        "docs",
        "encoder",
        "environments",
        "feature_extractor",
        "fsdr",
        "ggu",
        "hardware",
        "integration",
        "mvsplat",
        "saes",
        "scripts",
        "tests",
        "transplat",
    }
)
SOURCE_RELEASE_ARTIFACT_FILES = frozenset(
    {
        "artifact/CALIBRATION.md",
        "artifact/CLAIMS.md",
        "artifact/EVALUATION_PROTOCOL.md",
        "artifact/HARDWARE_SCOPE.md",
        "artifact/HOTCRP_SUBMISSION.md",
        "artifact/appendix.tex",
        "artifact/claim_status.json",
        "artifact/evaluation_catalog.json",
        "artifact/evaluation_protocol.json",
        "artifact/expected_results.json",
        "artifact/lsh_projection.json",
        "artifact/manifests/checkpoints.json",
        "artifact/manifests/datasets.json",
        "artifact/manifests/runtime_assets.json",
        "artifact/mechanism_config.json",
        "artifact/plot_style.mplstyle",
        "artifact/protocol/compiled.json",
        "artifact/protocol/reviewer/acid.json",
        "artifact/protocol/reviewer/dl3dv.json",
        "artifact/protocol/reviewer/manifest.json",
        "artifact/protocol/reviewer/re10k.json",
        "artifact/quick/evaluation_index.json",
        "artifact/reference_results/README.md",
        "artifact/reference_results/manifest.json",
        "artifact/reference_results/orin_nx_reference.csv",
        "artifact/release.json",
    }
)
SOURCE_RELEASE_QUICK_DATASET_PREFIX = "datasets/quick-re10k/"
# Non-release support files are excluded explicitly so the source bundle keeps
# the ordinary implementation, documentation, and tests in each subsystem.
SOURCE_RELEASE_EXCLUDED_PATH_PREFIXES = frozenset(
    {
        "data/acid_joint_",
        "data/context_only_audit_input.py",
        "data/frozen_audit_contract.py",
        "data/plan_acid_joint_",
        "data/prepare_dl3dv_multicontext_tangent_audit_inputs.py",
        "data/prepare_dl3dv_target_free_audit_inputs.py",
        "docs/saes-execution-dependency-audit.md",
        "docs/saes-rtl-contract.md",
        "docs/three-badge-readiness.md",
        "integration/acid_joint_",
        "saes/frozen_audit_preflight.py",
        "saes/joint_materialization_",
        "saes/depthsplat_literal_t4_virtual_coverage_audit.py",
        "saes/projected_domain_coverage_audit.py",
        "saes/projected_optical_moment_audit.py",
        "scripts/acid_joint_",
        "scripts/saes_adapter_offset_attribute_transport_quality_gate.py",
        "scripts/saes_deletion_risk_oracle_audit.py",
        "scripts/saes_dependency_audit",
        "scripts/saes_dependency_locality_audit.py",
        "scripts/saes_depthsplat_coverage_enriched_t4_context_only_audit.py",
        "scripts/saes_depthsplat_l0_l1_context_only_audit.py",
        "scripts/saes_depthsplat_l0_l1_execution_audit.py",
        "scripts/saes_depthsplat_literal_t4_virtual_coverage_audit.py",
        "scripts/saes_depthsplat_selected_output_audit.py",
        "scripts/saes_incremental_selected_output_audit.py",
        "scripts/saes_l1_primary_reference_",
        "scripts/saes_multicontext_directional_audit.py",
        "scripts/saes_mvsplat_raw_cost_volume_audit.py",
        "scripts/saes_paper_l0_l1_compact_packet_pilot.py",
        "scripts/saes_same_budget_",
        "scripts/saes_selected_output_quality_gate.py",
        "scripts/saes_selected_output_replay_audit.py",
        "scripts/saes_sparse_consumer_render_audit.py",
        "scripts/saes_sparse_packet_quality_pilot.py",
        "scripts/saes_target_free_materialization_audit.py",
        "tests/test_acid_joint_",
        "tests/test_context_only_audit_input.py",
        "tests/test_depthsplat_literal_t4_virtual_coverage_audit.py",
        "tests/test_frozen_audit_",
        "tests/test_incremental_selected_output_audit.py",
        "tests/test_joint_materialization_",
        "tests/test_prepare_dl3dv_target_free_audit_inputs.py",
        "tests/test_projected_domain_coverage_audit.py",
        "tests/test_saes_deletion_risk_oracle_audit.py",
        "tests/test_saes_adapter_offset_attribute_transport_quality_gate.py",
        "tests/test_saes_dependency_audit.py",
        "tests/test_saes_dependency_locality_audit.py",
        "tests/test_saes_depthsplat_coverage_enriched_t4_context_only_audit.py",
        "tests/test_saes_depthsplat_l0_l1_context_only_audit.py",
        "tests/test_saes_depthsplat_l0_l1_execution_audit.py",
        "tests/test_saes_depthsplat_selected_output_audit.py",
        "tests/test_saes_guard_partition_audit.py",
        "tests/test_saes_l1_primary_reference_attribute_audit.py",
        "tests/test_saes_materialization_audit.py",
        "tests/test_saes_multicontext_directional_audit.py",
        "tests/test_saes_multicontext_tangent_materialization.py",
        "tests/test_saes_mvsplat_raw_cost_volume_audit.py",
        "tests/test_saes_projected_optical_moment_audit.py",
        "tests/test_saes_same_budget_",
        "tests/test_saes_selected_output_quality_gate.py",
        "tests/test_saes_selected_output_replay.py",
        "tests/test_saes_sparse_packet_quality_pilot.py",
        "tests/test_sparse_consumer_render_audit.py",
    }
)
EVIDENCE_RELEASE_ROOT_FILES = frozenset({"reference-manifest.json"})
EVIDENCE_RELEASE_CONTRACT_FILES = frozenset(
    {
        "contracts/THIRD_PARTY.md",
        "contracts/checkpoints.json",
        "contracts/claim_status.json",
        "contracts/compiled_protocol.json",
        "contracts/datasets.json",
        "contracts/evaluation_protocol.json",
        "contracts/expected_results.json",
        "contracts/release.json",
        "contracts/runtime_assets.json",
    }
)


def normalize_archive_relative(value: str) -> str:
    """Normalize one safe, relative POSIX archive path without resolving ``..``."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"unsafe archive path: {value!r}")
    if "\x00" in value or "\\" in value:
        raise ValueError(f"unsafe archive path: {value}")
    windows = PureWindowsPath(value)
    pure = PurePosixPath(value)
    raw_parts = value.split("/")
    if pure.is_absolute() or windows.is_absolute() or windows.drive or ".." in raw_parts:
        raise ValueError(f"unsafe archive path: {value}")
    parts = tuple(part for part in pure.parts if part not in {".", ""})
    if not parts or any(part in {".", ".."} for part in parts):
        raise ValueError(f"unsafe archive path: {value}")
    return PurePosixPath(*parts).as_posix()


def _is_normalized_archive_relative(value: str, normalized: str) -> bool:
    return value == normalized


def _require_regular_file(path: Path, *, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise FileNotFoundError(path) from exc
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")


def _is_source_release_path(normalized: str) -> bool:
    parts = PurePosixPath(normalized).parts
    if any(
        normalized.startswith(prefix)
        for prefix in SOURCE_RELEASE_EXCLUDED_PATH_PREFIXES
    ):
        return False
    if "checkpoints" in parts:
        return False
    if len(parts) == 1:
        return normalized in SOURCE_RELEASE_ROOT_FILES
    if normalized in SOURCE_RELEASE_ARTIFACT_FILES:
        return True
    if normalized.startswith(SOURCE_RELEASE_QUICK_DATASET_PREFIX):
        return True
    return parts[0] in SOURCE_RELEASE_DIRECTORIES


def _is_evidence_release_path(normalized: str) -> bool:
    parts = PurePosixPath(normalized).parts
    return (
        normalized in EVIDENCE_RELEASE_ROOT_FILES
        or normalized in EVIDENCE_RELEASE_CONTRACT_FILES
        or (len(parts) > 1 and parts[0] == "evidence")
    )


def _bundle_path_is_allowed(bundle_kind: str | None, normalized: str) -> bool:
    if bundle_kind == "source":
        return _is_source_release_path(normalized)
    if bundle_kind == "evidence":
        return _is_evidence_release_path(normalized)
    return False


def _validate_prefix(prefix: str) -> str:
    try:
        normalized = normalize_archive_relative(prefix)
    except ValueError as exc:
        raise ValueError("archive prefix must be one safe path component") from exc
    if (
        normalized != prefix
        or len(PurePosixPath(normalized).parts) != 1
        or normalized in {".", ".."}
    ):
        raise ValueError("archive prefix must be one safe path component")
    return normalized


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_bundle_binding(source: dict[str, Any], evidence: dict[str, Any]) -> None:
    for field in ("git_commit", "submodules", "zenodo_doi"):
        if source.get(field) != evidence.get(field):
            raise ValueError(f"source and evidence bundles disagree on {field}")


def write_sha256sums(paths: list[Path], output: Path) -> None:
    if not paths:
        raise ValueError("at least one release bundle is required")
    resolved = [Path(path).resolve() for path in paths]
    if len({path.name for path in resolved}) != len(resolved):
        raise ValueError("release bundle names must be unique")
    lines = []
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(path)
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def release_files() -> list[Path]:
    names = git("ls-files", "--recurse-submodules", "-z").split("\0")
    files = []
    for name in names:
        if not name:
            continue
        normalized = normalize_archive_relative(name)
        if normalized != name:
            raise ValueError(f"tracked path is not normalized: {name}")
        path = ROOT / normalized
        _require_regular_file(path, label="tracked release input")
        files.append(path)
    return sorted(files)


def include_in_source_release(relative: Path) -> bool:
    """Return whether one tracked path belongs in the public source bundle."""
    raw = relative.as_posix()
    try:
        normalized = normalize_archive_relative(raw)
    except ValueError:
        return False
    if not _is_normalized_archive_relative(raw, normalized):
        return False
    return _is_source_release_path(normalized)


def source_release_files() -> list[Path]:
    selected = []
    for path in release_files():
        relative = path.relative_to(ROOT)
        if include_in_source_release(relative):
            selected.append(path)
    return selected


def evidence_release_files(
    root: Path = ROOT, *, reference_results: Path | None = None
) -> dict[str, Path]:
    reference_root = (
        Path(reference_results).resolve()
        if reference_results is not None
        else root / "artifact/reference_results"
    )
    manifest_path = reference_root / "manifest.json"
    _require_regular_file(manifest_path, label="reference evidence manifest")
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(manifest, dict):
        raise ValueError("reference evidence manifest must be a JSON object")
    if manifest.get("status") != "complete" or manifest.get("validation_status") != "PASS":
        raise ValueError("reference evidence is not staged from a passing run")
    if manifest.get("validation_require_key_results") is not True:
        raise ValueError(
            "reference evidence was not validated with --require-key-results"
        )
    files = {"reference-manifest.json": manifest_path}
    declared_files = manifest.get("files")
    if not isinstance(declared_files, dict):
        raise ValueError("reference evidence manifest has no valid file mapping")
    for relative, declared in declared_files.items():
        normalized = normalize_archive_relative(relative)
        if (
            normalized != relative
            or not _is_evidence_release_path(normalized)
            or not normalized.startswith("evidence/")
        ):
            raise ValueError(f"unsafe reference evidence path: {relative}")
        source = reference_root / normalized
        _require_regular_file(source, label="reference evidence input")
        if (
            not isinstance(declared, dict)
            or declared.get("size") != source.stat().st_size
            or declared.get("sha256") != hashlib.sha256(source.read_bytes()).hexdigest()
        ):
            raise ValueError(f"reference evidence hash mismatch: {relative}")
        files[relative] = source
    contracts = {
        "contracts/evaluation_protocol.json": root / "artifact/evaluation_protocol.json",
        "contracts/claim_status.json": root / "artifact/claim_status.json",
        "contracts/compiled_protocol.json": root / "artifact/protocol/compiled.json",
        "contracts/expected_results.json": root / "artifact/expected_results.json",
        "contracts/datasets.json": root / "artifact/manifests/datasets.json",
        "contracts/checkpoints.json": root / "artifact/manifests/checkpoints.json",
        "contracts/runtime_assets.json": root / "artifact/manifests/runtime_assets.json",
        "contracts/THIRD_PARTY.md": root / "THIRD_PARTY.md",
    }
    for relative, source in contracts.items():
        _require_regular_file(source, label="evidence contract input")
        files[relative] = source
    release_path = root / "artifact/release.json"
    if release_path.is_file():
        _require_regular_file(release_path, label="evidence release metadata")
        files["contracts/release.json"] = release_path
    return files


def _validated_archive_file_mapping(
    files: Mapping[str, Path], *, bundle_kind: str | None
) -> dict[str, Path]:
    if bundle_kind not in {"source", "evidence"}:
        raise ValueError(f"unsupported release bundle kind: {bundle_kind!r}")
    normalized_files: dict[str, Path] = {}
    non_normalized: list[str] = []
    for relative, source in files.items():
        normalized = normalize_archive_relative(relative)
        if normalized == MANIFEST_NAME:
            raise ValueError(f"archive file mapping reserves {MANIFEST_NAME}")
        if normalized in normalized_files:
            raise ValueError(f"duplicate normalized archive path: {normalized}")
        if not _is_normalized_archive_relative(relative, normalized):
            non_normalized.append(relative)
        if not _bundle_path_is_allowed(bundle_kind, normalized):
            raise ValueError(
                f"archive path is not allowed in {bundle_kind} bundle: {relative}"
            )
        path = Path(source)
        _require_regular_file(path, label="release archive input")
        normalized_files[normalized] = path
    if non_normalized:
        raise ValueError(
            "archive file mapping contains non-normalized paths: "
            + ", ".join(sorted(non_normalized)[:10])
        )
    return normalized_files


def _write_tar(
    tar_path: Path,
    prefix: str,
    files: dict[str, Path],
    manifest: dict[str, Any],
) -> None:
    prefix = _validate_prefix(prefix)
    files = _validated_archive_file_mapping(
        files, bundle_kind=manifest.get("bundle_kind")
    )
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as archive:
        for relative, path in sorted(files.items()):
            data = path.read_bytes()
            info = tarfile.TarInfo(f"{prefix}/{relative}")
            info.size = len(data)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o755 if os.access(path, os.X_OK) else 0o644
            archive.addfile(info, io.BytesIO(data))
        info = tarfile.TarInfo(f"{prefix}/{MANIFEST_NAME}")
        info.size = len(manifest_bytes)
        info.mtime = 0
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(manifest_bytes))


def _compress_tar(tar_path: Path, output: Path, archive_format: str) -> None:
    if archive_format == "tar.gz":
        with tar_path.open("rb") as source, output.open("wb") as raw_output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    compressed.write(block)
        return
    if archive_format != "tar.zst":
        raise ValueError(f"unsupported archive format: {archive_format}")
    result = subprocess.run(
        ["zstd", "--threads=1", "--no-progress", "-19", "-f", str(tar_path), "-o", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "zstd compression failed")


def _build_from_mapping(
    output: Path,
    prefix: str,
    manifest: dict[str, Any],
    files: dict[str, Path],
    archive_format: str,
) -> dict[str, Any]:
    manifest = copy.deepcopy(manifest)
    prefix = _validate_prefix(prefix)
    files = _validated_archive_file_mapping(
        files, bundle_kind=manifest.get("bundle_kind")
    )
    manifest["files"] = {
        relative: hashlib.sha256(path.read_bytes()).hexdigest()
        for relative, path in sorted(files.items())
    }
    manifest["archive"] = {
        "prefix": prefix,
        "format": archive_format,
        "deterministic_metadata": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".tar", delete=False) as temporary:
        tar_path = Path(temporary.name)
    try:
        _write_tar(tar_path, prefix, files, manifest)
        _compress_tar(tar_path, output, archive_format)
    finally:
        tar_path.unlink(missing_ok=True)
    verified = verify(output)
    verified["archive_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    verified["archive_size"] = output.stat().st_size
    return verified


def build(
    output: Path,
    prefix: str,
    require_doi: bool = False,
    *,
    reference_results: Path | None = None,
) -> dict[str, Any]:
    if git("status", "--porcelain"):
        raise ValueError("release archive requires a clean worktree")
    _validate_prefix(prefix)
    from scripts.check_release import build_manifest

    identity = build_manifest(require_doi, reference_results=reference_results)
    if not identity["validation"]["pass"]:
        raise ValueError("release checks failed: " + "; ".join(identity["validation"]["failures"]))
    files = {
        path.relative_to(ROOT).as_posix(): path for path in source_release_files()
    }
    source_manifest = {**identity, "bundle_kind": "source"}
    return _build_from_mapping(output, prefix, source_manifest, files, "tar.gz")


def build_source_only(
    output: Path,
    prefix: str,
    require_doi: bool = False,
) -> dict[str, Any]:
    """Build an Artifact-Available source archive without an evidence claim.

    This route retains every source-release validation except the staged
    Results-Reproduced evidence requirement.  It deliberately labels the
    archive as source-only, so it cannot be mistaken for the full source plus
    evidence release produced by :func:`build_bundles`.
    """

    if git("status", "--porcelain"):
        raise ValueError("source-only archive requires a clean worktree")
    _validate_prefix(prefix)
    from scripts.check_release import build_manifest

    identity = build_manifest(
        require_doi,
        require_reference_evidence=False,
    )
    if not identity["validation"]["pass"]:
        raise ValueError(
            "source-only release checks failed: "
            + "; ".join(identity["validation"]["failures"])
        )
    files = {
        path.relative_to(ROOT).as_posix(): path for path in source_release_files()
    }
    source_manifest = {
        **identity,
        "bundle_kind": "source",
        "release_scope": "artifact-available-source-only",
        "evidence_bundle_included": False,
        "results_reproduced_evidence": "not-included",
    }
    return _build_from_mapping(output, prefix, source_manifest, files, "tar.gz")


def build_bundles(
    output_dir: Path,
    version: str,
    *,
    require_doi: bool = False,
    reference_results: Path | None = None,
) -> dict[str, Any]:
    if not VERSION_RE.fullmatch(version):
        raise ValueError("release version must have the form vMAJOR.MINOR.PATCH")
    if git("status", "--porcelain"):
        raise ValueError("release bundles require a clean worktree")
    from scripts.check_release import build_manifest

    identity = build_manifest(require_doi, reference_results=reference_results)
    if not identity["validation"]["pass"]:
        raise ValueError(
            "release checks failed: " + "; ".join(identity["validation"]["failures"])
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / f"SCARF-AE-source-{version}.tar.gz"
    evidence_path = output_dir / f"SCARF-AE-evidence-{version}.tar.zst"
    source_manifest = {**identity, "bundle_kind": "source", "release_version": version}
    evidence_manifest = {
        key: copy.deepcopy(identity.get(key))
        for key in ("schema_version", "git_commit", "submodules", "zenodo_doi")
    }
    evidence_manifest.update(
        {"bundle_kind": "evidence", "release_version": version, "validation": {"pass": True, "failures": []}}
    )
    source_files = {
        path.relative_to(ROOT).as_posix(): path for path in source_release_files()
    }
    source = _build_from_mapping(
        source_path, f"SCARF-AE-source-{version}", source_manifest, source_files, "tar.gz"
    )
    evidence = _build_from_mapping(
        evidence_path,
        f"SCARF-AE-evidence-{version}",
        evidence_manifest,
        evidence_release_files(reference_results=reference_results),
        "tar.zst",
    )
    validate_bundle_binding(source, evidence)
    checksums = output_dir / "SHA256SUMS"
    write_sha256sums([source_path, evidence_path], checksums)
    return {
        "schema_version": "1.0",
        "status": "PASS",
        "version": version,
        "source": source,
        "evidence": evidence,
        "sha256sums": str(checksums),
    }


@contextlib.contextmanager
def _readable_tar_path(archive_path: Path):
    if not archive_path.name.endswith(".tar.zst"):
        yield archive_path
        return
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as temporary:
        tar_path = Path(temporary.name)
        result = subprocess.run(
            ["zstd", "-d", "--no-progress", "-c", str(archive_path)],
            stdout=temporary,
            stderr=subprocess.PIPE,
            check=False,
        )
    try:
        if result.returncode:
            raise ValueError("cannot decompress evidence archive")
        yield tar_path
    finally:
        tar_path.unlink(missing_ok=True)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate JSON object key in archive manifest: {key}")
        record[key] = value
    return record


def _manifest_files(
    manifest: dict[str, Any], *, bundle_kind: str
) -> dict[str, str]:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("archive manifest has no valid file mapping")
    if bundle_kind not in {"source", "evidence"}:
        raise ValueError(f"unsupported release bundle kind: {bundle_kind!r}")
    normalized_files: dict[str, str] = {}
    non_normalized: list[str] = []
    for relative, expected_hash in files.items():
        normalized = normalize_archive_relative(relative)
        if normalized == MANIFEST_NAME:
            raise ValueError(f"archive manifest reserves {MANIFEST_NAME}")
        if normalized in normalized_files:
            raise ValueError(f"duplicate normalized manifest path: {normalized}")
        if not _is_normalized_archive_relative(relative, normalized):
            non_normalized.append(relative)
        if not _bundle_path_is_allowed(bundle_kind, normalized):
            raise ValueError(
                f"archive manifest path is not allowed in {bundle_kind} bundle: {relative}"
            )
        if not isinstance(expected_hash, str) or not SHA256_RE.fullmatch(expected_hash):
            raise ValueError(f"archive manifest has an invalid SHA256: {relative}")
        normalized_files[normalized] = expected_hash
    if non_normalized:
        raise ValueError(
            "archive manifest contains non-normalized paths: "
            + ", ".join(sorted(non_normalized)[:10])
        )
    return normalized_files


def verify(archive_path: Path) -> dict[str, Any]:
    with _readable_tar_path(archive_path) as readable, tarfile.open(readable, "r:*") as archive:
        members = archive.getmembers()
        if not members:
            raise ValueError("archive is empty")
        archived: dict[str, tarfile.TarInfo] = {}
        non_normalized: list[str] = []
        for member in members:
            if not member.isreg() or member.issparse():
                raise ValueError(f"non-regular archive member: {member.name}")
            normalized = normalize_archive_relative(member.name)
            if normalized in archived:
                raise ValueError(f"duplicate normalized archive member: {normalized}")
            if not _is_normalized_archive_relative(member.name, normalized):
                non_normalized.append(member.name)
            if len(PurePosixPath(normalized).parts) < 2:
                raise ValueError(f"archive member is outside the archive root: {member.name}")
            archived[normalized] = member
        if non_normalized:
            raise ValueError(
                "archive contains non-normalized paths: "
                + ", ".join(sorted(non_normalized)[:10])
            )
        roots = {PurePosixPath(name).parts[0] for name in archived}
        if len(roots) != 1:
            raise ValueError("archive must have exactly one root directory")
        root = next(iter(roots))
        manifest_name = f"{root}/{MANIFEST_NAME}"
        manifest_member = archived.get(manifest_name)
        if manifest_member is None:
            raise ValueError("archive is missing release-manifest.json")
        manifest_stream = archive.extractfile(manifest_member)
        if manifest_stream is None:
            raise ValueError("release manifest cannot be read")
        manifest = json.loads(
            manifest_stream.read(), object_pairs_hook=_reject_duplicate_json_keys
        )
        if not isinstance(manifest, dict):
            raise ValueError("archive manifest must be a JSON object")
        archive_record = manifest.get("archive")
        if archive_record is not None and (
            not isinstance(archive_record, dict) or archive_record.get("prefix") != root
        ):
            raise ValueError("archive manifest prefix does not match archive root")
        bundle_kind = manifest.get("bundle_kind", "source")
        manifest_files = _manifest_files(manifest, bundle_kind=bundle_kind)
        failures = []
        for relative, expected_hash in manifest_files.items():
            name = f"{root}/{relative}"
            member = archived.get(name)
            if member is None:
                failures.append(f"missing {relative}")
                continue
            stream = archive.extractfile(member)
            actual = sha256_bytes(stream.read()) if stream is not None else "unreadable"
            if actual != expected_hash:
                failures.append(f"hash mismatch {relative}")
        expected_names = {f"{root}/{name}" for name in manifest_files}
        extra = sorted(set(archived) - expected_names - {manifest_name})
        failures.extend(f"unmanifested {name}" for name in extra)
    if failures:
        raise ValueError("archive verification failed: " + "; ".join(failures[:10]))
    return {
        "schema_version": "1.0",
        "status": "PASS",
        "prefix": root,
        "verified_files": len(manifest_files),
        "git_commit": manifest.get("git_commit"),
        "submodules": manifest.get("submodules"),
        "zenodo_doi": manifest.get("zenodo_doi"),
        "bundle_kind": bundle_kind,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prefix", default="SCARF-AE")
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--version", default="v1.0.0")
    parser.add_argument("--require-doi", action="store_true")
    parser.add_argument(
        "--reference-results",
        type=Path,
        help="Use a separately staged, hash-verified reference-results directory",
    )
    parser.add_argument(
        "--source-only",
        action="store_true",
        help=(
            "Build a clean Artifact-Available source archive without packaging "
            "or claiming Results-Reproduced evidence."
        ),
    )
    args = parser.parse_args()
    if sum(bool(value) for value in (args.output, args.verify, args.output_dir)) != 1:
        parser.error("specify exactly one of --output, --output-dir, or --verify")
    if args.source_only and (args.output is None or args.reference_results is not None):
        parser.error("--source-only requires --output and does not accept --reference-results")
    try:
        if args.source_only:
            result = build_source_only(
                args.output.resolve(),
                args.prefix,
                args.require_doi,
            )
        elif args.output_dir:
            result = build_bundles(
                args.output_dir.resolve(),
                args.version,
                require_doi=args.require_doi,
                reference_results=(
                    args.reference_results.resolve()
                    if args.reference_results
                    else None
                ),
            )
        elif args.output:
            result = build(
                args.output.resolve(),
                args.prefix,
                args.require_doi,
                reference_results=(
                    args.reference_results.resolve()
                    if args.reference_results
                    else None
                ),
            )
        else:
            if args.reference_results:
                raise ValueError("--reference-results is not used with --verify")
            result = verify(args.verify.resolve())
    except (OSError, RuntimeError, tarfile.TarError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
