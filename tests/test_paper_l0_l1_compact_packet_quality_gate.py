"""Unit coverage for the compact-packet target-free quality gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _sha(character: str) -> str:
    return character * 64


def _calibration(character: str) -> dict:
    return {
        "sha256": _sha(character),
        "threshold_value": 0.25,
        "acid_binding": {"plan_sha256": _sha("a")},
        "application": {
            "model": "transplat",
            "dataset": "dl3dv",
            "checkpoint_sha256": _sha("b"),
        },
    }


def _audit_record(pilot) -> tuple[dict, dict, dict, dict]:
    checkpoint_sha256 = _sha("b")
    v15 = _calibration("c")
    v16 = _calibration("d")
    route = {
        "source_selection_mask_sha256": _sha("e"),
        "selected_output_mask_sha256": _sha("f"),
        "packed_source_trace_sha256": _sha("0"),
    }
    record = {
        "schema_version": "1.0",
        "kind": pilot.TARGET_FREE_AUDIT_KIND,
        "status": pilot.TARGET_FREE_AUDIT_STATUS,
        "paper_result_eligible": False,
        "model": pilot.MODEL,
        "dataset": pilot.DATASET,
        "scene": "sample-zero",
        "checkpoint_sha256": checkpoint_sha256,
        "target_mapping_present": False,
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
        "access_evidence": {
            "target_mapping_present": False,
            "target_rgb_accessed": False,
            "target_camera_metadata_accessed": False,
            "target_index_accessed": False,
            "frozen_calibrations_target_free": True,
            "frozen_calibrations_verified_before_encoder_execution": True,
            "packed_guard_executed_before_target_access": True,
        },
        "execution": {
            "checkpoint_sha256": checkpoint_sha256,
            "encoder_only": True,
            "decoder_constructed": False,
        },
        "execution_boundary": {
            "renderer_executed": False,
            "quality_metrics_computed": False,
            "compact_nonzero_materialization_enabled": True,
            "frozen_l1_15_packed_guard_executed": True,
            "frozen_calibrations_verified_before_encoder_execution": True,
        },
        "input_identity": {
            "scene": "sample-zero",
            "source_sample_index": pilot.SAMPLE_INDEX,
            "target_rgb_accessed": False,
            "target_camera_metadata_accessed": False,
            "context_indices": [0, 9],
        },
        "frozen_l1_15_mechanism": {
            "decision_semantics": pilot.DECISION_SEMANTICS,
            "l1_anchor_semantics": pilot.L1_ANCHOR_SEMANTICS,
            "l1_anchor_count": 15,
            "execution_policy": (
                pilot.ENGINEERING_L1_15_ADAPTIVE_ABSOLUTE_RESIDUAL_V4_LOO_DEV_POLICY
            ),
        },
        "route_plan": {
            "decision_semantics": pilot.DECISION_SEMANTICS,
            "l1_anchor_semantics": pilot.L1_ANCHOR_SEMANTICS,
            "l1_anchor_count": 15,
            "tile_size": pilot.TILE_SIZE,
            "feature_threshold": pilot.FEATURE_THRESHOLD,
            "depth_threshold": pilot.DEPTH_THRESHOLD,
        },
        "v15_calibration": pilot._frozen_calibration_identity(v15),
        "v16_calibration": pilot._frozen_calibration_identity(v16),
        "route_binding": dict(route),
    }
    return record, v15, v16, route


def _write_audit(path: Path, record: dict, pilot) -> None:
    payload = dict(record)
    payload["sha256"] = pilot._canonical_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _validate(path: Path, pilot, v15: dict, v16: dict, route: dict) -> dict:
    return pilot._validate_target_free_quality_audit(
        path,
        scene="sample-zero",
        context_indices=[0, 9],
        checkpoint_sha256=_sha("b"),
        v15_calibration=v15,
        v16_calibration=v16,
        **route,
    )


def test_quality_gate_binds_the_target_free_audit_to_packet_and_calibrations(tmp_path: Path):
    from scripts import saes_paper_l0_l1_compact_packet_pilot as pilot

    record, v15, v16, route = _audit_record(pilot)
    path = tmp_path / "target-free-audit.json"
    _write_audit(path, record, pilot)

    gate = _validate(path, pilot, v15, v16, route)

    assert gate["record_sha256"] == json.loads(path.read_text(encoding="utf-8"))["sha256"]
    assert gate["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert gate["v15_calibration_sha256"] == v15["sha256"]
    assert gate["v16_calibration_sha256"] == v16["sha256"]
    assert gate["packed_source_trace_sha256"] == route["packed_source_trace_sha256"]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda record: record["v15_calibration"].update({"sha256": _sha("1")}), "V15"),
        (
            lambda record: record["route_binding"].update(
                {"selected_output_mask_sha256": _sha("2")}
            ),
            "route binding",
        ),
        (lambda record: record.update({"target_rgb_accessed": True}), "target_rgb_accessed"),
        (lambda record: record.update({"checkpoint_sha256": _sha("3")}), "checkpoint"),
        (
            lambda record: record["frozen_l1_15_mechanism"].update(
                {"decision_semantics": "legacy-route"}
            ),
            "mechanism",
        ),
        (
            lambda record: record["route_plan"].update(
                {"decision_semantics": "legacy-route"}
            ),
            "route plan",
        ),
    ],
)
def test_quality_gate_rejects_audit_identity_or_route_drift(
    tmp_path: Path, mutate, match: str
):
    from scripts import saes_paper_l0_l1_compact_packet_pilot as pilot

    record, v15, v16, route = _audit_record(pilot)
    mutate(record)
    path = tmp_path / "target-free-audit.json"
    _write_audit(path, record, pilot)

    with pytest.raises(ValueError, match=match):
        _validate(path, pilot, v15, v16, route)


def test_quality_gate_rejects_audit_self_hash_drift(tmp_path: Path):
    from scripts import saes_paper_l0_l1_compact_packet_pilot as pilot

    record, v15, v16, route = _audit_record(pilot)
    path = tmp_path / "target-free-audit.json"
    _write_audit(path, record, pilot)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["route_binding"]["packed_source_trace_sha256"] = _sha("4")
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA256"):
        _validate(path, pilot, v15, v16, route)


def test_quality_gate_rejects_legacy_audit_without_the_calibration_and_packet_bindings(
    tmp_path: Path,
):
    from scripts import saes_paper_l0_l1_compact_packet_pilot as pilot

    record, v15, v16, route = _audit_record(pilot)
    record.pop("v15_calibration")
    record.pop("v16_calibration")
    record.pop("route_binding")
    path = tmp_path / "legacy-audit.json"
    _write_audit(path, record, pilot)

    with pytest.raises(ValueError, match="V15 calibration"):
        _validate(path, pilot, v15, v16, route)


def test_quality_gate_rejects_rehashed_route_or_threshold_identity_drift(tmp_path: Path):
    from scripts import saes_paper_l0_l1_compact_packet_pilot as pilot

    record, v15, v16, route = _audit_record(pilot)
    path = tmp_path / "target-free-audit.json"
    record["route_binding"]["selected_output_mask_sha256"] = _sha("4")
    _write_audit(path, record, pilot)
    with pytest.raises(ValueError, match="route binding"):
        _validate(path, pilot, v15, v16, route)

    record, v15, v16, route = _audit_record(pilot)
    record["v16_calibration"]["threshold_value"] = 0.75
    _write_audit(path, record, pilot)
    with pytest.raises(ValueError, match="V16 calibration"):
        _validate(path, pilot, v15, v16, route)


def test_quality_gate_requires_the_full_audited_context_input_identity():
    from scripts import saes_paper_l0_l1_compact_packet_pilot as pilot

    identity = {
        "scene": "sample-zero",
        "context_indices": [0, 9],
        "tree_sha256": _sha("1"),
        "manifest_sha256": _sha("2"),
        "audit_input_sha256": _sha("3"),
        "source_sample_index": 0,
        "source_binding": {"canonical_index_sha256": _sha("4")},
        "sidecar": {"record_sha256": _sha("5")},
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
    }
    audit = {"input_identity": identity}

    assert pilot._require_exact_audited_context_input_identity(audit, identity) == identity

    changed = {**identity, "sidecar": {"record_sha256": _sha("6")}}
    with pytest.raises(ValueError, match="input identity"):
        pilot._require_exact_audited_context_input_identity(audit, changed)


def test_quality_gate_rejects_a_context_root_with_changed_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from data import context_only_audit_input
    from scripts import saes_paper_l0_l1_compact_packet_pilot as pilot

    identity = {
        "scene": "sample-zero",
        "context_indices": [0, 9],
        "tree_sha256": _sha("1"),
        "manifest_sha256": _sha("2"),
        "audit_input_sha256": _sha("3"),
        "source_sample_index": 0,
        "source_binding": {"canonical_index_sha256": _sha("4")},
        "sidecar": {"record_sha256": _sha("5")},
        "target_rgb_accessed": False,
        "target_camera_metadata_accessed": False,
    }
    audit = {"input_identity": identity}
    monkeypatch.setattr(
        context_only_audit_input,
        "validate_context_only_audit_input",
        lambda _root: {**identity, "tree_sha256": _sha("6")},
    )

    with pytest.raises(ValueError, match="input identity"):
        pilot._validate_target_free_context_input(audit, tmp_path / "other-root")


def test_native_target_loader_runs_after_gate_and_discards_native_context():
    from scripts import saes_paper_l0_l1_compact_packet_pilot as pilot

    target = {
        "image": torch.zeros((1, 2, 3, 4, 4)),
        "index": torch.tensor([[1, 3]]),
    }
    calls: list[dict] = []

    class Loader:
        def load_data(self, bundle, **kwargs):
            calls.append({"bundle": bundle, **kwargs})
            return SimpleNamespace(
                batch={
                    "scene": ["sample-zero"],
                    "context": {"index": torch.tensor([[0, 9]])},
                    "target": target,
                }
            )

    bundle = object()
    target_batch = pilot._load_native_target_batch_after_packet_gate(
        Loader(), bundle, scene="sample-zero", context_indices=[0, 9]
    )

    assert calls == [
        {
            "bundle": bundle,
            "dataset_name": "dl3dv",
            "num_samples": 1,
            "sample_index": 0,
        }
    ]
    assert target_batch["scene"] == ["sample-zero"]
    assert target_batch["target"] is target
    assert "context" not in target_batch


def test_quality_cli_requires_disjoint_calibrations_and_target_free_audit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    from scripts import saes_paper_l0_l1_compact_packet_pilot as pilot

    with pytest.raises(SystemExit) as error:
        pilot.main(["--output-dir", str(tmp_path / "output")])

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "--v15-calibration-record" in stderr
    assert "--v16-calibration-record" in stderr
    assert "--target-free-audit-artifact" in stderr
    assert "--target-free-input-root" in stderr
