"""Unit coverage for the compact-packet target-free quality gate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _sha(character: str) -> str:
    return character * 64


def _calibration(
    character: str,
    *,
    native_dense: bool = False,
    model_name: str = "transplat",
    classic_backend_identity: dict | None = None,
) -> dict:
    application = {
        "model": model_name,
        "dataset": "dl3dv",
        "checkpoint_sha256": _sha("b"),
    }
    if classic_backend_identity is not None:
        application["classic_backend_identity"] = dict(classic_backend_identity)
    record = {
        "sha256": _sha(character),
        "threshold_value": 0.25,
        "acid_binding": {"plan_sha256": _sha("a")},
        "application": application,
    }
    if native_dense:
        from saes.incremental_selected_output_execution import (
            RAW_HEAD_EXECUTION_CONTRACT,
        )

        record["raw_head_execution_contract"] = RAW_HEAD_EXECUTION_CONTRACT
    return record


def _native_dense_execution(pilot, *, finalized: bool) -> dict:
    from saes.incremental_selected_output_execution import (
        NATIVE_DENSE_HEAD_EXECUTION_EVIDENCE_VERSION,
        RAW_HEAD_EXECUTION_CONTRACT,
    )

    dense_positions = 32
    phases = []
    for index, phase in enumerate(("primary", "secondary", "full")):
        positions = dense_positions if index == 0 else 0
        phases.append(
            {
                "phase": phase,
                "mask_sha256": _sha("1" if index == 0 else "2" if index == 1 else "3"),
                "tile_trace_sha256": _sha(
                    "4" if index == 0 else "5" if index == 1 else "6"
                ),
                "first_conv_positions_executed": positions,
                "native_dense_first_conv_positions_executed": positions,
                "second_conv_positions_executed": positions,
                "native_dense_second_conv_positions_executed": positions,
            }
        )
    return {
        "schema_version": NATIVE_DENSE_HEAD_EXECUTION_EVIDENCE_VERSION,
        "raw_head_execution_contract": RAW_HEAD_EXECUTION_CONTRACT,
        "head_weight_sha256": _sha("7"),
        "head_input_sha256": _sha("8"),
        "phase_trace_sha256": _sha("9"),
        "tile_trace_sha256": _sha("0"),
        "execution_finalized": finalized,
        "dense_head_positions": dense_positions,
        "dense_head_macs": 4096,
        "actual_head_macs": 4096,
        "head_mac_delta": 0,
        "phases": phases,
    }


def _audit_record(
    pilot,
    *,
    sample_index: int | None = None,
    model_name: str | None = None,
    classic_backend_identity: dict | None = None,
) -> tuple[dict, dict, dict, dict]:
    if sample_index is None:
        sample_index = pilot.SAMPLE_INDEX
    if model_name is None:
        model_name = pilot.MODEL
    checkpoint_sha256 = _sha("b")
    v15 = _calibration(
        "c",
        model_name=model_name,
        classic_backend_identity=classic_backend_identity,
    )
    v16 = _calibration(
        "d",
        native_dense=True,
        model_name=model_name,
        classic_backend_identity=classic_backend_identity,
    )
    initial_native_dense_execution = _native_dense_execution(pilot, finalized=False)
    guarded_native_dense_execution = _native_dense_execution(pilot, finalized=True)
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
        "model": model_name,
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
            "source_sample_index": sample_index,
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
        "v16_calibration": pilot._frozen_calibration_identity(
            v16, require_native_dense_head_execution=True
        ),
        "raw_head": {
            "native_dense_head_execution": {
                "initial": initial_native_dense_execution,
                "guarded": guarded_native_dense_execution,
            }
        },
        "packed_adapter": {
            "final_packet_source_trace": {
                "raw_head_execution_contract": v16["raw_head_execution_contract"],
                "native_dense_head_execution": guarded_native_dense_execution,
            }
        },
        "route_binding": dict(route),
    }
    if classic_backend_identity is not None:
        record["execution"]["model"] = model_name
        record["execution"]["classic_backend_identity"] = dict(classic_backend_identity)
        record["execution_boundary"]["backend_coordinate_semantics"] = (
            classic_backend_identity["coordinate_semantics"]
        )
    return record, v15, v16, route


def _write_audit(path: Path, record: dict, pilot) -> None:
    payload = dict(record)
    payload["sha256"] = pilot._canonical_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _validate(
    path: Path,
    pilot,
    v15: dict,
    v16: dict,
    route: dict,
    *,
    sample_index: int | None = None,
    model_name: str | None = None,
    classic_backend_identity: dict | None = None,
) -> dict:
    if sample_index is None:
        sample_index = pilot.SAMPLE_INDEX
    kwargs = {
        "scene": "sample-zero",
        "context_indices": [0, 9],
        "checkpoint_sha256": _sha("b"),
        "v15_calibration": v15,
        "v16_calibration": v16,
        "sample_index": sample_index,
        **route,
    }
    if model_name is not None:
        kwargs["model_name"] = model_name
    if classic_backend_identity is not None:
        kwargs["classic_backend_identity"] = classic_backend_identity
    return pilot._validate_target_free_quality_audit(
        path,
        **kwargs,
    )


def _native_decoder_modules(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pilot
) -> tuple[object, type, SimpleNamespace, ModuleType, ModuleType]:
    model_root = tmp_path / "mvsplat"
    decoder_source = (
        model_root / "src" / "model" / "decoder" / "decoder_splatting_cuda.py"
    )
    gaussians_source = model_root / "src" / "model" / "types.py"
    decoder_source.parent.mkdir(parents=True, exist_ok=True)
    gaussians_source.parent.mkdir(parents=True, exist_ok=True)
    decoder_source.write_text("# fixture decoder\n", encoding="utf-8")
    gaussians_source.write_text("# fixture Gaussians\n", encoding="utf-8")

    gaussians_module = ModuleType("fixture_mvsplat.src.model.types")
    gaussians_module.__file__ = str(gaussians_source)

    @dataclass
    class DecoderGaussians:
        means: object
        covariances: object
        harmonics: object
        opacities: object

    DecoderGaussians.__module__ = gaussians_module.__name__
    gaussians_module.Gaussians = DecoderGaussians

    decoder_module = ModuleType(
        "fixture_mvsplat.src.model.decoder.decoder_splatting_cuda"
    )
    decoder_module.__file__ = str(decoder_source)
    decoder_module.Gaussians = DecoderGaussians

    class Decoder:
        def forward(self, *_args, **_kwargs):
            raise AssertionError("fixture decoder should not render")

    Decoder.__module__ = decoder_module.__name__
    decoder_module.Decoder = Decoder
    monkeypatch.setitem(sys.modules, gaussians_module.__name__, gaussians_module)
    monkeypatch.setitem(sys.modules, decoder_module.__name__, decoder_module)
    monkeypatch.setattr(pilot, "ROOT", tmp_path)

    checkpoint = model_root / "checkpoints" / "re10k.ckpt"
    return (
        Decoder(),
        DecoderGaussians,
        SimpleNamespace(model="mvsplat", checkpoint=checkpoint),
        decoder_module,
        gaussians_module,
    )


def test_quality_gate_binds_the_target_free_audit_to_packet_and_calibrations(
    tmp_path: Path,
):
    from scripts import saes_paper_l0_l1_compact_packet_pilot as pilot

    record, v15, v16, route = _audit_record(pilot)
    path = tmp_path / "target-free-audit.json"
    _write_audit(path, record, pilot)

    gate = _validate(path, pilot, v15, v16, route)

    assert (
        gate["record_sha256"] == json.loads(path.read_text(encoding="utf-8"))["sha256"]
    )
    assert gate["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert gate["v15_calibration_sha256"] == v15["sha256"]
    assert gate["v16_calibration_sha256"] == v16["sha256"]
    assert gate["sample_index"] == pilot.SAMPLE_INDEX
    assert gate["packed_source_trace_sha256"] == route["packed_source_trace_sha256"]


def test_decoder_gaussian_resolver_uses_the_active_mvsplat_decoder_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from scripts import saes_paper_l0_l1_compact_packet_pilot as pilot

    decoder, gaussians_type, contract, _decoder_module, _gaussians_module = (
        _native_decoder_modules(monkeypatch, tmp_path, pilot)
    )

    resolved_type, binding = pilot._resolve_decoder_gaussians_type(
        decoder, backend_contract=contract
    )

    assert resolved_type is gaussians_type
    assert binding["model"] == "mvsplat"
    assert binding["decoder_module"]["path"] == (
        "mvsplat/src/model/decoder/decoder_splatting_cuda.py"
    )
    assert binding["gaussians_module"]["path"] == "mvsplat/src/model/types.py"


@pytest.mark.parametrize("foreign_module", ("decoder", "gaussians"))
def test_decoder_gaussian_resolver_rejects_foreign_mvsplat_module_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, foreign_module: str
):
    from scripts import saes_paper_l0_l1_compact_packet_pilot as pilot

    decoder, _gaussians_type, contract, decoder_module, gaussians_module = (
        _native_decoder_modules(monkeypatch, tmp_path, pilot)
    )
    foreign_source = tmp_path / "foreign" / f"{foreign_module}.py"
    foreign_source.parent.mkdir()
    foreign_source.write_text("# foreign fixture\n", encoding="utf-8")
    if foreign_module == "decoder":
        decoder_module.__file__ = str(foreign_source)
    else:
        gaussians_module.__file__ = str(foreign_source)

    with pytest.raises(RuntimeError, match="foreign source tree"):
        pilot._resolve_decoder_gaussians_type(decoder, backend_contract=contract)


def test_quality_gate_binds_mvsplat_audit_to_the_matching_backend_identity(
    tmp_path: Path,
):
    from scripts import saes_paper_l0_l1_compact_packet_pilot as pilot

    backend_identity = {
        "schema_version": "classic-backend-frozen-identity-v1",
        "model": "mvsplat",
        "coordinate_semantics": "mvsplat-inline-pixel-center-plus-sigmoid-offset",
        "source_files": {"raw_head": {"sha256": _sha("a")}},
    }
    record, v15, v16, route = _audit_record(
        pilot,
        model_name="mvsplat",
        classic_backend_identity=backend_identity,
    )
    path = tmp_path / "mvsplat-target-free-audit.json"
    _write_audit(path, record, pilot)

    gate = _validate(
        path,
        pilot,
        v15,
        v16,
        route,
        model_name="mvsplat",
        classic_backend_identity=backend_identity,
    )

    assert gate["model"] == "mvsplat"
    with pytest.raises(ValueError, match="classic backend identity"):
        _validate(path, pilot, v15, v16, route, model_name="mvsplat")
    with pytest.raises(ValueError, match="classic backend identity"):
        _validate(
            path,
            pilot,
            v15,
            v16,
            route,
            model_name="mvsplat",
            classic_backend_identity={**backend_identity, "source_files": {}},
        )
    with pytest.raises(ValueError, match="identity"):
        _validate(path, pilot, v15, v16, route)

    record["execution_boundary"]["backend_coordinate_semantics"] = (
        "transplat-inline-pixel-center-plus-sigmoid-offset"
    )
    _write_audit(path, record, pilot)
    with pytest.raises(ValueError, match="rendering boundary"):
        _validate(
            path,
            pilot,
            v15,
            v16,
            route,
            model_name="mvsplat",
            classic_backend_identity=backend_identity,
        )


def test_quality_gate_requires_the_audited_nondefault_sample_index(tmp_path: Path):
    from scripts import saes_paper_l0_l1_compact_packet_pilot as pilot

    sample_index = 7
    record, v15, v16, route = _audit_record(pilot, sample_index=sample_index)
    path = tmp_path / "target-free-audit.json"
    _write_audit(path, record, pilot)

    assert (
        _validate(path, pilot, v15, v16, route, sample_index=sample_index)[
            "sample_index"
        ]
        == sample_index
    )
    with pytest.raises(ValueError, match="input identity"):
        _validate(path, pilot, v15, v16, route, sample_index=sample_index + 1)


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
        (
            lambda record: record.update({"target_rgb_accessed": True}),
            "target_rgb_accessed",
        ),
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


def test_quality_gate_rejects_rehashed_route_or_threshold_identity_drift(
    tmp_path: Path,
):
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

    assert (
        pilot._require_exact_audited_context_input_identity(audit, identity) == identity
    )

    current_identity = {
        **identity,
        "target_mapping_present": False,
        "target_index_accessed": False,
    }
    assert (
        pilot._require_exact_audited_context_input_identity(audit, current_identity)
        == current_identity
    )

    with pytest.raises(ValueError, match="input identity"):
        pilot._require_exact_audited_context_input_identity(
            audit, {**current_identity, "target_mapping_present": True}
        )

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
        lambda _root, *, model: (
            {**identity, "tree_sha256": _sha("6")}
            if model == pilot.MODEL
            else pytest.fail("legacy context validator received the wrong model")
        ),
    )

    with pytest.raises(ValueError, match="input identity"):
        pilot._validate_target_free_context_input(audit, tmp_path / "other-root")


def test_quality_gate_context_validator_forwards_mvsplat_model(
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
    calls: list[tuple[Path, str]] = []

    def validate(input_root: Path, *, model: str) -> dict:
        calls.append((input_root, model))
        return identity

    monkeypatch.setattr(
        context_only_audit_input, "validate_context_only_audit_input", validate
    )

    assert (
        pilot._validate_target_free_context_input(
            {"input_identity": identity},
            tmp_path / "mvsplat-context",
            model_name="mvsplat",
        )
        == identity
    )
    assert calls == [(tmp_path / "mvsplat-context", "mvsplat")]


def test_native_target_loader_uses_the_audited_execution_index_after_gate():
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
        Loader(),
        bundle,
        scene="sample-zero",
        context_indices=[0, 9],
        target_indices=[1, 3],
        execution_index=7,
        native_sample_count=8,
    )

    assert calls == [
        {
            "bundle": bundle,
            "dataset_name": "dl3dv",
            "num_samples": 8,
            "sample_index": 7,
        }
    ]
    assert target_batch["scene"] == ["sample-zero"]
    assert target_batch["target"] is target
    assert "context" not in target_batch


def test_native_target_loader_rejects_target_index_drift_after_packet_gate():
    from scripts import saes_paper_l0_l1_compact_packet_pilot as pilot

    class Loader:
        def load_data(self, _bundle, **_kwargs):
            return SimpleNamespace(
                batch={
                    "scene": ["sample-zero"],
                    "context": {"index": torch.tensor([[0, 9]])},
                    "target": {
                        "image": torch.zeros((1, 2, 3, 4, 4)),
                        "index": torch.tensor([[1, 5]]),
                    },
                }
            )

    with pytest.raises(RuntimeError, match="target indices"):
        pilot._load_native_target_batch_after_packet_gate(
            Loader(),
            object(),
            scene="sample-zero",
            context_indices=[0, 9],
            target_indices=[1, 3],
            execution_index=0,
            native_sample_count=1,
        )


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


def test_quality_collector_rejects_a_negative_sample_index_before_loading(
    tmp_path: Path,
):
    from scripts import saes_paper_l0_l1_compact_packet_pilot as pilot

    with pytest.raises(ValueError, match="sample_index"):
        pilot.collect_paper_compact_packet_pilot(
            device=torch.device("cpu"),
            v15_calibration_record=tmp_path / "v15.json",
            v16_calibration_record=tmp_path / "v16.json",
            target_free_audit_artifact=tmp_path / "audit.json",
            target_free_input_root=tmp_path / "input",
            sample_index=-1,
        )


def test_quality_cli_rejects_a_negative_sample_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    from scripts import saes_paper_l0_l1_compact_packet_pilot as pilot

    with pytest.raises(SystemExit) as error:
        pilot.main(
            [
                "--output-dir",
                str(tmp_path / "output"),
                "--v15-calibration-record",
                str(tmp_path / "v15.json"),
                "--v16-calibration-record",
                str(tmp_path / "v16.json"),
                "--target-free-audit-artifact",
                str(tmp_path / "audit.json"),
                "--target-free-input-root",
                str(tmp_path / "input"),
                "--sample-index",
                "-1",
            ]
        )

    assert error.value.code == 2
    assert "--sample-index must be non-negative" in capsys.readouterr().err
