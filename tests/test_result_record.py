import json
from pathlib import Path

import pytest


def test_result_record_is_schema_valid_and_preserves_all_sections(tmp_path):
    from scripts.result_record import build_result_record, write_result
    from scripts.validate_result import validate

    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    dataset_manifest = tmp_path / "dataset.manifest.json"
    dataset_manifest.write_text('{"dataset":"re10k"}\n', encoding="utf-8")

    record = build_result_record(
        model="mvsplat",
        dataset="re10k",
        checkpoint=checkpoint,
        checkpoint_load={
            "matched_tensors": 10,
            "matched_checkpoint_numel_fraction": 0.99,
        },
        environment={"profile": "classic", "digest_sha256": "c" * 64},
        dataset_manifest=dataset_manifest,
        dataset_representation="re10k-native",
        dataset_tree_sha256="a" * 64,
        device={"type": "cuda", "name": "test gpu"},
        seed=7,
        quality={
            "baseline": {"psnr_db": 28.0, "ssim": 0.91, "lpips": 0.12},
            "scarf": {"psnr_db": 27.9, "ssim": 0.90, "lpips": 0.13},
        },
        quality_views=[
            {
                "target_index": 0,
                "baseline": {"psnr_db": 28.0, "ssim": 0.91, "lpips": 0.12},
                "scarf": {"psnr_db": 27.9, "ssim": 0.90, "lpips": 0.13},
            },
            {
                "target_index": 1,
                "baseline": {"psnr_db": 28.0, "ssim": 0.91, "lpips": 0.12},
                "scarf": {"psnr_db": 27.9, "ssim": 0.90, "lpips": 0.13},
            },
            {
                "target_index": 2,
                "baseline": {"psnr_db": 28.0, "ssim": 0.91, "lpips": 0.12},
                "scarf": {"psnr_db": 27.9, "ssim": 0.90, "lpips": 0.13},
            },
        ],
        baseline_cycles=1000,
        cycles={"feature": 100, "depth": 200, "gaussian": 50, "ggu": 25},
        cycle_source="scarf_simulator",
        ablation={"asic": {"cycles": 500}},
        fsdr_saes={"guided_rate": 0.3},
        command=["python", "scripts/demo.py"],
        runtime_assets={"VGG16": {"sha256": "a" * 64}},
        sample_identity={
            "scene": "scene-0001",
            "context_indices": [0, 10],
            "target_indices": [0, 1, 2],
        },
    )
    validate(record)
    assert set(record) == {
        "schema_version",
        "evidence_class",
        "provenance",
        "quality",
        "performance",
        "events",
        "energy",
        "ablation",
        "fsdr_saes",
        "hardware",
        "validation",
    }
    assert record["schema_version"] == "2.1"
    assert record["evidence_class"] == "deterministic_execution"
    assert set(record["performance"]["stages"]) == {"s1", "s2", "s3", "s4"}
    assert record["events"]["fsdr"]["total_pixels"] == 0
    assert record["events"]["fsdr"]["discrete_top1_available"] is False
    assert record["events"]["saes"]["total_tiles"] == 0
    assert record["energy"]["available"] is False
    assert record["provenance"]["execution_contract"] == {
        "run_class": "diagnostic",
        "saes_materialization": "representative",
    }
    assert record["provenance"]["checkpoint"]["path"].startswith("<external>/")
    output = tmp_path / "results.json"
    write_result(record, output)
    assert json.loads(output.read_text(encoding="utf-8")) == record


def test_functional_result_binds_the_runtime_saes_identity(tmp_path):
    from scripts.result_record import build_result_record
    from scripts.saes_execution_identity import build_saes_execution_identity
    from scripts.validate_result import validate

    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    dataset_manifest = tmp_path / "dataset.manifest.json"
    dataset_manifest.write_text("{}\n", encoding="utf-8")
    identity = build_saes_execution_identity()
    view = {
        "target_index": 0,
        "baseline": {"psnr_db": 28.0, "ssim": 0.91, "lpips": 0.12},
        "scarf": {"psnr_db": 27.9, "ssim": 0.90, "lpips": 0.13},
    }

    def build(*, runtime_identity, route_sha256):
        return build_result_record(
            model="mvsplat",
            dataset="re10k",
            checkpoint=checkpoint,
            checkpoint_load={
                "matched_tensors": 10,
                "matched_checkpoint_numel_fraction": 0.99,
            },
            environment={"profile": "classic", "digest_sha256": "c" * 64},
            dataset_manifest=dataset_manifest,
            dataset_representation="re10k-synthetic-functional-v1",
            dataset_tree_sha256="a" * 64,
            device={"type": "cuda", "name": "test gpu"},
            seed=7,
            quality={"baseline": view["baseline"], "scarf": view["scarf"]},
            quality_views=[view],
            baseline_cycles=1000,
            cycles={"feature": 100, "depth": 200, "gaussian": 50, "ggu": 25},
            cycle_source="scarf_simulator",
            ablation={"asic": {"cycles": 500}},
            fsdr_saes={
                "saes": {
                    "saes_execution_identity": runtime_identity,
                    "route_sha256": route_sha256,
                }
            },
            command=["python", "scripts/demo.py"],
            runtime_assets={"VGG16": {"sha256": "a" * 64}},
            sample_identity={
                "scene": "scene-0001",
                "context_indices": [0, 10],
                "target_indices": [0],
            },
            run_class="functional",
            paper_result_eligible=False,
        )

    with pytest.raises(ValueError, match="runtime SAES execution identity"):
        build(runtime_identity=None, route_sha256=identity["route_sha256"])

    record = build(
        runtime_identity=identity,
        route_sha256=identity["route_sha256"],
    )
    validate(record)
    assert record["provenance"]["saes_execution_identity"] == identity
    assert record["provenance"]["saes_execution_route_sha256"] == identity[
        "route_sha256"
    ]

    drifted = json.loads(json.dumps(record))
    drifted["fsdr_saes"]["saes"]["route_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="runtime SAES route SHA256"):
        validate(drifted)


def test_execution_trace_binds_nonquality_inputs_without_hashing_quality(tmp_path):
    from scripts.result_record import build_result_record, execution_trace_sha256
    from scripts.validate_result import validate

    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    dataset_manifest = tmp_path / "dataset.manifest.json"
    dataset_manifest.write_text('{"dataset":"re10k"}\n', encoding="utf-8")

    def build(
        *,
        scarf_psnr: float,
        level0_tiles: int,
        command: list[str] | None = None,
        measured_encoder_time_ms: float = 1.0,
        runtime_asset_path: str = "assets/vgg16.pth",
        run_class: str = "diagnostic",
        paper_result_eligible: bool = False,
    ) -> dict:
        view = {
            "target_index": 7,
            "baseline": {"psnr_db": 28.0, "ssim": 0.91, "lpips": 0.12},
            "scarf": {"psnr_db": scarf_psnr, "ssim": 0.90, "lpips": 0.13},
        }
        return build_result_record(
            model="mvsplat",
            dataset="re10k",
            checkpoint=checkpoint,
            checkpoint_load={
                "matched_tensors": 10,
                "matched_checkpoint_numel_fraction": 0.99,
            },
            environment={"profile": "classic", "digest_sha256": "c" * 64},
            dataset_manifest=dataset_manifest,
            dataset_representation="re10k-native",
            dataset_tree_sha256="a" * 64,
            device={
                "type": "cuda",
                "name": "test gpu",
                "measured_encoder_time_ms": measured_encoder_time_ms,
                "encoder_timing_samples_ms": [measured_encoder_time_ms],
            },
            seed=7,
            quality={"baseline": view["baseline"], "scarf": view["scarf"]},
            quality_views=[view],
            baseline_cycles=1000,
            cycles={"feature": 100, "depth": 200, "gaussian": 50, "ggu": 25},
            cycle_source="scarf_simulator",
            ablation={
                "asic": {
                    "eff_total": 300,
                    "quality": {
                        "psnr": 28.0,
                        "quality_views": [{"target_index": 7, "ssim": 0.91}],
                    },
                    "results_path": "/private/ablation/results.json",
                }
            },
            fsdr_saes={
                "fsdr": {
                    "guided_rate": 0.5,
                    "in_window_rate": 0.75,
                    "top1_coverage": 0.7,
                },
                "saes": {
                    "total_tiles_processed": 1,
                    "level0_tiles": level0_tiles,
                    "level1_tiles": 0,
                    "full_tiles": 1 - level0_tiles,
                    "level0_ratio": float(level0_tiles),
                    "level1_ratio": 0.0,
                    "modification_ratio": 0.5,
                    "hardware_accounting": {
                        "timing_class": "analytic_no_overlap_not_rtl_cycle_equivalent",
                        "target_rgb_accessed": False,
                        "cycles": {
                            "serialized_accounting_cycles": 17,
                            "controller_total": 3,
                        },
                        "assumptions": {
                            "rtl_cycle_equivalent": False,
                            "stage_semantics": "S2/S3 stay dense",
                        },
                        "results_path": "/private/ledger.json",
                        "quality": {"psnr": 1.0},
                    },
                },
                "preservation": {
                    "saes_low_var_agree": 0.9,
                    "saes_low_var_tiles": 9,
                    "saes_early_tiles": 10,
                    "saes_low_var_mean_similarity": 0.95,
                    "saes_low_var_threshold": 0.9,
                }
            },
            command=command or ["python", "scripts/demo.py"],
            runtime_assets={
                "VGG16": {"path": runtime_asset_path, "sha256": "a" * 64}
            },
            sample_identity={
                "scene": "scene-0001",
                "context_indices": [0, 10],
                "target_indices": [7],
            },
            run_class=run_class,
            paper_result_eligible=paper_result_eligible,
        )

    reference = build(scarf_psnr=27.9, level0_tiles=1)
    quality_changed = build(scarf_psnr=27.8, level0_tiles=1)
    wrapper_changed = build(
        scarf_psnr=27.9,
        level0_tiles=1,
        command=["python", "scripts/alternate_wrapper.py", "--output", "/tmp/run"],
        measured_encoder_time_ms=3.5,
    )
    asset_path_changed = build(
        scarf_psnr=27.9,
        level0_tiles=1,
        runtime_asset_path="/different-checkout/assets/vgg16.pth",
    )
    route_changed = build(scarf_psnr=27.9, level0_tiles=0)

    trace = reference["provenance"]["execution_trace"]
    digest = reference["provenance"]["execution_trace_sha256"]
    assert execution_trace_sha256(trace["inputs"]) == digest
    assert reference["quality"]["execution_trace_sha256"] == digest
    assert reference["performance"]["execution_trace_sha256"] == digest
    assert "quality" not in trace["inputs"]
    assert "command" not in trace["inputs"]["execution"]
    assert "device" not in trace["inputs"]["execution"]
    assert trace["inputs"]["execution"]["runtime_assets"] == {
        "VGG16": {"sha256": "a" * 64}
    }
    assert trace["inputs"]["selection"] == {
        "sample_index": 0,
        "execution_index": 0,
        "candidate_count": 1,
        "scene": "scene-0001",
        "context_indices": [0, 10],
        "target_indices": [7],
    }
    route_evidence = trace["inputs"]["route_evidence"]
    assert route_evidence["scarf_cycles"] == 375
    assert route_evidence["components"] == {
        "feature": 100,
        "depth": 200,
        "gaussian": 50,
        "ggu": 25,
    }
    assert route_evidence["events"]["saes"]["level0_tiles"] == 1
    assert route_evidence["ablation"] == {"asic": {"eff_total": 300}}
    assert route_evidence["mechanism_metrics"]["fsdr"] == {
        "guided_rate": 0.5,
        "in_window_rate": 0.75,
        "top1_coverage": 0.7,
    }
    assert route_evidence["mechanism_metrics"]["preservation"] == {
        "saes_low_var_agree": 0.9,
        "saes_low_var_tiles": 9,
        "saes_early_tiles": 10,
        "saes_low_var_mean_similarity": 0.95,
        "saes_low_var_threshold": 0.9,
    }
    hardware_ledger = route_evidence["events"]["saes"]["hardware_accounting"]
    assert hardware_ledger["cycles"]["serialized_accounting_cycles"] == 17
    assert "target_rgb_accessed" not in hardware_ledger
    assert hardware_ledger["assumptions"]["stage_semantics"] == "S2/S3 stay dense"
    assert "quality" not in hardware_ledger
    assert "results_path" not in hardware_ledger
    assert "baseline_cycles" not in route_evidence
    assert "baseline_source" not in route_evidence
    assert "speedup" not in route_evidence
    assert quality_changed["provenance"]["execution_trace_sha256"] == digest
    assert wrapper_changed["provenance"]["execution_trace_sha256"] == digest
    assert asset_path_changed["provenance"]["execution_trace_sha256"] == digest
    assert route_changed["provenance"]["execution_trace_sha256"] != digest

    validate(reference)
    validate(quality_changed)
    validate(wrapper_changed)
    validate(asset_path_changed)
    validate(route_changed)

    mutations = (
        lambda result: result["performance"].update(scarf_cycles=376),
        lambda result: result["ablation"]["asic"].update(eff_total=301),
        lambda result: result["events"]["saes"].update(
            source="tampered-event-counter"
        ),
        lambda result: result["events"]["saes"]["hardware_accounting"][
            "cycles"
        ].update(serialized_accounting_cycles=18),
        lambda result: result["fsdr_saes"]["preservation"].update(
            saes_low_var_agree=0.8
        ),
    )
    for mutation in mutations:
        tampered = json.loads(json.dumps(reference))
        mutation(tampered)
        with pytest.raises(ValueError, match="execution trace inputs"):
            validate(tampered)

    with pytest.raises(ValueError, match="source-bound timing evidence"):
        build(
            scarf_psnr=27.9,
            level0_tiles=1,
            run_class="claim",
            paper_result_eligible=True,
        )


def test_result_record_reports_signed_degradation_and_absolute_change(tmp_path):
    from scripts.result_record import build_quality_record

    quality = build_quality_record(
        baseline={"psnr_db": 20.0, "ssim": 0.8, "lpips": 0.2},
        scarf={"psnr_db": 19.0, "ssim": 0.79, "lpips": 0.21},
    )
    assert quality["change"]["psnr_signed_pct"] == -5.0
    assert quality["change"]["psnr_degradation_pct"] == 5.0
    assert quality["change"]["psnr_absolute_pct"] == 5.0


def test_result_record_rejects_assignment_consensus_before_quality_recording():
    from scripts.result_record import build_result_record

    with pytest.raises(ValueError, match="cannot produce a normal"):
        build_result_record(
            model="transplat",
            dataset="dl3dv",
            checkpoint=Path("unused.ckpt"),
            checkpoint_load={},
            environment={},
            dataset_manifest=Path("unused.json"),
            dataset_representation="re10k-compatible-360x640-v1",
            dataset_tree_sha256="0" * 64,
            device={},
            seed=0,
            quality={},
            quality_views=[],
            baseline_cycles=0,
            cycles={},
            cycle_source="unused",
            ablation={},
            fsdr_saes={},
            command=[],
            runtime_assets={},
            sample_identity={},
            saes_materialization=(
                "assignment-consensus-adapter-pseudo-descriptor-diagnostic"
            ),
        )


def test_result_record_preserves_explicit_v2_stage_and_event_evidence(tmp_path):
    from scripts.result_record import build_result_record
    from scripts.validate_result import validate

    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    dataset_manifest = tmp_path / "dataset.manifest.json"
    dataset_manifest.write_text("{}\n", encoding="utf-8")
    view = {
        "target_index": 0,
        "baseline": {"psnr_db": 28.0, "ssim": 0.9, "lpips": 0.1},
        "scarf": {"psnr_db": 27.9, "ssim": 0.9, "lpips": 0.1},
    }
    stages = {
        f"s{index}": {
            "cycles": index * 10,
            "useful_mmcu_slots": index * 7,
            "scheduled_mmcu_slots": index * 8,
            "mmcu_slots_available": True,
            "source": "event_simulator",
        }
        for index in range(1, 5)
    }
    events = {
        "fsdr": {
            "total_pixels": 100,
            "cache_hits": 80,
            "guided_pixels": 70,
            "guided_top1_covered": 69,
            "guided_top1_missed": 1,
            "discrete_top1_available": True,
            "full_depth_evaluations": 6400,
            "executed_depth_evaluations": 2200,
            "depth_evaluations_available": True,
            "feature_buffer_bytes_baseline": 1024,
            "feature_buffer_bytes_actual": 512,
            "feature_buffer_bytes_available": True,
            "source": "event_simulator",
        },
        "saes": {
            "total_tiles": 10,
            "level0_tiles": 2,
            "level1_tiles": 3,
            "full_tiles": 5,
            "tile_path_available": True,
            "baseline_gaussians": 100,
            "actual_gaussians": 70,
            "gaussian_counts_available": True,
            "full_s2_evaluations": 6400,
            "executed_s2_evaluations": 4000,
            "s2_evaluations_available": True,
            "source": "event_simulator",
        },
    }

    record = build_result_record(
        model="mvsplat",
        dataset="re10k",
        checkpoint=checkpoint,
        checkpoint_load={
            "matched_tensors": 10,
            "matched_checkpoint_numel_fraction": 0.99,
        },
        environment={"profile": "classic", "digest_sha256": "c" * 64},
        dataset_manifest=dataset_manifest,
        dataset_representation="re10k-native",
        dataset_tree_sha256="a" * 64,
        device={"type": "cuda", "name": "test"},
        seed=0,
        quality={"baseline": view["baseline"], "scarf": view["scarf"]},
        quality_views=[view],
        baseline_cycles=100,
        cycles={"feature": 10, "depth": 20, "gaussian": 30, "ggu": 40},
        cycle_source="scarf_simulator",
        ablation={},
        fsdr_saes={},
        command=["python", "scripts/demo.py"],
        runtime_assets={"VGG16": {"sha256": "b" * 64}},
        sample_identity={
            "scene": "scene",
            "context_indices": [0, 1],
            "target_indices": [0],
        },
        stage_records=stages,
        event_records=events,
    )

    validate(record)
    assert record["performance"]["stages"] == stages
    for namespace, values in events.items():
        for field, value in values.items():
            assert record["events"][namespace][field] == value
    assert record["events"]["fsdr"]["hamming_hits"] == 80
    assert record["events"]["fsdr"]["local_valid_hits"] == 70
    assert record["events"]["fsdr"]["local_invalid_fallbacks"] == 10
    assert record["events"]["saes"]["l0_representatives"] == 8
    assert record["events"]["saes"]["l1_lightweight_anchors"] == 36


def test_default_saes_events_count_executed_tile_paths():
    from scripts.result_record import _default_event_records
    from saes.hardware_accounting import build_saes_event_ledger

    saes_stats = {
        "total_tiles_processed": 10,
        "level0_tiles": 2,
        "level1_tiles": 3,
        "full_tiles": 5,
        "l0_representatives": 8,
            "l1_lightweight_anchors": 36,
        "full_stage3_gaussians": 80,
        "full_s2_evaluations": 1280,
        "executed_s2_evaluations": 640,
        "s2_evaluations_available": True,
    }
    saes_stats["hardware_accounting"] = build_saes_event_ledger(
        saes_stats, feature_dim=128, tile_size=4, sh_degree=4
    )

    events = _default_event_records(
        {
            "saes": saes_stats
        }
    )

    assert events["saes"]["full_s2_evaluations"] == 1280
    assert events["saes"]["executed_s2_evaluations"] == 640
    assert events["saes"]["s2_evaluations_available"] is True
    assert (
        events["saes"]["hardware_accounting"]["cycles"]
        ["serialized_accounting_cycles"]
        > 0
    )


def test_target_view_quality_mean_is_computed_from_per_view_records():
    from scripts.result_record import mean_view_quality

    views = [
        {
            "baseline": {
                "psnr_db": 20.0,
                "ssim": 0.80,
                "lpips": 0.10,
            }
        },
        {
            "baseline": {
                "psnr_db": 22.0,
                "ssim": 0.85,
                "lpips": 0.13,
            }
        },
        {
            "baseline": {
                "psnr_db": 24.0,
                "ssim": 0.90,
                "lpips": 0.17,
            }
        },
    ]

    aggregate = mean_view_quality(views, "baseline")

    assert aggregate["psnr_db"] == pytest.approx(22.0)
    assert aggregate["ssim"] == pytest.approx(0.85)
    assert aggregate["lpips"] == pytest.approx(0.4 / 3)


def test_result_record_marks_diagnostic_fallback_as_nonreproducible(tmp_path):
    from scripts.result_record import build_result_record
    from scripts.validate_result import validate

    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    dataset_manifest = tmp_path / "dataset.manifest.json"
    dataset_manifest.write_text("{}\n", encoding="utf-8")
    view = {
        "target_index": 0,
        "baseline": {"psnr_db": 28.0, "ssim": 0.9, "lpips": 0.1},
        "scarf": {"psnr_db": 27.9, "ssim": 0.9, "lpips": 0.1},
    }
    record = build_result_record(
        model="mvsplat",
        dataset="re10k",
        checkpoint=checkpoint,
        checkpoint_load={
            "matched_tensors": 10,
            "matched_checkpoint_numel_fraction": 0.99,
        },
        environment={"profile": "classic", "digest_sha256": "c" * 64},
        dataset_manifest=dataset_manifest,
        dataset_representation="re10k-native",
        dataset_tree_sha256="a" * 64,
        device={"type": "cuda", "name": "test"},
        seed=0,
        quality={"baseline": view["baseline"], "scarf": view["scarf"]},
        quality_views=[view],
        baseline_cycles=100,
        cycles={"feature": 10, "depth": 20, "gaussian": 30, "ggu": 40},
        cycle_source="scarf_simulator",
        ablation={},
        fsdr_saes={},
        command=["python", "scripts/demo.py"],
        runtime_assets={"VGG16": {"sha256": "b" * 64}},
        sample_identity={
            "scene": "scene",
            "context_indices": [0, 1],
            "target_indices": [0],
        },
        fallback_stages=["depth"],
        paper_result_eligible=False,
    )

    assert record["provenance"]["dataset"]["paper_result_eligible"] is False
    assert record["validation"] == {
        "reproducible": False,
        "reference_fallback_used": True,
        "fallback_stages": ["depth"],
    }
    with pytest.raises(ValueError, match="reference_fallback_used"):
        validate(record)


def test_claim_timing_uses_source_bound_variants_not_analytic_eff_total(tmp_path):
    from scripts.result_record import (
        CLAIM_TIMING_CYCLE_SOURCE,
        claim_ablation_speedups,
        build_result_record,
        bind_execution_trace,
    )
    from scripts.validate_result import validate
    from scripts.saes_execution_identity import build_saes_execution_identity

    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    dataset_manifest = tmp_path / "dataset.manifest.json"
    dataset_manifest.write_text("{}\n", encoding="utf-8")
    view = {
        "target_index": 0,
        "baseline": {"psnr_db": 28.0, "ssim": 0.9, "lpips": 0.1},
        "scarf": {"psnr_db": 27.9, "ssim": 0.89, "lpips": 0.11},
    }
    timing = {
        "schema_version": "source-bound-claim-timing-v1",
        "timing_class": "rtl_cycle_equivalent_source_bound",
        "rtl_cycle_equivalent": True,
        "clock_mhz": 1000,
        "trace": {"path": "timing-trace/sample.json", "sha256": "d" * 64},
        "variants": {
            "asic": {"total_cycles": 500},
            "asic_fsdr": {"total_cycles": 400},
            "asic_saes": {"total_cycles": 375},
            "asic_fsdr_saes": {"total_cycles": 300},
        },
        "combined_stage_cycles": {"s1": 50, "s2": 100, "s3": 75, "s4": 75},
    }
    stages = {
        stage: {
            "cycles": cycles,
            "useful_mmcu_slots": 1,
            "scheduled_mmcu_slots": 1,
            "mmcu_slots_available": True,
            "source": "rtl_trace",
        }
        for stage, cycles in timing["combined_stage_cycles"].items()
    }
    identity = build_saes_execution_identity()
    record = build_result_record(
        model="mvsplat",
        dataset="re10k",
        checkpoint=checkpoint,
        checkpoint_load={
            "matched_tensors": 1,
            "matched_checkpoint_numel_fraction": 1.0,
        },
        environment={"profile": "classic", "digest_sha256": "c" * 64},
        dataset_manifest=dataset_manifest,
        dataset_representation="re10k-native",
        dataset_tree_sha256="a" * 64,
        device={"type": "cuda", "name": "test"},
        seed=0,
        quality={"baseline": view["baseline"], "scarf": view["scarf"]},
        quality_views=[view],
        baseline_cycles=1000,
        scarf_cycles=300,
        cycles={"feature": 50, "depth": 100, "gaussian": 75, "ggu": 75},
        cycle_source=CLAIM_TIMING_CYCLE_SOURCE,
        ablation={key: {} for key in timing["variants"]},
        fsdr_saes={
            "saes": {
                "saes_execution_identity": identity,
                "route_sha256": identity["route_sha256"],
            }
        },
        command=["python", "scripts/demo.py"],
        runtime_assets={"VGG16": {"sha256": "b" * 64}},
        sample_identity={
            "scene": "scene",
            "context_indices": [0, 1],
            "target_indices": [0],
        },
        run_class="claim",
        paper_result_eligible=False,
        stage_records=stages,
        claim_timing=timing,
    )

    validate(record)
    assert claim_ablation_speedups(record) == {
        "fsdr": 1.25,
        "saes": 500 / 375,
        "combined": 500 / 300,
    }
    assert "claim_timing" in record["provenance"]["execution_trace"]["inputs"]["route_evidence"]

    tampered = json.loads(json.dumps(record))
    tampered["ablation"]["asic"]["eff_total"] = 1
    bind_execution_trace(tampered)
    with pytest.raises(ValueError, match="analytic eff_total"):
        validate(tampered)


def test_portable_command_removes_author_local_paths():
    from scripts.result_record import ROOT, portable_command

    command = portable_command(
        [
            "/" + "home/author/env/bin/python3.10",
            str(ROOT / "scripts/demo.py"),
            "--output-dir",
            "/scratch/private/run",
        ]
    )

    assert command == [
        "python",
        "scripts/demo.py",
        "--output-dir",
        "<external>/run",
    ]


def test_source_identity_uses_release_manifest_without_git(tmp_path):
    from scripts.result_record import source_identity

    source = tmp_path / "scripts/demo.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('fixture')\n", encoding="utf-8")
    import hashlib

    manifest = {
        "bundle_kind": "source",
        "validation": {"pass": True, "failures": []},
        "git_commit": "a" * 40,
        "submodules": {
            "transplat": "b" * 40,
            "mvsplat": "c" * 40,
            "depthsplat": "d" * 40,
        },
        "files": {
            "scripts/demo.py": hashlib.sha256(source.read_bytes()).hexdigest()
        },
    }
    (tmp_path / "release-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    identity = source_identity(tmp_path)

    assert identity == {
        "git_commit": "a" * 40,
        "git_dirty": False,
        "submodules": manifest["submodules"],
        "source": "release_manifest",
        "source_tree_sha256": identity["source_tree_sha256"],
    }
    assert len(identity["source_tree_sha256"]) == 64


def test_source_identity_rejects_modified_release_files(tmp_path):
    import hashlib

    from scripts.result_record import source_identity

    source = tmp_path / "script.py"
    source.write_text("original\n", encoding="utf-8")
    manifest = {
        "bundle_kind": "source",
        "validation": {"pass": True, "failures": []},
        "git_commit": "a" * 40,
        "submodules": {
            "transplat": "b" * 40,
            "mvsplat": "c" * 40,
            "depthsplat": "d" * 40,
        },
        "files": {"script.py": hashlib.sha256(source.read_bytes()).hexdigest()},
    }
    (tmp_path / "release-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    source.write_text("modified\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="hash mismatch"):
        source_identity(tmp_path)


def test_cached_file_hash_invalidates_when_the_file_changes(tmp_path):
    from scripts.result_record import sha256_file

    path = tmp_path / "asset.bin"
    path.write_bytes(b"a")
    first = sha256_file(path)
    path.write_bytes(b"b")
    second = sha256_file(path)

    assert first != second


def test_worktree_source_digest_changes_for_tracked_and_untracked_source(tmp_path):
    import subprocess
    from scripts.result_record import _worktree_source_sha256

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "ae@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "AE Test"], cwd=tmp_path, check=True
    )
    tracked = tmp_path / "source.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "source.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    submodules = {"transplat": "b" * 40, "mvsplat": "c" * 40, "depthsplat": "d" * 40}

    clean = _worktree_source_sha256(tmp_path, commit, submodules)
    tracked.write_text("value = 2\n", encoding="utf-8")
    tracked_dirty = _worktree_source_sha256(tmp_path, commit, submodules)
    (tmp_path / "new_source.py").write_text("value = 3\n", encoding="utf-8")
    untracked_dirty = _worktree_source_sha256(tmp_path, commit, submodules)

    assert clean != tracked_dirty
    assert tracked_dirty != untracked_dirty
