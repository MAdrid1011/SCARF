import json
from pathlib import Path

import pytest


def evidence_binding(*, calibration_status: str = "preregistered") -> dict:
    from scripts.stage_reference_results import canonical_sha256

    calibration = {
        "status": calibration_status,
        "manifest_sha256": "4" * 64,
        "candidate_records_sha256": "5" * 64
        if calibration_status == "calibrated"
        else None,
        "evaluation_disjoint": calibration_status == "calibrated",
        "expected_results_accessed": False,
        "global_configuration": True,
    }
    return {
        "source": {
            "git_commit": "a" * 40,
            "git_dirty": False,
            "source_identity": "git",
            "source_tree_sha256": "b" * 64,
            "submodules": {
                "transplat": "c" * 40,
                "mvsplat": "d" * 40,
                "depthsplat": "e" * 40,
            },
        },
        "mechanism": {
            "mechanism_config_sha256": "f" * 64,
            "calibration_status": calibration_status,
            "calibration_provenance_sha256": canonical_sha256(calibration),
            "calibration_provenance": calibration,
        },
    }


def provenance_result(binding: dict) -> dict:
    return {
        "provenance": {
            **binding["source"],
            "mechanism_config_sha256": binding["mechanism"]["mechanism_config_sha256"],
            "calibration_provenance": json.loads(
                json.dumps(binding["mechanism"]["calibration_provenance"])
            ),
        }
    }


def executable_sample_result(binding: dict, index: int = 0) -> dict:
    return {
        "schema_version": "1.0",
        "provenance": {
            **binding["source"],
            "mechanism_config_sha256": binding["mechanism"]["mechanism_config_sha256"],
            "calibration_provenance": json.loads(
                json.dumps(binding["mechanism"]["calibration_provenance"])
            ),
            "command": ["python", "scripts/demo.py"],
            "runtime_assets": {"VGG16": {"sha256": "1" * 64}},
            "environment": {"profile": "classic", "digest_sha256": "2" * 64},
            "seed": 0,
            "model": "mvsplat",
            "device": {"type": "cuda", "name": "test"},
            "dataset": {
                "name": "re10k",
                "representation": "re10k-native",
                "functional_fixture": False,
                "paper_result_eligible": False,
                "manifest": "manifest.json",
                "sha256": "3" * 64,
                "tree_sha256": "4" * 64,
            },
            "checkpoint": {
                "path": "mvsplat/re10k.ckpt",
                "sha256": "5" * 64,
                "load": {
                    "matched_tensors": 1,
                    "matched_checkpoint_numel_fraction": 1.0,
                },
            },
            "evaluation": {
                "kind": "sample",
                "sample_index": index,
                "execution_index": index,
                "candidate_count": 1,
                "scene": f"scene-{index}",
                "context_indices": [0, 1],
                "target_indices": [2],
                "target_view_count": 1,
                "target_view_aggregation": "arithmetic mean over selected target views",
            },
        },
        "quality": {
            "baseline": {"psnr_db": 28.0, "ssim": 0.9, "lpips": 0.1},
            "scarf": {"psnr_db": 27.9, "ssim": 0.89, "lpips": 0.11},
            "change": {
                "psnr_signed_pct": -0.1,
                "psnr_degradation_pct": 0.1,
                "psnr_absolute_pct": 0.1,
            },
            "views": [
                {
                    "target_index": 2,
                    "baseline": {"psnr_db": 28.0, "ssim": 0.9, "lpips": 0.1},
                    "scarf": {"psnr_db": 27.9, "ssim": 0.89, "lpips": 0.11},
                }
            ],
        },
        "performance": {
            "baseline_cycles": 1000,
            "scarf_cycles": 500,
            "speedup": 2.0,
            "baseline_source": "workstation_cuda_events",
            "cycle_source": "test_counter",
            "components": {"feature": 100, "depth": 200, "gaussian": 100, "ggu": 100},
        },
        "ablation": {"asic": {"eff_total": 500}},
        "fsdr_saes": {"fsdr": {"guided_rate": 0.5}},
        "hardware": {"physical_ppa_included": False},
        "validation": {"reproducible": True, "reference_fallback_used": False},
    }


def write_scarf_pair(source: Path, binding: dict) -> tuple[Path, Path]:
    from scripts.aggregate_results import aggregate

    sample = source / "quick/mvsplat_re10k/samples/sample_00000/results.json"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text(json.dumps(executable_sample_result(binding)), encoding="utf-8")
    aggregate_path = source / "quick/mvsplat_re10k/results.json"
    aggregate_path.write_text(
        json.dumps(aggregate([sample], 1)), encoding="utf-8"
    )
    return aggregate_path, sample


def write_pending_catalog(output: Path) -> Path:
    contract = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "artifact/evaluation_catalog.json"
        ).read_text(encoding="utf-8")
    )
    report_dir = output / "reports"
    status_dir = report_dir / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for requirement in contract["results"]:
        result_id = requirement["id"]
        status_path = status_dir / f"{result_id}.json"
        status_path.write_text(
            json.dumps(
                {
                    "id": result_id,
                    "status": "NOT_RUN",
                    "required_evidence_class": requirement["required_evidence_class"],
                }
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "id": result_id,
                "selected": False,
                "evidence_class": requirement["required_evidence_class"],
                "status": "NOT_RUN",
                "source_data": [],
                "exports": [f"status/{result_id}.json"],
            }
        )
    (report_dir / "reproduction_report.md").write_text(
        "generated evidence\n", encoding="utf-8"
    )
    catalog = report_dir / "figure_catalog.json"
    catalog.write_text(
        json.dumps({"schema_version": "2.0", "results": rows}),
        encoding="utf-8",
    )
    return catalog


def forge_pending_figure8_catalog_as_pass(output: Path) -> Path:
    """Forge a catalog-only Figure 8 pass with internally consistent metadata."""
    from scripts.generate_report import _source_records

    contract = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "artifact/evaluation_catalog.json"
        ).read_text(encoding="utf-8")
    )
    requirement = next(item for item in contract["results"] if item["id"] == "figure8")
    measurement = (
        output
        / "speedup/mvsplat_re10k/samples/sample_00000/orin-evidence/measurement.json"
    )
    measurement.parent.mkdir(parents=True, exist_ok=True)
    measurement.write_text("{}\n", encoding="utf-8")

    catalog_path = output / "reports/figure_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    figure8 = next(row for row in catalog["results"] if row["id"] == "figure8")
    figure8.update(
        {
            "selected": True,
            "status": "PASS",
            "source_data": _source_records(output, requirement["raw_inputs"]),
        }
    )
    status_path = output / "reports" / figure8["exports"][0]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["status"] = "PASS"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    return catalog_path


def write_minimal_staging_tree(source: Path, binding: dict) -> None:
    (source / "validation.json").write_text(
        json.dumps({"status": "PASS", "require_key_results": True}),
        encoding="utf-8",
    )
    write_scarf_pair(source, binding)
    for relative in ("rtl/results.json", "dram/results.json"):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(provenance_result(binding)), encoding="utf-8")
    for relative in (
        "datasets/re10k-validation.json",
        "environments/classic.json",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    write_pending_catalog(source)


def write_deepscale_pair(source: Path, binding: dict) -> tuple[Path, Path]:
    from hardware.scaling.deepscale import scale_record, sha256_file

    raw = {
        "schema_version": "1.0",
        "evidence_type": "asap7_predictive_postroute",
        "physical_valid": True,
        "metrics": {"logic_area_mm2": 1.0},
        "provenance": provenance_result(binding)["provenance"],
    }
    raw_path = source / "physical/asap7/ppa.json"
    estimate_path = source / "physical/asap7/ppa_28nm_estimated.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    estimate = scale_record(
        raw,
        7,
        28,
        raw_input_sha256=sha256_file(raw_path),
    )
    estimate_path.write_text(json.dumps(estimate), encoding="utf-8")
    return raw_path, estimate_path


def test_environment_and_dataset_records_are_required_release_categories():
    from scripts.stage_reference_results import (
        required_categories,
        required_environment_profiles,
    )

    assert "environments" in required_categories()
    assert "datasets" in required_categories()
    assert "quick" in required_categories()
    assert required_environment_profiles() == {"classic"}


def test_staging_rejects_nonpassing_or_incomplete_result_tree(tmp_path):
    from scripts.stage_reference_results import stage

    source = tmp_path / "outputs"
    source.mkdir()
    (source / "validation.json").write_text(
        json.dumps({"status": "FAIL"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="status PASS"):
        stage(source, tmp_path / "archive")

    (source / "validation.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="require-key-results"):
        stage(source, tmp_path / "archive")

    (source / "validation.json").write_text(
        json.dumps({"status": "PASS", "require_key_results": True}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing archive categories"):
        stage(source, tmp_path / "archive")


def test_generated_records_bind_source_tree_and_mechanism_config(tmp_path):
    from scripts.stage_reference_results import validate_generated_records

    binding = evidence_binding()
    result, sample = write_scarf_pair(tmp_path, binding)
    physical = tmp_path / "physical/asap7/ppa.json"
    physical.parent.mkdir(parents=True)
    physical.write_text(json.dumps(provenance_result(binding)), encoding="utf-8")
    files = {
        result: Path("quick/mvsplat_re10k/results.json"),
        sample: Path("quick/mvsplat_re10k/samples/sample_00000/results.json"),
        physical: Path("physical/asap7/ppa.json"),
    }

    records, failures = validate_generated_records(files, expected_binding=binding)
    assert failures == []
    assert set(records) == {
        "evidence/quick/mvsplat_re10k/results.json",
        "evidence/quick/mvsplat_re10k/samples/sample_00000/results.json",
        "evidence/physical/asap7/ppa.json",
    }

    original = json.loads(result.read_text(encoding="utf-8"))
    stale = json.loads(json.dumps(original))
    stale["provenance"]["source_tree_sha256"] = "0" * 64
    result.write_text(json.dumps(stale), encoding="utf-8")
    _, failures = validate_generated_records(files, expected_binding=binding)
    assert failures == [
        "generated evidence source provenance mismatch source_tree_sha256: "
        "evidence/quick/mvsplat_re10k/results.json"
    ]

    stale = json.loads(json.dumps(original))
    stale["provenance"]["mechanism_config_sha256"] = "0" * 64
    result.write_text(json.dumps(stale), encoding="utf-8")
    _, failures = validate_generated_records(files, expected_binding=binding)
    assert failures == [
        "generated evidence mechanism provenance mismatch mechanism_config_sha256: "
        "evidence/quick/mvsplat_re10k/results.json"
    ]

    stale = json.loads(json.dumps(original))
    stale["provenance"]["calibration_provenance"]["manifest_sha256"] = "0" * 64
    result.write_text(json.dumps(stale), encoding="utf-8")
    _, failures = validate_generated_records(files, expected_binding=binding)
    assert failures == [
        "generated evidence calibration provenance digest mismatch: "
        "evidence/quick/mvsplat_re10k/results.json"
    ]


def test_generated_scarf_results_require_a_complete_result_schema(tmp_path):
    from scripts.stage_reference_results import validate_generated_records

    binding = evidence_binding()
    result = tmp_path / "quick/mvsplat_re10k/results.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps(provenance_result(binding)), encoding="utf-8")

    _, failures = validate_generated_records(
        {result: Path("quick/mvsplat_re10k/results.json")},
        expected_binding=binding,
    )

    assert any("generated SCARF result is invalid" in failure for failure in failures)


def test_claim_facing_results_require_trace_bound_schema_v21(tmp_path):
    from scripts.stage_reference_results import validate_generated_records

    binding = evidence_binding()
    result = tmp_path / "quality/mvsplat_re10k/results.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps(executable_sample_result(binding)), encoding="utf-8")

    _, failures = validate_generated_records(
        {result: Path("quality/mvsplat_re10k/results.json")},
        expected_binding=binding,
    )

    assert any("require schema 2.1" in failure for failure in failures)


def test_generated_aggregate_is_rebuilt_from_its_staged_samples(tmp_path):
    from scripts.stage_reference_results import validate_generated_records

    binding = evidence_binding()
    aggregate_path, sample_path = write_scarf_pair(tmp_path, binding)
    aggregate_record = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate_record["quality"]["scarf"]["psnr_db"] = 20.0
    aggregate_path.write_text(json.dumps(aggregate_record), encoding="utf-8")

    _, failures = validate_generated_records(
        {
            aggregate_path: Path("quick/mvsplat_re10k/results.json"),
            sample_path: Path("quick/mvsplat_re10k/samples/sample_00000/results.json"),
        },
        expected_binding=binding,
    )

    assert failures == [
        "aggregate result does not match its staged sample results: "
        "evidence/quick/mvsplat_re10k/results.json"
    ]


def test_staging_records_the_current_source_and_config_binding(tmp_path, monkeypatch):
    import scripts.stage_reference_results as staging

    binding = evidence_binding()
    source = tmp_path / "outputs"
    source.mkdir()
    write_minimal_staging_tree(source, binding)
    monkeypatch.setattr(staging, "release_evidence_binding", lambda *_: binding)

    manifest = staging.stage(source, tmp_path / "archive")

    assert manifest["schema_version"] == "2.0"
    assert manifest["provenance"]["source"] == binding["source"]
    assert manifest["provenance"]["mechanism"] == binding["mechanism"]
    assert set(manifest["provenance"]["generated_records"]) == {
        "evidence/quick/mvsplat_re10k/results.json",
        "evidence/quick/mvsplat_re10k/samples/sample_00000/results.json",
        "evidence/rtl/results.json",
        "evidence/dram/results.json",
    }


def test_staging_rejects_stale_execution_records_before_copying(tmp_path, monkeypatch):
    import scripts.stage_reference_results as staging

    stale_binding = evidence_binding()
    current_binding = json.loads(json.dumps(stale_binding))
    current_binding["source"]["source_tree_sha256"] = "0" * 64
    source = tmp_path / "outputs"
    source.mkdir()
    write_minimal_staging_tree(source, stale_binding)
    monkeypatch.setattr(
        staging, "release_evidence_binding", lambda *_: current_binding
    )

    with pytest.raises(ValueError, match="source provenance mismatch source_tree_sha256"):
        staging.stage(source, tmp_path / "archive")


def test_staging_requires_a_revalidated_report_catalog(tmp_path, monkeypatch):
    import scripts.stage_reference_results as staging

    binding = evidence_binding()
    source = tmp_path / "outputs"
    source.mkdir()
    write_minimal_staging_tree(source, binding)
    (source / "reports/figure_catalog.json").unlink()
    monkeypatch.setattr(staging, "release_evidence_binding", lambda *_: binding)

    with pytest.raises(ValueError, match="generated report catalog is missing"):
        staging.stage(source, tmp_path / "archive")


def test_pending_figure8_catalog_pass_is_rejected_before_staging(tmp_path):
    from scripts.generate_report import _source_records
    from scripts.stage_reference_results import selected_files, validate_report_catalog

    binding = evidence_binding()
    source = tmp_path / "outputs"
    source.mkdir()
    aggregate_path, _ = write_scarf_pair(source, binding)
    speedup = source / "speedup/mvsplat_re10k/results.json"
    speedup.parent.mkdir(parents=True)
    speedup.write_bytes(aggregate_path.read_bytes())
    catalog_path = write_pending_catalog(source)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    figure8 = next(row for row in catalog["results"] if row["id"] == "figure8")
    figure8.update(
        {
            "selected": True,
            "status": "PASS",
            "source_data": _source_records(
                source,
                next(
                    item
                    for item in json.loads(
                        (
                            Path(__file__).resolve().parents[1]
                            / "artifact/evaluation_catalog.json"
                        ).read_text(encoding="utf-8")
                    )["results"]
                    if item["id"] == "figure8"
                )["raw_inputs"],
            ),
        }
    )
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    failures = validate_report_catalog(source, selected_files(source).values())

    assert any("Figure 8 catalog does not satisfy independent Orin evidence" in failure for failure in failures)


def test_staging_rejects_forged_pending_figure8_catalog_pass(tmp_path, monkeypatch):
    import scripts.stage_reference_results as staging

    binding = evidence_binding()
    source = tmp_path / "outputs"
    source.mkdir()
    write_minimal_staging_tree(source, binding)
    catalog_path = forge_pending_figure8_catalog_as_pass(source)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    figure8 = next(row for row in catalog["results"] if row["id"] == "figure8")
    status = json.loads(
        (source / "reports" / figure8["exports"][0]).read_text(encoding="utf-8")
    )

    assert figure8["source_data"]
    assert status == {
        "id": "figure8",
        "status": "PASS",
        "required_evidence_class": "independent_measurement",
    }
    assert staging.validate_report_catalog(source, staging.selected_files(source).values()) == [
        "Figure 8 catalog does not satisfy independent Orin evidence: "
        "CLAIMED_AWAITING_INDEPENDENT_ORIN_EVALUATION"
    ]

    monkeypatch.setattr(staging, "release_evidence_binding", lambda *_: binding)
    with pytest.raises(
        ValueError, match="Figure 8 catalog does not satisfy independent Orin evidence"
    ):
        staging.stage(source, tmp_path / "archive")


def test_staging_rejects_a_deepscale_estimate_with_the_wrong_raw_input_hash(
    tmp_path, monkeypatch
):
    import scripts.stage_reference_results as staging

    binding = evidence_binding()
    source = tmp_path / "outputs"
    source.mkdir()
    write_minimal_staging_tree(source, binding)
    _, estimate_path = write_deepscale_pair(source, binding)
    estimate = json.loads(estimate_path.read_text(encoding="utf-8"))
    estimate["raw_asap7_input"]["sha256"] = "0" * 64
    estimate_path.write_text(json.dumps(estimate), encoding="utf-8")
    monkeypatch.setattr(staging, "release_evidence_binding", lambda *_: binding)

    with pytest.raises(ValueError, match="DeepScale estimate input binding is invalid"):
        staging.stage(source, tmp_path / "archive")


def test_dram_raw_trace_files_are_selected_for_staging(tmp_path):
    from scripts.stage_reference_results import selected_files

    dram = tmp_path / "dram"
    dram.mkdir()
    for name in ("memory-events.jsonl", "ramulator.trace"):
        (dram / name).write_text("evidence\n", encoding="utf-8")

    selected = set(selected_files(tmp_path).values())
    assert selected == {
        Path("dram/memory-events.jsonl"),
        Path("dram/ramulator.trace"),
    }


def test_paper_reference_preview_files_are_not_selected_for_staging(tmp_path):
    from scripts.stage_reference_results import selected_files

    reports = tmp_path / "reports"
    reports.mkdir()
    for name in (
        "paper_reference.md",
        "table1_reference.csv",
        "mechanisms_reference.csv",
        "summary_reference.csv",
    ):
        (reports / name).write_text("PAPER_REFERENCE_ONLY\n", encoding="utf-8")
    (reports / "reproduction_report.md").write_text("evidence\n", encoding="utf-8")

    assert set(selected_files(tmp_path).values()) == {
        Path("reports/reproduction_report.md"),
    }


def test_marked_reference_preview_is_not_selected_after_rename_or_move(tmp_path):
    from scripts.stage_reference_results import selected_files

    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "opaque-output.csv").write_text(
        "artifact_class,PAPER_REFERENCE_ONLY\n", encoding="utf-8"
    )
    (reports / "reproduction_report.md").write_text("evidence\n", encoding="utf-8")
    moved = tmp_path / "speedup" / "transplat_re10k" / "results.json"
    moved.parent.mkdir(parents=True)
    moved.write_text(
        '{"artifact_class": "PAPER_REFERENCE_ONLY"}\n', encoding="utf-8"
    )
    sample = tmp_path / "quick" / "mvsplat_re10k" / "samples" / "sample_00000"
    sample.mkdir(parents=True)
    (sample / "results.json").write_text(
        '{"artifact_class": "PAPER_REFERENCE_ONLY"}\n', encoding="utf-8"
    )
    (sample / "opaque.png").write_text("PAPER_REFERENCE_ONLY\n", encoding="utf-8")

    assert set(selected_files(tmp_path).values()) == {
        Path("reports/reproduction_report.md"),
    }


def test_rtl_emission_and_execution_manifests_are_selected(tmp_path):
    from scripts.stage_reference_results import selected_files

    emitted = tmp_path / "rtl/rtl"
    emitted.mkdir(parents=True)
    (emitted / "ScarfTop.sv").write_text("module ScarfTop; endmodule\n")
    (emitted / "filelist.f").write_text("ScarfTop.sv\n")
    (tmp_path / "manifest-rtl.json").write_text("{}\n")

    selected = set(selected_files(tmp_path).values())
    assert selected == {
        Path("manifest-rtl.json"),
        Path("rtl/rtl/ScarfTop.sv"),
        Path("rtl/rtl/filelist.f"),
    }


def test_execution_manifest_archival_copy_normalizes_local_paths(tmp_path):
    from scripts.stage_reference_results import portable_execution_manifest

    root = tmp_path / "author/repo"
    manifest = tmp_path / "manifest-quick.json"
    manifest.write_text(
        json.dumps(
            {
                "root": str(root),
                "experiments": [
                    {
                        "environment_profile": "classic",
                        "command": [
                            "/opt/author/envs/classic/bin/python",
                            str(root / "scripts/demo.py"),
                        ],
                        "aggregate_command": [
                            "/opt/author/envs/classic/bin/python3",
                            str(root / "scripts/aggregate_results.py"),
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    portable = portable_execution_manifest(
        manifest,
        repository_root=root,
        orchestrator_python=Path("/opt/author/driver/bin/python3"),
    ).decode()

    assert str(root) not in portable
    assert "/opt/author" not in portable
    assert "$SCARF_ROOT/scripts/demo.py" in portable
    assert "$SCARF_PYTHON_CLASSIC" in portable
    assert '"applied": true' in portable


def test_execution_manifest_normalizes_declared_archive_root(tmp_path):
    from scripts.stage_reference_results import portable_execution_manifest

    archive_root = tmp_path / "extracted/SCARF-AE-source-v1.0.0"
    manifest = tmp_path / "manifest-quick.json"
    manifest.write_text(
        json.dumps(
            {
                "root": str(archive_root),
                "commands": [
                    [
                        "/opt/author/miniconda/bin/python3",
                        str(archive_root / "scripts/run_ae.py"),
                    ]
                ],
                "dataset_commands": [],
                "experiments": [],
            }
        ),
        encoding="utf-8",
    )

    portable = portable_execution_manifest(
        manifest,
        repository_root=tmp_path / "staging/repository",
        orchestrator_python=Path("/different/staging/env/bin/python"),
    ).decode()

    assert str(archive_root) not in portable
    assert "$SCARF_ROOT/scripts/run_ae.py" in portable
    assert "/opt/author" not in portable


def test_execution_manifest_normalizes_top_level_orchestrator_commands(tmp_path):
    from scripts.stage_reference_results import portable_execution_manifest

    root = tmp_path / "author/repo"
    manifest = tmp_path / "manifest-report.json"
    manifest.write_text(
        json.dumps(
            {
                "root": str(root),
                "commands": [
                    [
                        "/opt/author/miniconda/bin/python3",
                        str(root / "scripts/generate_report.py"),
                    ]
                ],
                "dataset_commands": [],
                "experiments": [],
            }
        ),
        encoding="utf-8",
    )

    portable = portable_execution_manifest(
        manifest,
        repository_root=root,
        orchestrator_python=Path("/different/staging/env/bin/python"),
    ).decode()

    assert "/opt/author" not in portable
    assert "$PYTHON" in portable


def test_prepared_dataset_validations_are_selected_for_staging(tmp_path):
    from scripts.stage_reference_results import selected_files

    datasets = tmp_path / "datasets"
    datasets.mkdir()
    for name in ("re10k-validation.json", "acid-validation.json"):
        (datasets / name).write_text("{}\n", encoding="utf-8")

    selected = set(selected_files(tmp_path).values())
    assert selected == {
        Path("datasets/re10k-validation.json"),
        Path("datasets/acid-validation.json"),
    }


def test_pair_progress_is_selected_for_staging(tmp_path):
    from scripts.stage_reference_results import selected_files

    pair = tmp_path / "quality/mvsplat_re10k"
    pair.mkdir(parents=True)
    (pair / "pair-execution.json").write_text("{}\n", encoding="utf-8")
    (pair / "progress.jsonl").write_text("{}\n", encoding="utf-8")

    selected = set(selected_files(tmp_path).values())
    assert selected == {
        Path("quality/mvsplat_re10k/pair-execution.json"),
        Path("quality/mvsplat_re10k/progress.jsonl"),
    }


def test_only_one_representative_image_set_per_pair_is_staged(tmp_path):
    from scripts.stage_reference_results import selected_files

    for sample in ("sample_00003", "sample_00007"):
        root = tmp_path / "ablation/mvsplat_re10k/samples" / sample
        root.mkdir(parents=True)
        (root / "results.json").write_text("{}\n", encoding="utf-8")
        (root / "scarf_00.png").write_bytes(b"png")

    selected = set(selected_files(tmp_path).values())

    assert Path(
        "ablation/mvsplat_re10k/samples/sample_00003/scarf_00.png"
    ) in selected
    assert Path(
        "ablation/mvsplat_re10k/samples/sample_00007/scarf_00.png"
    ) not in selected


def test_orin_measurement_raw_artifacts_are_selected_for_staging(tmp_path):
    from scripts.stage_reference_results import selected_files

    sample = tmp_path / "speedup/transplat_re10k/samples/sample_00000/orin-evidence"
    profile = tmp_path / "speedup/transplat_re10k/orin-profile"
    sample.mkdir(parents=True)
    profile.mkdir(parents=True)
    for name in ("measurement.json", "cuda-events.json", "tegrastats.log", "nsight.nsys-rep"):
        (sample / name).write_text("evidence\n", encoding="utf-8")
    for name in ("tegrastats.log", "nsight.nsys-rep"):
        (profile / name).write_text("evidence\n", encoding="utf-8")

    selected = set(selected_files(tmp_path).values())
    assert Path("speedup/transplat_re10k/samples/sample_00000/orin-evidence/tegrastats.log") in selected
    assert Path("speedup/transplat_re10k/samples/sample_00000/orin-evidence/nsight.nsys-rep") in selected
    assert Path("speedup/transplat_re10k/orin-profile/tegrastats.log") in selected
    assert Path("speedup/transplat_re10k/orin-profile/nsight.nsys-rep") in selected
