import json
from pathlib import Path

import pytest


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
    with pytest.raises(ValueError, match="missing archive categories"):
        stage(source, tmp_path / "archive")


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
