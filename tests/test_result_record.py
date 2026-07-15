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
        "provenance",
        "quality",
        "performance",
        "ablation",
        "fsdr_saes",
        "hardware",
        "validation",
    }
    assert record["provenance"]["checkpoint"]["path"].startswith("<external>/")
    output = tmp_path / "results.json"
    write_result(record, output)
    assert json.loads(output.read_text(encoding="utf-8")) == record


def test_result_record_reports_signed_degradation_and_absolute_change(tmp_path):
    from scripts.result_record import build_quality_record

    quality = build_quality_record(
        baseline={"psnr_db": 20.0, "ssim": 0.8, "lpips": 0.2},
        scarf={"psnr_db": 19.0, "ssim": 0.79, "lpips": 0.21},
    )
    assert quality["change"]["psnr_signed_pct"] == -5.0
    assert quality["change"]["psnr_degradation_pct"] == 5.0
    assert quality["change"]["psnr_absolute_pct"] == 5.0


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
    )

    assert record["validation"] == {
        "reproducible": False,
        "reference_fallback_used": True,
        "fallback_stages": ["depth"],
    }
    with pytest.raises(ValueError, match="reference_fallback_used"):
        validate(record)


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
    }


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
