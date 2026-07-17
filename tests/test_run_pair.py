import json
from pathlib import Path


class FakeSession:
    loads = 0

    def __init__(self):
        type(self).loads += 1

    def run_sample(self, selection: dict, output_dir: Path) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "sample_index": selection["sample_index"],
            "scene": selection["scene"],
            "context_indices": selection["context_indices"],
            "target_indices": selection["target_indices"],
        }
        (output_dir / "results.json").write_text(json.dumps(result), encoding="utf-8")
        return result


def test_pair_worker_loads_once_and_resumes_complete_samples(tmp_path: Path):
    from scripts.run_pair import execute_pair

    selections = [
        {
            "sample_index": index,
            "scene": f"scene-{index}",
            "context_indices": [0, 2],
            "target_indices": [1],
        }
        for index in range(3)
    ]
    FakeSession.loads = 0
    first = execute_pair(selections, tmp_path, FakeSession, resume=True)
    second = execute_pair(selections, tmp_path, FakeSession, resume=True)

    assert first["executed"] == 3
    assert second["executed"] == 0
    assert second["resumed"] == 3
    assert FakeSession.loads == 1
    events = [
        json.loads(line)
        for line in (tmp_path / "progress.jsonl").read_text().splitlines()
    ]
    assert sum(event["event"] == "sample_complete" for event in events) == 3
    assert events[-1]["event"] == "pair_complete"


def test_pair_worker_reruns_incomplete_sample_boundary(tmp_path: Path):
    from scripts.run_pair import execute_pair

    sample = tmp_path / "samples/sample_00000"
    sample.mkdir(parents=True)
    (sample / "partial.tmp").write_text("incomplete", encoding="utf-8")
    selection = {
        "sample_index": 0,
        "scene": "scene-0",
        "context_indices": [0, 2],
        "target_indices": [1],
    }
    FakeSession.loads = 0
    record = execute_pair([selection], tmp_path, FakeSession, resume=True)

    assert record["executed"] == 1
    assert json.loads((sample / "results.json").read_text())["scene"] == "scene-0"


def test_pair_worker_ignores_internal_execution_ordinal_for_resume(tmp_path: Path):
    from scripts.run_pair import execute_pair

    selection = {
        "sample_index": 7,
        "execution_index": 3,
        "scene": "scene-7",
        "context_indices": [0, 2],
        "target_indices": [1],
    }
    FakeSession.loads = 0
    execute_pair([selection], tmp_path, FakeSession, resume=True)
    resumed = execute_pair([selection], tmp_path, FakeSession, resume=True)

    assert resumed["resumed"] == 1
    assert FakeSession.loads == 1


def test_resume_rejects_a_schema_record_from_another_commit(tmp_path, monkeypatch):
    import scripts.run_pair as runner

    selection = {
        "sample_index": 0,
        "scene": "scene-0",
        "context_indices": [0, 2],
        "target_indices": [1],
    }
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "provenance": {
                    "git_commit": "a" * 40,
                    "git_dirty": False,
                    "source_tree_sha256": "9" * 64,
                    "evaluation": {"kind": "sample", **selection},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "validate", lambda record: None)

    assert not runner._complete_sample(
        path,
        selection,
        {
            "git_commit": "b" * 40,
            "git_dirty": False,
            "source_tree_sha256": "8" * 64,
        },
    )


def test_pair_worker_records_sample_error_before_propagating(tmp_path: Path):
    from scripts.run_pair import execute_pair

    class FailingSession:
        def run_sample(self, selection: dict, output_dir: Path) -> dict:
            raise RuntimeError("intentional failure")

    selection = {
        "sample_index": 0,
        "scene": "scene-0",
        "context_indices": [0, 2],
        "target_indices": [1],
    }
    try:
        execute_pair([selection], tmp_path, FailingSession, resume=False)
    except RuntimeError as exc:
        assert str(exc) == "intentional failure"
    else:
        raise AssertionError("pair worker did not propagate the sample failure")

    events = [
        json.loads(line)
        for line in (tmp_path / "progress.jsonl").read_text().splitlines()
    ]
    assert events[-1]["event"] == "sample_error"
    assert events[-1]["error_type"] == "RuntimeError"


def test_worstcase_retention_keeps_only_the_strongest_real_view(tmp_path: Path):
    from scripts.run_pair import update_worstcase_evidence

    def write_candidate(sample_index: int, fsdr_loss: float, saes_lpips: float):
        sample_dir = tmp_path / "samples" / f"sample_{sample_index:05d}"
        sample_dir.mkdir(parents=True)
        record = {
            "provenance": {
                "evaluation": {
                    "sample_index": sample_index,
                    "scene": f"scene-{sample_index}",
                    "target_indices": [7],
                }
            },
            "ablation": {
                "asic": {
                    "quality": {"quality_views": [
                        {
                            "target_index": 7,
                            "psnr_db": 30.0,
                            "ssim": 0.9,
                            "lpips": 0.1,
                        }
                    ]}
                },
                "asic_fsdr": {
                    "quality": {"quality_views": [
                        {
                            "target_index": 7,
                            "psnr_db": 30.0 - fsdr_loss,
                            "ssim": 0.89,
                            "lpips": 0.11,
                        }
                    ]}
                },
                "asic_saes": {
                    "quality": {"quality_views": [
                        {
                            "target_index": 7,
                            "psnr_db": 29.9,
                            "ssim": 0.88,
                            "lpips": 0.1 + saes_lpips,
                        }
                    ]}
                },
            },
        }
        result = sample_dir / "results.json"
        result.write_text(json.dumps(record), encoding="utf-8")
        for name in ("gt", "ablation_asic", "ablation_asic_fsdr", "ablation_asic_saes"):
            (sample_dir / f"{name}_00.png").write_bytes(
                f"{sample_index}:{name}".encode()
            )
        update_worstcase_evidence(tmp_path, result, record)
        return sample_dir

    first = write_candidate(0, fsdr_loss=0.2, saes_lpips=0.03)
    second = write_candidate(1, fsdr_loss=0.4, saes_lpips=0.01)

    fsdr = json.loads((tmp_path / "worstcase/fsdr/manifest.json").read_text())
    saes = json.loads((tmp_path / "worstcase/saes/manifest.json").read_text())
    assert fsdr["sample_index"] == 1
    assert fsdr["loss_metric"] == "psnr_loss_db"
    assert saes["sample_index"] == 0
    assert saes["loss_metric"] == "lpips_increase"
    assert all(item["sha256"] for item in fsdr["artifacts"].values())
    assert not list(first.glob("*.png"))
    assert not list(second.glob("*.png"))


def test_resume_reconstructs_worstcase_after_result_write_crash(
    tmp_path: Path, monkeypatch
):
    import scripts.run_pair as runner

    selection = {
        "sample_index": 0,
        "scene": "scene-0",
        "context_indices": [0, 2],
        "target_indices": [7],
    }
    sample_dir = tmp_path / "samples/sample_00000"
    sample_dir.mkdir(parents=True)
    quality = {
        "psnr_db": 30.0,
        "ssim": 0.9,
        "lpips": 0.1,
        "target_index": 7,
    }
    record = {
        **selection,
        "provenance": {"evaluation": selection},
        "ablation": {
            "asic": {"quality": {"quality_views": [quality]}},
            "asic_fsdr": {
                "quality": {"quality_views": [{**quality, "psnr_db": 29.6}]}
            },
            "asic_saes": {
                "quality": {"quality_views": [{**quality, "lpips": 0.13}]}
            },
        },
    }
    (sample_dir / "results.json").write_text(json.dumps(record), encoding="utf-8")
    for name in ("gt", "ablation_asic", "ablation_asic_fsdr", "ablation_asic_saes"):
        (sample_dir / f"{name}_00.png").write_bytes(name.encode())

    monkeypatch.setattr(runner, "_complete_sample", lambda *args: True)
    FakeSession.loads = 0
    result = runner.execute_pair([selection], tmp_path, FakeSession, resume=True)

    assert result["executed"] == 0
    assert result["resumed"] == 1
    assert FakeSession.loads == 0
    assert (tmp_path / "worstcase/fsdr/manifest.json").is_file()
    assert (tmp_path / "worstcase/saes/manifest.json").is_file()
    assert not list(sample_dir.glob("*.png"))
