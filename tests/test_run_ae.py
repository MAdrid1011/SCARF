import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_ae.py"


def dry_run(tmp_path: Path, mode: str, *extra: str):
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            mode,
            "--dry-run",
            "--output-root",
            str(tmp_path),
            *extra,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_quality_dry_run_contains_all_claimed_dataset_aware_commands(tmp_path):
    plan = dry_run(tmp_path, "quality", "--num-samples", "2")
    assert plan["mode"] == "quality"
    assert len(plan["experiments"]) == 6
    pairs = {(item["model"], item["dataset"]) for item in plan["experiments"]}
    assert len(pairs) == 6
    assert all(dataset != "dl3dv" for _, dataset in pairs)
    assert len(plan["dataset_commands"]) == 2
    assert {
        command[command.index("--representation") + 1]
        for command in plan["dataset_commands"]
    } == {"re10k-native", "acid-native"}
    for item in plan["experiments"]:
        command = item["command"]
        assert command[1].endswith("scripts/run_pair.py")
        assert command[command.index("--evaluation-index") + 1] == item["evaluation_index"]
        pair_separator = command.index("--")
        pair_dataset_root = command.index("--dataset-root")
        assert pair_dataset_root < pair_separator
        assert command[pair_dataset_root + 1] == item["dataset_root"]
        demo = command[command.index("--") + 1 :]
        assert command[0] == demo[0]
        assert demo[demo.index("--model") + 1] == item["model"]
        assert demo[demo.index("--dataset") + 1] == item["dataset"]
        assert demo[demo.index("--checkpoint") + 1].endswith(".ckpt")
        assert len(item["commands"]) == 1
        assert item["sample_count"] == 2
        assert item["result"].endswith(f"{item['model']}_{item['dataset']}/results.json")
        assert "aggregate_results.py" in item["aggregate_command"][1]


def test_quick_dry_run_is_small_and_re10k_based(tmp_path):
    plan = dry_run(tmp_path, "quick")
    assert len(plan["experiments"]) == 1
    item = plan["experiments"][0]
    assert (item["model"], item["dataset"]) == ("mvsplat", "re10k")
    assert "--num-samples" in item["command"]
    assert item["dataset_root"].endswith("datasets/quick-re10k")
    assert item["dataset_representation"] == "re10k-synthetic-functional-v1"
    assert "--functional-run" in item["command"]
    assert "--claim-run" not in item["command"]
    assert len(plan["dataset_commands"]) == 1
    dataset_command = plan["dataset_commands"][0]
    assert dataset_command[1].endswith("data/verify_prepared_dataset.py")
    assert dataset_command[dataset_command.index("--representation") + 1] == (
        "re10k-synthetic-functional-v1"
    )


def test_all_reuses_quality_runs_for_embedded_ablation_and_validation(tmp_path):
    plan = dry_run(tmp_path, "all", "--num-samples", "2")

    assert len(plan["experiments"]) == 6
    assert {item["workflow"] for item in plan["experiments"]} == {"quality"}
    assert sum(item["workflow"] == "quality" for item in plan["experiments"]) == 6
    assert len(plan["dataset_commands"]) == 2
    for item in plan["experiments"]:
        assert len(item["commands"]) == 1
        assert all("--ablation" not in command for command in item["commands"])
    command_text = [" ".join(command) for command in plan["commands"]]
    assert any("hardware/dram/run.sh" in command for command in command_text)
    assert command_text[-1].endswith(f"validate_ae.py --input {tmp_path}")


def test_dram_mode_uses_only_the_labeled_smoke_vector(tmp_path):
    plan = dry_run(tmp_path, "dram")

    assert not plan["experiments"]
    command = plan["commands"][0]
    assert "hardware/dram/run.sh" in " ".join(command)
    assert command[command.index("--events") + 1].endswith("scarf_smoke_events.jsonl")


def test_dry_run_does_not_create_output_tree(tmp_path):
    dry_run(tmp_path / "not-created", "quality", "--num-samples", "1")
    assert not (tmp_path / "not-created").exists()


def test_full_mode_uses_recovered_executable_sample_counts(tmp_path):
    plan = dry_run(tmp_path, "quality")
    counts = {
        item["dataset"]: item["sample_count"] for item in plan["experiments"]
    }
    assert counts == {"re10k": 6474, "acid": 1595}


def test_unavailable_orin_and_sensitivity_are_explicitly_not_claimed(tmp_path):
    orin = dry_run(tmp_path / "orin", "orin", "--num-samples", "1")
    sensitivity = dry_run(tmp_path / "sensitivity", "sensitivity", "--num-samples", "1")

    assert not orin["experiments"]
    assert orin["claim_status"]["figure8"].startswith("NOT_CLAIMED")
    assert not sensitivity["commands"]
    assert sensitivity["claim_status"]["sensitivity"].startswith("NOT_CLAIMED")


def test_all_skips_downgraded_physical_claims(tmp_path, monkeypatch):
    import scripts.run_ae as runner

    status = runner.load_claim_status()
    status["physical_asap7"] = "NOT_CLAIMED_RESOURCE_LIMIT"
    status["deepscale"] = "NOT_CLAIMED_NO_PHYSICAL_INPUT"
    monkeypatch.setattr(runner, "load_claim_status", lambda: status)
    plan = runner.build_plan(
        type(
            "Args",
            (),
            {
                "mode": "all",
                "output_root": tmp_path,
                "python": None,
                "num_samples": 1,
            },
        )()
    )

    commands = [" ".join(command) for command in plan["commands"]]
    assert not any("hardware/iflow/run.sh" in command for command in commands)
    assert not any("hardware/scaling/deepscale.py" in command for command in commands)
    assert "generate_report.py" in commands[-2]
    assert "validate_ae.py" in commands[-1]


def test_unknown_mode_fails():
    result = subprocess.run(
        [sys.executable, str(RUNNER), "unknown", "--dry-run"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_default_modes_share_one_output_root():
    result = subprocess.run(
        [sys.executable, str(RUNNER), "validate", "--dry-run"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["output_root"] == str(ROOT / "outputs/ae")


def test_profile_python_resolution_prefers_override_env_and_local_venv(
    tmp_path, monkeypatch
):
    import scripts.run_ae as runner

    local_python = tmp_path / ".venv/classic/bin/python"
    local_python.parent.mkdir(parents=True)
    local_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.delenv("SCARF_PYTHON_CLASSIC", raising=False)

    assert runner._python_for("classic", None) == str(local_python)
    monkeypatch.setenv("SCARF_PYTHON_CLASSIC", "/profiles/classic/python")
    assert runner._python_for("classic", None) == "/profiles/classic/python"
    assert runner._python_for("classic", "/override/python") == "/override/python"


def test_execute_plan_validates_each_profile_before_experiments(tmp_path, monkeypatch):
    import scripts.run_ae as runner

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    plan = {
        "schema_version": "1.0",
        "mode": "quality",
        "dataset_commands": [["/profiles/classic/python", "verify.py"]],
        "experiments": [
            {
                "environment_profile": "classic",
                "command": ["/profiles/classic/python", "worker.py"],
                "commands": [["/profiles/classic/python", "worker.py"]],
                "aggregate_command": ["python", "aggregate.py"],
            },
            {
                "environment_profile": "depthsplat",
                "command": ["/profiles/depthsplat/python", "worker.py"],
                "commands": [["/profiles/depthsplat/python", "worker.py"]],
                "aggregate_command": ["python", "aggregate.py"],
            },
        ],
    }

    assert runner.execute_plan(plan, tmp_path) == 0
    assert calls[0][:4] == [
        "/profiles/classic/python",
        str(runner.SCRIPT_DIR / "check_environment.py"),
        "--profile",
        "classic",
    ]
    assert calls[1][:4] == [
        "/profiles/depthsplat/python",
        str(runner.SCRIPT_DIR / "check_environment.py"),
        "--profile",
        "depthsplat",
    ]
    assert calls[2] == ["/profiles/classic/python", "verify.py"]
    assert calls[3] == ["/profiles/classic/python", "worker.py"]
