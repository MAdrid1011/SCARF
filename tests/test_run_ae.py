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


def test_quality_dry_run_plans_all_pairs_independently_of_current_evidence(tmp_path):
    plan = dry_run(tmp_path, "quality", "--num-samples", "2")
    assert plan["mode"] == "quality"
    assert len(plan["experiments"]) == 9
    assert len(plan["dataset_commands"]) == 4
    assert plan["software_claim_scope"] == {
        "status": "ACTIVE",
        "pair_count": 9,
        "diagnostic_results_are_claim_evidence": False,
    }


def test_pair_filter_runs_only_requested_protocol_pairs(tmp_path):
    plan = dry_run(
        tmp_path,
        "mechanisms",
        "--num-samples",
        "1",
        "--pairs",
        "transplat/re10k,mvsplat/acid,depthsplat/re10k",
    )

    assert [(item["model"], item["dataset"]) for item in plan["experiments"]] == [
        ("transplat", "re10k"),
        ("mvsplat", "acid"),
        ("depthsplat", "re10k"),
    ]
    assert len(plan["dataset_commands"]) == 2
    assert plan["requested_pairs"] == [
        "transplat/re10k",
        "mvsplat/acid",
        "depthsplat/re10k",
    ]


def test_pair_filter_rejects_unknown_duplicate_or_nonsoftware_pairs(tmp_path):
    for mode, value in (
        ("quality", "transplat/unknown"),
        ("quality", "transplat/re10k,transplat/re10k"),
        ("rtl", "transplat/re10k"),
    ):
        result = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                mode,
                "--dry-run",
                "--output-root",
                str(tmp_path),
                "--pairs",
                value,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0


def test_reviewer_profile_uses_frozen_hash_selected_protocol(tmp_path):
    plan = dry_run(
        tmp_path,
        "all-eval",
        "--profile",
        "reviewer",
        "--pairs",
        "transplat/re10k",
    )

    assert plan["evidence_profile"] == "reviewer"
    assert {item["sample_count"] for item in plan["experiments"]} == {512}
    assert all(
        "artifact/protocol/reviewer/re10k.json" in item["evaluation_index"]
        for item in plan["experiments"]
    )
    assert all(len(item["sample_selection_sha256"]) == 64 for item in plan["experiments"])
    all_commands = [
        command
        for item in plan["experiments"]
        for command in item["commands"]
    ] + plan["dataset_commands"]
    assert all(
        "downloads/calibration" not in " ".join(command)
        for command in all_commands
    )


def test_full_profile_preserves_the_complete_upstream_protocol(tmp_path):
    plan = dry_run(
        tmp_path,
        "quality",
        "--profile",
        "full",
        "--pairs",
        "transplat/re10k",
    )

    assert plan["evidence_profile"] == "full"
    assert plan["experiments"][0]["sample_count"] == 6474
    assert plan["experiments"][0]["evaluation_index"].endswith(
        "transplat/assets/evaluation_index_re10k.json"
    )


def test_pilot_all_expands_to_every_pair_with_one_sample(tmp_path):
    plan = dry_run(tmp_path, "pilot", "--pairs", "all")

    assert len(plan["experiments"]) == 9
    assert {item["sample_count"] for item in plan["experiments"]} == {1}
    assert {item["workflow"] for item in plan["experiments"]} == {"mechanisms"}
    assert plan["evidence_profile"] == "pilot"
    assert all(
        "--diagnostic-run" in item["command"]
        and "--claim-run" not in item["command"]
        for item in plan["experiments"]
    )


def test_calibrate_mode_is_bound_to_the_public_contract(tmp_path):
    plan = dry_run(tmp_path, "calibrate")
    commands = plan["commands"]

    assert plan["evidence_profile"] == "calibration"
    assert len(commands) == 2
    assert commands[0][1].endswith("scripts/calibration_sweep.py")
    assert "--manifest" in commands[0]
    assert commands[0][commands[0].index("--manifest") + 1].endswith(
        "outputs/calibration/dl3dv-protocol/manifest.json"
    )
    assert commands[1][1].endswith("scripts/calibrate_mechanisms.py")
    assert "artifact/CALIBRATION.md" in plan["calibration_contract"]
    assert "expected_results" not in " ".join(
        argument for command in commands for argument in command
    )


def test_calibrate_mode_uses_the_locked_classic_profile(tmp_path, monkeypatch):
    import scripts.run_ae as runner

    classic = "/opt/scarf/classic/bin/python"
    monkeypatch.setenv("SCARF_PYTHON_CLASSIC", classic)
    plan = runner.build_plan(
        SimpleNamespace(
            mode="calibrate",
            output_root=tmp_path,
            python=None,
            num_samples=None,
            profile="full",
            pairs=None,
            device="auto",
            figures="all",
            require_key_results=False,
            allow_low_memory_attempt=False,
            calibration_root=runner.DEFAULT_CALIBRATION_ROOT,
        )
    )

    assert plan["calibration_python"] == classic
    assert all(command[0] == classic for command in plan["commands"])


def test_pair_filter_intersects_each_composite_workflow(tmp_path, monkeypatch):
    import scripts.run_ae as runner

    status = runner.load_claim_status()
    status["software_pairs"]["transplat/dl3dv"] = "CLAIMED"
    monkeypatch.setattr(runner, "load_claim_status", lambda: status)
    all_plan = runner.build_plan(
        SimpleNamespace(
            mode="all",
            output_root=tmp_path / "all",
            python=None,
            num_samples=1,
            pairs="transplat/dl3dv",
        )
    )
    eval_plan = dry_run(
        tmp_path / "all-eval",
        "all-eval",
        "--num-samples",
        "1",
        "--pairs",
        "transplat/re10k",
    )

    assert [item["workflow"] for item in all_plan["experiments"]] == ["quality"]
    assert [item["workflow"] for item in eval_plan["experiments"]] == [
        "quality",
        "performance",
        "mechanisms",
        "utilization",
    ]


def test_quality_plan_contains_all_dataset_aware_commands_when_claimed(
    tmp_path, monkeypatch
):
    import scripts.run_ae as runner

    status = runner.load_claim_status()
    for pair in status["software_pairs"]:
        if not pair.endswith("/dl3dv"):
            status["software_pairs"][pair] = "CLAIMED"
    monkeypatch.setattr(runner, "load_claim_status", lambda: status)
    plan = runner.build_plan(
        type(
            "Args",
            (),
            {
                "mode": "quality",
                "output_root": tmp_path,
                "python": None,
                "num_samples": 2,
            },
        )()
    )

    assert plan["software_claim_scope"]["status"] == "ACTIVE"
    assert len(plan["experiments"]) == 9
    pairs = {(item["model"], item["dataset"]) for item in plan["experiments"]}
    assert len(pairs) == 9
    assert {dataset for _, dataset in pairs} == {"re10k", "acid", "dl3dv"}
    assert len(plan["dataset_commands"]) == 4
    assert {
        command[command.index("--representation") + 1]
        for command in plan["dataset_commands"]
    } == {
        "re10k-native",
        "acid-native",
        "re10k-compatible-360x640-v1",
        "depthsplat-native-270x480-v1",
    }
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


def test_fsdr_plan_runs_six_claim_pairs_without_saes_or_rendering(tmp_path):
    plan = dry_run(tmp_path, "fsdr", "--num-samples", "1")

    assert len(plan["experiments"]) == 6
    assert {(item["model"], item["dataset"]) for item in plan["experiments"]} == {
        (model, dataset)
        for model in ("transplat", "mvsplat", "depthsplat")
        for dataset in ("re10k", "acid")
    }
    for item in plan["experiments"]:
        demo = item["command"][item["command"].index("--") + 1 :]
        assert "--claim-run" in demo
        assert "--fsdr-only" in demo
        assert demo[demo.index("--image-output-policy") + 1] == "none"
        assert "aggregate_fsdr.py" in item["aggregate_command"][1]
        assert "--require-match" not in item["aggregate_command"]


def test_full_fsdr_plan_remains_isolated_from_paper_targets(tmp_path):
    import scripts.run_ae as runner

    plan = runner.build_plan(
        type(
            "Args",
            (),
            {
                "mode": "fsdr",
                "output_root": tmp_path,
                "python": None,
                "num_samples": None,
            },
        )()
    )
    assert len(plan["experiments"]) == 6
    assert all("--require-match" not in item["aggregate_command"] for item in plan["experiments"])
    assert all("--expected-results" not in item["aggregate_command"] for item in plan["experiments"])


def test_all_reuses_claimed_quality_runs_for_embedded_ablation_and_validation(
    tmp_path, monkeypatch
):
    import scripts.run_ae as runner

    status = runner.load_claim_status()
    for pair in status["software_pairs"]:
        status["software_pairs"][pair] = "CLAIMED"
    monkeypatch.setattr(runner, "load_claim_status", lambda: status)
    plan = runner.build_plan(
        SimpleNamespace(
            mode="all",
            output_root=tmp_path,
            python=None,
            num_samples=2,
        )
    )

    assert len(plan["experiments"]) == 15
    assert {item["workflow"] for item in plan["experiments"]} == {"quality", "fsdr"}
    assert len(plan["dataset_commands"]) == 4
    assert plan["software_claim_scope"]["status"] == "ACTIVE"
    command_text = [" ".join(command) for command in plan["commands"]]
    assert any("hardware/dram/run.sh" in command for command in command_text)
    assert command_text[-1].endswith(f"validate_ae.py --input {tmp_path}")


def test_all_skips_unclaimed_software_but_executes_nonmodel_steps(
    tmp_path, monkeypatch
):
    import scripts.run_ae as runner

    status = runner.load_claim_status()
    for pair in status["software_pairs"]:
        status["software_pairs"][pair] = "NOT_CLAIMED_SAES_SPARSE_QUALITY_MISMATCH"
    monkeypatch.setattr(runner, "load_claim_status", lambda: status)
    plan = runner.build_plan(
        SimpleNamespace(
            mode="all",
            output_root=tmp_path,
            python=None,
            num_samples=1,
        )
    )

    assert plan["experiments"] == []
    assert plan["dataset_commands"] == []
    assert plan["software_claim_scope"] == {
        "status": "NO_CLAIMED_PAIRS",
        "pair_count": 0,
        "diagnostic_results_are_claim_evidence": False,
    }
    command_text = [" ".join(command) for command in plan["commands"]]
    assert any("scripts/run_rtl.sh" in command for command in command_text)
    assert any("hardware/dram/run.sh" in command for command in command_text)
    assert any("generate_report.py" in command for command in command_text)
    assert any("validate_ae.py" in command for command in command_text)
    assert not any(
        marker in command
        for command in command_text
        for marker in ("demo.py", "run_pair.py", "verify_prepared_dataset.py")
    )

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.execute_plan(plan, tmp_path) == 0
    assert calls == plan["commands"]


def test_dram_mode_uses_only_the_labeled_smoke_vector(tmp_path):
    plan = dry_run(tmp_path, "dram")

    assert not plan["experiments"]
    command = plan["commands"][0]
    assert "hardware/dram/run.sh" in " ".join(command)
    assert command[command.index("--events") + 1].endswith("scarf_smoke_events.jsonl")


def test_physical_low_memory_attempt_is_explicit_and_recorded(tmp_path):
    plan = dry_run(tmp_path, "physical", "--allow-low-memory-attempt")

    assert plan["allow_low_memory_attempt"] is True
    assert "--allow-low-memory-attempt" in plan["commands"][0]


def test_low_memory_attempt_is_rejected_for_nonphysical_modes(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "quick",
            "--dry-run",
            "--output-root",
            str(tmp_path),
            "--allow-low-memory-attempt",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0


def test_dry_run_does_not_create_output_tree(tmp_path):
    dry_run(tmp_path / "not-created", "quality", "--num-samples", "1")
    assert not (tmp_path / "not-created").exists()


def test_full_mode_uses_recovered_executable_sample_counts(tmp_path, monkeypatch):
    import scripts.run_ae as runner

    status = runner.load_claim_status()
    for pair in status["software_pairs"]:
        if not pair.endswith("/dl3dv"):
            status["software_pairs"][pair] = "CLAIMED"
    monkeypatch.setattr(runner, "load_claim_status", lambda: status)
    plan = runner.build_plan(
        type(
            "Args",
            (),
            {
                "mode": "quality",
                "output_root": tmp_path,
                "python": None,
                "num_samples": None,
            },
        )()
    )
    counts = {
        item["dataset"]: item["sample_count"] for item in plan["experiments"]
    }
    assert counts == {"re10k": 6474, "acid": 1595, "dl3dv": 140}


def test_awaiting_independent_orin_is_not_locally_planned_but_sensitivity_remains_executable(tmp_path):
    orin = dry_run(tmp_path / "orin", "orin", "--num-samples", "1")
    sensitivity = dry_run(tmp_path / "sensitivity", "sensitivity", "--num-samples", "1")

    assert not orin["experiments"]
    assert orin["claim_status"]["figure8"] == (
        "CLAIMED_AWAITING_INDEPENDENT_ORIN_EVALUATION"
    )
    assert len(sensitivity["commands"]) == 1
    assert sensitivity["commands"][0][1].endswith("scripts/sensitivity_sweep.py")
    command = sensitivity["commands"][0]
    assert command[command.index("--profile") + 1] == "full"
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
