import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "scripts" / "demo.py"


def test_demo_help_exposes_public_ae_arguments():
    result = subprocess.run(
        [sys.executable, str(DEMO), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for option in (
        "--dataset",
        "--checkpoint",
        "--device",
        "--seed",
        "--functional-run",
        "--image-output-policy",
    ):
        assert option in result.stdout


def test_zero_or_missing_cycles_are_rejected():
    from scripts.result_record import require_positive_cycles

    with pytest.raises(ValueError, match="feature"):
        require_positive_cycles({"feature": 0, "depth": 4, "gaussian": 5, "ggu": 6})
    with pytest.raises(ValueError, match="depth"):
        require_positive_cycles({"feature": 3, "gaussian": 5, "ggu": 6})


def test_demo_source_contains_no_orin_or_tsmc_heuristics():
    source = DEMO.read_text(encoding="utf-8")
    assert "RTX3060_FP32_TFLOPS" not in source
    assert "Using reference S1 feature cycles" not in source
    assert "28nm ASIC Power & Area Estimation" not in source
    assert "Typical for TranSplat (fallback)" not in source
    assert "approximate de-duplication" not in source
    assert "S2 cost-volume cycle breakdown is missing or inconsistent" in source
    assert 'timing_repetitions = 5 if is_orin else 1' in source
    assert "depth_predictor_sim.set_strict_mode(strict_run)" in source


def test_claim_pipeline_preserves_reference_numerics_while_counting_cycles():
    source = DEMO.read_text(encoding="utf-8")

    assert "set_accurate_mode(False)" not in source
    assert "set_accurate_mode(True)" in source
    assert "capture_torch_rng_state" in source
    assert "restore_torch_rng_state" in source


def test_torch_rng_state_can_replay_a_probabilistic_sample():
    torch = pytest.importorskip("torch")

    from scripts.reproducibility import (
        capture_torch_rng_state,
        restore_torch_rng_state,
    )

    torch.manual_seed(37)
    state = capture_torch_rng_state()
    first = torch.rand(8)
    restore_torch_rng_state(state)
    second = torch.rand(8)

    assert torch.equal(first, second)


def test_transplat_camera_encoding_matches_encoder_formula():
    torch = pytest.importorskip("torch")

    from feature_extractor.transplat_extractor import transplat_image_to_world

    extrinsics = torch.eye(4).reshape(1, 1, 4, 4)
    extrinsics[..., 0, 3] = 2.0
    intrinsics = torch.tensor(
        [[[[0.5, 0.0, 0.5], [0.0, 0.25, 0.5], [0.0, 0.0, 1.0]]]]
    )
    actual = transplat_image_to_world(extrinsics, intrinsics, 100, 200)

    pixels = torch.eye(4).reshape(1, 1, 4, 4)
    pixels[:, :, :3, :3] = intrinsics
    pixels[:, :, 0, :] *= 200
    pixels[:, :, 1, :] *= 100
    expected = extrinsics @ torch.linalg.inv(pixels)

    assert torch.allclose(actual, expected)


def test_depthsplat_s3_cycles_are_traced_from_executed_modules():
    source = DEMO.read_text(encoding="utf-8")

    assert "run_module_with_cycle_trace" in source
    assert "feature_upsampler_trace.total_cycles" in source
    assert "gaussian_regressor_trace.total_cycles" in source
    assert "gaussian_head_trace.total_cycles" in source


def test_strict_stage_errors_preserve_the_original_failure():
    from scripts.result_record import strict_stage_error

    error = strict_stage_error("depth simulator", IndexError("missing scale 1"))

    assert isinstance(error, RuntimeError)
    assert str(error) == (
        "strict run depth simulator failed; GPU fallback is forbidden: "
        "IndexError: missing scale 1"
    )


@pytest.mark.parametrize(
    "disabled",
    ("--no-feature", "--no-depth", "--no-gaussian", "--no-saes", "--no-fsdr"),
)
def test_claim_run_rejects_disabled_or_fallback_stages(disabled):
    from scripts.demo_cli import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--claim-run", "--evaluation-index", "index.json", disabled])


def test_functional_run_has_the_same_strict_stage_contract():
    from scripts.demo_cli import parse_args

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--functional-run",
                "--evaluation-index",
                "index.json",
                "--baseline-only",
            ]
        )


def test_claim_run_rejects_dense_saes_diagnostic():
    from scripts.demo_cli import parse_args

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--claim-run",
                "--evaluation-index",
                "index.json",
                "--saes-materialization",
                "dense-diagnostic",
            ]
        )
