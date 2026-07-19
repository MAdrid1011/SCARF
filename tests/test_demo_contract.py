import ast
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "scripts" / "demo.py"


def assignment_consensus_boundary() -> tuple[dict, ast.FunctionDef]:
    tree = ast.parse(DEMO.read_text(encoding="utf-8"))
    names = {
        "ASSIGNMENT_CONSENSUS_PSEUDO_DESCRIPTOR_MATERIALIZATION",
        "_reject_assignment_consensus_rendering",
    }
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
        )
        or (isinstance(node, ast.FunctionDef) and node.name in names)
    ]
    namespace: dict = {}
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(DEMO), "exec"), namespace)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    return namespace, main


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
        "--diagnostic-run",
        "--image-output-policy",
        "--calibration-parameters",
        "--fsdr-only",
        "--fsdr-feature-source",
        "--saes-feature-source",
        "--saes-routing-audit",
        "--saes-hardware-audit",
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


def test_claim_run_fails_before_model_or_dataset_work_without_a_timing_backend():
    source = DEMO.read_text(encoding="utf-8")

    gate = source.index("if args.claim_run:\n        require_claim_timing_backend()")
    calibration = source.index("require_calibrated_mechanism()")
    assert gate < calibration
    assert "source-bound RTL or gate-level timing evidence" in source


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


def test_diagnostic_run_is_strict_but_never_a_claim_run():
    from scripts.demo_cli import parse_args

    args = parse_args(["--diagnostic-run", "--evaluation-index", "index.json"])
    assert args.diagnostic_run is True
    assert args.claim_run is False
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--diagnostic-run",
                "--evaluation-index",
                "index.json",
                "--no-depth",
            ]
        )


def test_context_safety_guard_is_diagnostic_only():
    from scripts.demo_cli import parse_args

    diagnostic = parse_args(
        [
            "--diagnostic-run",
            "--evaluation-index",
            "index.json",
            "--context-safety-guard",
        ]
    )

    assert diagnostic.context_safety_guard is True
    for mode in ("--claim-run", "--functional-run"):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    mode,
                    "--evaluation-index",
                    "index.json",
                    "--context-safety-guard",
                ]
            )
    with pytest.raises(SystemExit):
        parse_args(["--context-safety-guard"])


def test_context_safety_guard_reaches_step4b_saes_path():
    source = DEMO.read_text(encoding="utf-8")
    step4b = source.split("# ---- Step 4b: SAES v4", maxsplit=1)[1]
    step4b = step4b.split("# ---- Step 4c: FSDR", maxsplit=1)[0]

    assert "context_safety_guard=saes_execution_route['context_safety_guard']" in step4b


def test_context_safety_guard_rejects_pre_step4b_sensitivity_trace():
    from scripts.demo_cli import parse_args

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--diagnostic-run",
                "--evaluation-index",
                "index.json",
                "--sensitivity-trace",
                "--context-safety-guard",
            ]
        )


def test_strict_saes_route_uses_the_frozen_execution_identity():
    from scripts.demo_cli import resolve_frozen_saes_execution
    from scripts.saes_execution_identity import build_saes_execution_identity

    identity = build_saes_execution_identity()
    args = SimpleNamespace(
        claim_run=False,
        functional_run=True,
        tune_thresholds=False,
        saes_fv=None,
        saes_ds=None,
        saes_cc=None,
        tile_size=None,
        saes_materialization="representative",
        saes_feature_source="pipeline",
        saes_decision_semantics="current",
        context_safety_guard=False,
    )
    route = resolve_frozen_saes_execution(
        args,
        tile_size=identity["tile_size"],
        feature_threshold=identity["feature_threshold"],
        depth_threshold=identity["depth_threshold"],
        cross_check_threshold=identity["cross_check_threshold"],
        execution_identity=identity,
    )

    assert route["decision_semantics"] == "probe-normalized-std-first-hit"
    assert route["depth_routing_semantics"] == identity["depth_routing_semantics"]
    assert route["materialization_guard"] is True
    assert route["context_safety_guard"] is True
    assert route["require_deletion_certificate"] is True
    assert route["cross_check_threshold"] == identity["cross_check_threshold"]
    assert route["execution_identity"] == identity
    assert route["route_sha256"] == identity["route_sha256"]

    effective_values = {
        "tile_size": identity["tile_size"],
        "feature_threshold": identity["feature_threshold"],
        "depth_threshold": identity["depth_threshold"],
        "cross_check_threshold": identity["cross_check_threshold"],
    }
    for field, drifted_value in (
        ("tile_size", identity["tile_size"] + 1),
        ("feature_threshold", identity["feature_threshold"] + 0.1),
        ("depth_threshold", identity["depth_threshold"] + 0.1),
        ("cross_check_threshold", identity["cross_check_threshold"] + 0.01),
    ):
        with pytest.raises(ValueError, match=field):
            resolve_frozen_saes_execution(
                args,
                **{**effective_values, field: drifted_value},
                execution_identity=identity,
            )


def test_frozen_saes_route_binds_identity_and_permits_only_no_fsdr():
    from scripts.demo_cli import parse_args, resolve_frozen_saes_execution
    from scripts.saes_execution_identity import build_saes_execution_identity

    identity = build_saes_execution_identity()
    args = parse_args(
        [
            "--frozen-saes-route",
            "--evaluation-index",
            "index.json",
            "--no-fsdr",
        ]
    )
    assert args.frozen_saes_route is True
    assert args.no_fsdr is True
    route = resolve_frozen_saes_execution(
        args,
        tile_size=identity["tile_size"],
        feature_threshold=identity["feature_threshold"],
        depth_threshold=identity["depth_threshold"],
        cross_check_threshold=identity["cross_check_threshold"],
        execution_identity=identity,
    )
    assert route["execution_identity"] == identity
    assert route["context_safety_guard"] is True
    assert route["decision_semantics"] == "probe-normalized-std-first-hit"

    for extra in (
        ("--no-feature",),
        ("--no-depth",),
        ("--no-gaussian",),
        ("--no-saes",),
        ("--baseline-only",),
        ("--saes-materialization", "dense-diagnostic"),
        ("--saes-decision-semantics", "probe-vector-first-hit"),
        ("--context-safety-guard",),
        ("--diagnostic-run",),
    ):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "--frozen-saes-route",
                    "--evaluation-index",
                    "index.json",
                    *extra,
                ]
            )


def test_diagnostic_saes_route_preserves_cli_runtime_values():
    from scripts.demo_cli import resolve_frozen_saes_execution

    args = SimpleNamespace(
        claim_run=False,
        functional_run=False,
        saes_materialization="dense-diagnostic",
        saes_feature_source="gaussian-head-input",
        saes_decision_semantics="probe-vector-first-hit",
        context_safety_guard=True,
    )
    route = resolve_frozen_saes_execution(
        args,
        tile_size=8,
        feature_threshold=0.3,
        depth_threshold=0.05,
        cross_check_threshold=0.4,
        execution_identity=None,
    )

    assert route["tile_size"] == 8
    assert route["feature_threshold"] == 0.3
    assert route["depth_threshold"] == 0.05
    assert route["cross_check_threshold"] == 0.4
    assert route["materialization"] == "dense-diagnostic"
    assert route["decision_semantics"] == "probe-vector-first-hit"
    assert route["context_safety_guard"] is True
    assert route["require_deletion_certificate"] is True
    assert route["execution_identity"] is None
    assert route["route_sha256"] is None


@pytest.mark.parametrize(
    "override",
    (
        ("--tune-thresholds",),
        ("--saes-fv", "0.2"),
        ("--saes-ds", "0.1"),
        ("--saes-cc", "0.015"),
        ("--tile-size", "4"),
    ),
)
def test_strict_runs_reject_explicit_frozen_saes_route_overrides(override):
    from scripts.demo_cli import parse_args

    for mode in ("--claim-run", "--functional-run"):
        with pytest.raises(SystemExit):
            parse_args([mode, "--evaluation-index", "index.json", *override])


@pytest.mark.parametrize(
    "audit_args",
    (
        ("--saes-s3-raw-audit",),
        (
            "--saes-routing-audit",
            "--saes-decision-semantics",
            "probe-normalized-std-first-hit",
        ),
        ("--saes-hardware-audit",),
    ),
)
def test_protocol_locked_saes_audits_reject_context_safety_guard(audit_args):
    from scripts.demo_cli import parse_args

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--diagnostic-run",
                "--evaluation-index",
                "index.json",
                "--image-output-policy",
                "none",
                *audit_args,
                "--context-safety-guard",
            ]
        )


def test_demo_step4b_uses_the_resolved_frozen_saes_route():
    source = DEMO.read_text(encoding="utf-8")
    step4b = source.split("# ---- Step 4b: SAES v4", maxsplit=1)[1]
    step4b = step4b.split("# ---- Step 4c: FSDR", maxsplit=1)[0]

    for field in (
        "tile_size",
        "feature_threshold",
        "depth_threshold",
        "cross_check_threshold",
        "materialization",
        "decision_semantics",
        "depth_routing_semantics",
        "materialization_guard",
        "context_safety_guard",
        "require_deletion_certificate",
    ):
        if field == "require_deletion_certificate":
            assert "require_deletion_certificate=saes_execution_route[" in step4b
        else:
            assert f"saes_execution_route['{field}']" in step4b
    assert "validate_frozen_saes_execution_stats(saes_stats, saes_execution_route)" in step4b
    assert "saes_stats['saes_execution_identity']" in step4b
    assert "saes_stats['route_sha256']" in step4b


@pytest.mark.parametrize(
    "extra",
    (
        ("--tune-thresholds",),
        ("--saes-fv", "0.1"),
        ("--fsdr-cache-size", "64"),
        ("--fsdr-guidance-policy", "paper-hamming"),
        ("--seed", "1"),
    ),
)
def test_calibration_trace_rejects_unregistered_overrides(extra):
    from scripts.demo_cli import parse_args

    base = [
        "--diagnostic-run",
        "--evaluation-index",
        "index.json",
        "--calibration-trace",
        "--image-output-policy",
        "none",
    ]
    with pytest.raises(SystemExit):
        parse_args([*base, *extra])


def test_calibration_holdout_accepts_only_one_complete_registered_tuple():
    from scripts.demo_cli import parse_args

    args = parse_args(
        [
            "--diagnostic-run",
            "--evaluation-index",
            "index.json",
            "--calibration-trace",
            "--image-output-policy",
            "none",
            "--calibration-parameters",
            '{"gamma_depth":0.05,"beta_x":0.25,"beta_f":0.05,"beta_d":0.5}',
        ]
    )
    assert args.calibration_parameters == {
        "gamma_depth": 0.05,
        "beta_x": 0.25,
        "beta_f": 0.05,
        "beta_d": 0.5,
    }
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--diagnostic-run",
                "--evaluation-index",
                "index.json",
                "--calibration-parameters",
                '{"gamma_depth":0.05}',
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


def test_assignment_consensus_is_rejected_before_demo_pipeline():
    namespace, main = assignment_consensus_boundary()
    candidate = namespace["ASSIGNMENT_CONSENSUS_PSEUDO_DESCRIPTOR_MATERIALIZATION"]
    reject = namespace["_reject_assignment_consensus_rendering"]

    with pytest.raises(RuntimeError, match="cannot render or compute quality metrics"):
        reject(candidate)

    parse_index = next(
        index
        for index, statement in enumerate(main.body)
        if isinstance(statement, ast.Assign)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "parse_demo_args"
    )
    guard_index = next(
        index
        for index, statement in enumerate(main.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "_reject_assignment_consensus_rendering"
    )
    strict_run_index = next(
        index
        for index, statement in enumerate(main.body)
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "strict_run"
            for target in statement.targets
        )
    )
    assert parse_index < guard_index < strict_run_index


def test_assignment_consensus_guard_preserves_public_materializations():
    from scripts.demo_cli import build_parser, parse_args

    namespace, _ = assignment_consensus_boundary()
    candidate = namespace["ASSIGNMENT_CONSENSUS_PSEUDO_DESCRIPTOR_MATERIALIZATION"]
    reject = namespace["_reject_assignment_consensus_rendering"]
    action = next(
        item
        for item in build_parser()._actions
        if item.dest == "saes_materialization"
    )
    assert candidate not in action.choices
    with pytest.raises(SystemExit):
        parse_args(["--saes-materialization", candidate])
    for materialization in action.choices:
        assert reject(materialization) is None


def test_probe_spread_materialization_is_non_claiming_only():
    from scripts.demo_cli import parse_args

    args = parse_args(["--saes-materialization", "probe-spread-diagnostic"])
    assert args.saes_materialization == "probe-spread-diagnostic"

    for strict_mode in ("--claim-run", "--functional-run"):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    strict_mode,
                    "--evaluation-index",
                    "index.json",
                    "--saes-materialization",
                    "probe-spread-diagnostic",
                ]
            )

    diagnostic = parse_args(
        [
            "--diagnostic-run",
            "--evaluation-index",
            "index.json",
            "--saes-materialization",
            "probe-spread-diagnostic",
        ]
    )
    assert diagnostic.diagnostic_run is True
    assert diagnostic.claim_run is False
    assert diagnostic.saes_materialization == "probe-spread-diagnostic"


def test_conditional_anchor_transport_is_diagnostic_only():
    from scripts.demo_cli import parse_args

    materialization = "conditional-anchor-transport-diagnostic"
    args = parse_args(["--saes-materialization", materialization])
    assert args.saes_materialization == materialization
    for strict_mode in ("--claim-run", "--functional-run"):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    strict_mode,
                    "--evaluation-index",
                    "index.json",
                    "--saes-materialization",
                    materialization,
                ]
            )


def test_conditional_optical_mass_is_diagnostic_only():
    from scripts.demo_cli import parse_args

    materialization = "conditional-optical-mass-diagnostic"
    args = parse_args(["--saes-materialization", materialization])
    assert args.saes_materialization == materialization
    for strict_mode in ("--claim-run", "--functional-run"):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    strict_mode,
                    "--evaluation-index",
                    "index.json",
                    "--saes-materialization",
                    materialization,
                ]
            )


def test_conditional_projected_optical_mass_is_diagnostic_only():
    from scripts.demo_cli import parse_args

    materialization = "conditional-projected-optical-mass-diagnostic"
    args = parse_args(["--saes-materialization", materialization])
    assert args.saes_materialization == materialization
    for strict_mode in ("--claim-run", "--functional-run"):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    strict_mode,
                    "--evaluation-index",
                    "index.json",
                    "--saes-materialization",
                    materialization,
                ]
            )


def test_probe_spread_materialization_forces_non_claim_result():
    source = DEMO.read_text(encoding="utf-8")

    assert "args.saes_materialization != 'representative'" in source


def test_saes_hardware_audit_is_target_free_and_protocol_locked():
    from scripts.demo_cli import parse_args

    args = parse_args(
        [
            "--diagnostic-run",
            "--evaluation-index",
            "index.json",
            "--saes-hardware-audit",
            "--image-output-policy",
            "none",
        ]
    )
    assert args.saes_hardware_audit is True

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--diagnostic-run",
                "--evaluation-index",
                "index.json",
                "--saes-hardware-audit",
            ]
        )
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--claim-run",
                "--evaluation-index",
                "index.json",
                "--saes-hardware-audit",
                "--image-output-policy",
                "none",
            ]
        )


def test_transmittance_materialization_is_non_claiming_only():
    from scripts.demo_cli import parse_args

    args = parse_args(["--saes-materialization", "transmittance-diagnostic"])
    assert args.saes_materialization == "transmittance-diagnostic"

    for strict_mode in ("--claim-run", "--functional-run"):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    strict_mode,
                    "--evaluation-index",
                    "index.json",
                    "--saes-materialization",
                    "transmittance-diagnostic",
                ]
            )

    diagnostic = parse_args(
        [
            "--diagnostic-run",
            "--evaluation-index",
            "index.json",
            "--saes-materialization",
            "transmittance-diagnostic",
        ]
    )
    assert diagnostic.diagnostic_run is True
    assert diagnostic.saes_materialization == "transmittance-diagnostic"


def test_l1_primary_depth_reference_is_not_an_exposed_diagnostic_mode():
    from scripts.demo_cli import parse_args

    materialization = "l1-primary-depth-reference-diagnostic"
    with pytest.raises(SystemExit):
        parse_args(["--saes-materialization", materialization])


def test_virtual_reconstruction_materialization_is_non_claiming_only():
    from scripts.demo_cli import parse_args

    args = parse_args(
        ["--saes-materialization", "virtual-reconstruction-diagnostic"]
    )
    assert args.saes_materialization == "virtual-reconstruction-diagnostic"

    for strict_mode in ("--claim-run", "--functional-run"):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    strict_mode,
                    "--evaluation-index",
                    "index.json",
                    "--saes-materialization",
                    "virtual-reconstruction-diagnostic",
                ]
            )

    diagnostic = parse_args(
        [
            "--diagnostic-run",
            "--evaluation-index",
            "index.json",
            "--saes-materialization",
            "virtual-reconstruction-diagnostic",
        ]
    )
    assert diagnostic.diagnostic_run is True
    assert diagnostic.saes_materialization == "virtual-reconstruction-diagnostic"


def test_saes_diagnostic_sweep_is_non_claiming_only():
    from scripts.demo_cli import parse_args

    args = parse_args(["--saes-diagnostic-sweep"])
    assert args.saes_diagnostic_sweep is True

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--claim-run",
                "--evaluation-index",
                "index.json",
                "--saes-diagnostic-sweep",
            ]
        )


def test_probe_vector_first_hit_is_non_claiming_only():
    from scripts.demo_cli import parse_args

    args = parse_args(["--saes-decision-semantics", "probe-vector-first-hit"])
    assert args.saes_decision_semantics == "probe-vector-first-hit"

    for strict_mode in ("--claim-run", "--functional-run"):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    strict_mode,
                    "--evaluation-index",
                    "index.json",
                    "--saes-decision-semantics",
                    "probe-vector-first-hit",
                ]
            )


def test_normalized_probe_standard_deviation_is_non_claiming_only():
    from scripts.demo_cli import parse_args

    args = parse_args(
        ["--saes-decision-semantics", "probe-normalized-std-first-hit"]
    )
    assert args.saes_decision_semantics == "probe-normalized-std-first-hit"

    for strict_mode in ("--claim-run", "--functional-run"):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    strict_mode,
                    "--evaluation-index",
                    "index.json",
                    "--saes-decision-semantics",
                    "probe-normalized-std-first-hit",
                ]
            )


def test_materialization_attribute_audit_is_diagnostic_only_and_isolated():
    from scripts.demo_cli import parse_args

    args = parse_args(
        [
            "--diagnostic-run",
            "--evaluation-index",
            "index.json",
            "--saes-materialization-audit",
        ]
    )
    assert args.saes_materialization_audit is True

    with pytest.raises(SystemExit):
        parse_args(["--saes-materialization-audit"])
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--claim-run",
                "--evaluation-index",
                "index.json",
                "--saes-materialization-audit",
            ]
        )
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--diagnostic-run",
                "--evaluation-index",
                "index.json",
                "--saes-materialization-audit",
                "--saes-diagnostic-sweep",
            ]
        )


def test_s3_raw_audit_is_diagnostic_only_and_protocol_fixed():
    from scripts.demo_cli import parse_args

    command = [
        "--diagnostic-run",
        "--evaluation-index",
        "index.json",
        "--image-output-policy",
        "none",
        "--saes-s3-raw-audit",
    ]
    args = parse_args(command)
    assert args.saes_s3_raw_audit is True

    with pytest.raises(SystemExit):
        parse_args(["--saes-s3-raw-audit"])
    for strict_mode in ("--claim-run", "--functional-run"):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    strict_mode,
                    "--evaluation-index",
                    "index.json",
                    "--image-output-policy",
                    "none",
                    "--saes-s3-raw-audit",
                ]
            )
    with pytest.raises(SystemExit):
        parse_args([*command, "--saes-materialization-audit"])
    with pytest.raises(SystemExit):
        parse_args([*command, "--saes-feature-source", "gaussian-head-input"])
    with pytest.raises(SystemExit):
        parse_args([*command, "--saes-fv", "0.1"])
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--diagnostic-run",
                "--evaluation-index",
                "index.json",
                "--saes-s3-raw-audit",
            ]
        )


def test_saes_routing_audit_is_diagnostic_only_and_protocol_fixed():
    from scripts.demo_cli import parse_args

    command = [
        "--diagnostic-run",
        "--evaluation-index",
        "index.json",
        "--image-output-policy",
        "none",
        "--saes-decision-semantics",
        "probe-normalized-std-first-hit",
        "--saes-routing-audit",
    ]
    args = parse_args(command)
    assert args.saes_routing_audit is True

    with pytest.raises(SystemExit):
        parse_args(["--saes-routing-audit"])
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--diagnostic-run",
                "--evaluation-index",
                "index.json",
                "--image-output-policy",
                "none",
                "--saes-routing-audit",
            ]
        )
    for strict_mode in ("--claim-run", "--functional-run"):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    strict_mode,
                    "--evaluation-index",
                    "index.json",
                    "--image-output-policy",
                    "none",
                    "--saes-decision-semantics",
                    "probe-normalized-std-first-hit",
                    "--saes-routing-audit",
                ]
            )
    with pytest.raises(SystemExit):
        parse_args([*command, "--saes-s3-raw-audit"])
    with pytest.raises(SystemExit):
        parse_args([*command, "--saes-ds", "0.05"])


def test_historical_fsdr_depth_guard_is_non_claiming_only():
    from scripts.demo_cli import parse_args

    args = parse_args(["--fsdr-guidance-policy", "historical-depth-guard"])
    assert args.fsdr_guidance_policy == "historical-depth-guard"

    for strict_mode in ("--claim-run", "--functional-run"):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    strict_mode,
                    "--evaluation-index",
                    "index.json",
                    "--fsdr-guidance-policy",
                    "historical-depth-guard",
                ]
            )


def test_gaussian_head_saes_feature_source_is_non_claiming_only():
    from scripts.demo_cli import parse_args

    args = parse_args(["--saes-feature-source", "gaussian-head-input"])
    assert args.saes_feature_source == "gaussian-head-input"

    for strict_mode in ("--claim-run", "--functional-run"):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    strict_mode,
                    "--evaluation-index",
                    "index.json",
                    "--saes-feature-source",
                    "gaussian-head-input",
                ]
            )


def test_gaussian_head_saes_feature_source_forces_non_claim_result():
    source = DEMO.read_text(encoding="utf-8")

    assert "dp_output.saes_features" in source
    assert "args.saes_feature_source != 'pipeline'" in source


def test_gaussian_head_saes_diagnostic_does_not_replace_fsdr_features():
    source = DEMO.read_text(encoding="utf-8")
    fsdr_runtime = source.split(
        "# ---- Step 4c: FSDR (Feature-Similarity Gaussian Reuse)", maxsplit=1
    )[1]
    fsdr_runtime = fsdr_runtime.split(
        "# ---- Step 4d: Build 4 Ablation Gaussian Configs", maxsplit=1
    )[0]

    assert "has_features = (fsdr_features is not None" in fsdr_runtime
    assert "prepare_fsdr_candidate_frames(\n                fsdr_features," in fsdr_runtime
    assert "process_discrete_frame(" in fsdr_runtime


def test_gaussian_head_retention_boundary_uses_selected_saes_features():
    source = DEMO.read_text(encoding="utf-8")

    assert "tile_scores = probe_feature_variances(\n            saes_features," in source


def test_fsdr_only_is_an_isolated_claim_mode():
    from scripts.demo_cli import parse_args

    args = parse_args(
        ["--claim-run", "--evaluation-index", "index.json", "--fsdr-only"]
    )
    assert args.fsdr_only is True

    with pytest.raises(SystemExit):
        parse_args(["--fsdr-only"])
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--model",
                "depthsplat",
                "--claim-run",
                "--evaluation-index",
                "index.json",
                "--fsdr-only",
                "--fsdr-feature-source",
                "depthsplat-mono",
            ]
        )
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--claim-run",
                "--evaluation-index",
                "index.json",
                "--fsdr-only",
                "--ablation",
            ]
        )

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--saes-diagnostic-sweep",
                "--saes-materialization",
                "dense-diagnostic",
            ]
        )

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--functional-run",
                "--evaluation-index",
                "index.json",
                "--saes-diagnostic-sweep",
            ]
        )


def test_fsdr_only_is_a_target_rgb_free_execution_mode():
    import scripts.demo as demo

    common = {
        "fsdr_only": False,
        "saes_materialization_audit": False,
        "saes_s3_raw_audit": False,
        "saes_routing_audit": False,
        "saes_hardware_audit": False,
    }
    assert demo.is_target_rgb_free_execution(
        SimpleNamespace(**{**common, "fsdr_only": True})
    )
    assert not demo.is_target_rgb_free_execution(SimpleNamespace(**common))


def test_fsdr_only_diagnostic_mode_is_target_free_and_non_rendering():
    from scripts.demo_cli import parse_args

    args = parse_args(
        [
            "--diagnostic-run",
            "--evaluation-index",
            "index.json",
            "--fsdr-only",
            "--image-output-policy",
            "none",
        ]
    )
    assert args.diagnostic_run is True
    assert args.fsdr_only is True

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--diagnostic-run",
                "--evaluation-index",
                "index.json",
                "--fsdr-only",
            ]
        )


def test_probe_channel_variance_first_hit_is_non_claiming_only():
    from scripts.demo_cli import parse_args

    args = parse_args(
        ["--saes-decision-semantics", "probe-channel-variance-first-hit"]
    )
    assert args.saes_decision_semantics == "probe-channel-variance-first-hit"

    for strict_mode in ("--claim-run", "--functional-run"):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    strict_mode,
                    "--evaluation-index",
                    "index.json",
                    "--saes-decision-semantics",
                    "probe-channel-variance-first-hit",
                ]
            )
