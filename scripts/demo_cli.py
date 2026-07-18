"""Command-line contract for the SCARF model experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.calibration_contract import canonical_parameters


MODELS = ("transplat", "mvsplat", "depthsplat")
DATASETS = ("re10k", "acid", "dl3dv")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def calibration_parameters(value: str) -> dict[str, float]:
    """Parse the one train-selected tuple allowed for a holdout trace."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            "must be a JSON object containing the complete registered tuple"
        ) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(
            "must be a JSON object containing the complete registered tuple"
        )
    try:
        return canonical_parameters(parsed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one SCARF AE experiment")
    parser.add_argument("--model", choices=MODELS, default="transplat")
    parser.add_argument("--dataset", choices=DATASETS, default="re10k")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--evaluation-index", type=Path)
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument("--claim-run", action="store_true")
    run_mode.add_argument("--functional-run", action="store_true")
    run_mode.add_argument(
        "--diagnostic-run",
        action="store_true",
        help="Use strict execution without making the result claim-eligible",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=positive_int, default=1)
    parser.add_argument("--sample-index", type=nonnegative_int, default=0)
    parser.add_argument(
        "--protocol-sample-index",
        type=nonnegative_int,
        help="Original source-index ordinal when null entries were filtered",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--image-output-policy",
        choices=("representative", "all", "none"),
        default="representative",
    )
    parser.add_argument("--freq", type=positive_int, default=1000)

    parser.add_argument("--no-feature", action="store_true")
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--no-gaussian", action="store_true")
    parser.add_argument("--no-saes", action="store_true")
    parser.add_argument("--no-fsdr", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--tune-thresholds", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--sensitivity-trace", action="store_true")
    parser.add_argument("--calibration-trace", action="store_true")
    parser.add_argument(
        "--calibration-parameters",
        type=calibration_parameters,
        help=(
            "One committed registered tuple for holdout validation; "
            "valid only with --calibration-trace"
        ),
    )
    parser.add_argument(
        "--fsdr-only",
        action="store_true",
        help="Emit target-free FSDR Table 2 evidence without SAES or rendering",
    )
    parser.add_argument(
        "--fsdr-feature-source",
        choices=("pipeline",),
        default="pipeline",
        help="Hash the executed cost-volume feature tensor",
    )
    parser.add_argument(
        "--fsdr-guidance-policy",
        choices=(
            "paper-hamming-local-validity",
            "paper-hamming",
            "historical-depth-guard",
        ),
        default="paper-hamming-local-validity",
        help="Select the claim guard or a non-claiming historical diagnostic",
    )
    parser.add_argument(
        "--saes-diagnostic-sweep",
        action="store_true",
        help="Run a non-claim decision-statistic and sparse-coverage sweep",
    )
    parser.add_argument(
        "--saes-materialization-audit",
        action="store_true",
        help="Run a target-free post-hoc Gaussian-attribute audit and stop before rendering",
    )
    parser.add_argument(
        "--saes-s3-raw-audit",
        action="store_true",
        help="Audit probe interpolation in raw S3 descriptor space before GGU conversion",
    )
    parser.add_argument(
        "--saes-routing-audit",
        action="store_true",
        help="Compare target-free S1/S2 probe-statistic units and stop before S4",
    )
    parser.add_argument(
        "--saes-hardware-audit",
        action="store_true",
        help="Record target-free SAES control, merge, and storage events without rendering",
    )
    parser.add_argument(
        "--saes-feature-source",
        choices=("pipeline", "gaussian-head-input"),
        default="pipeline",
        help="Select a non-claiming executed feature tensor for SAES diagnostics",
    )

    parser.add_argument("--saes-fv", type=float)
    parser.add_argument("--saes-ds", type=float)
    parser.add_argument("--saes-cc", type=float)
    parser.add_argument(
        "--saes-materialization",
        choices=(
            "representative",
            "dense-diagnostic",
            "probe-spread-diagnostic",
            "transmittance-diagnostic",
            "virtual-reconstruction-diagnostic",
            "conditional-anchor-transport-diagnostic",
            "conditional-optical-mass-diagnostic",
            "conditional-projected-optical-mass-diagnostic",
        ),
        default="representative",
        help="Use the paper-faithful sparse path or a non-claim materialization diagnostic",
    )
    parser.add_argument(
        "--saes-decision-semantics",
        choices=(
            "current",
            "probe-vector-first-hit",
            "probe-channel-variance-first-hit",
            "probe-normalized-std-first-hit",
        ),
        default="current",
        help="Select a non-claiming paper-variance first-hit decision diagnostic",
    )
    parser.add_argument("--tile-size", type=positive_int)
    parser.add_argument("--fsdr-cache-size", type=positive_int)
    parser.add_argument("--fsdr-hamming", type=nonnegative_int)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.sample_index >= args.num_samples:
        raise SystemExit("--sample-index must be smaller than --num-samples")
    strict_run = args.claim_run or args.functional_run or args.diagnostic_run
    claim_or_functional_run = args.claim_run or args.functional_run
    mode = (
        "--claim-run"
        if args.claim_run
        else "--functional-run" if args.functional_run else "--diagnostic-run"
    )
    if strict_run and args.evaluation_index is None:
        build_parser().error(f"{mode} requires --evaluation-index")
    disabled = [
        option
        for option, enabled in (
            ("--no-feature", args.no_feature),
            ("--no-depth", args.no_depth),
            ("--no-gaussian", args.no_gaussian),
            ("--no-saes", args.no_saes),
            ("--no-fsdr", args.no_fsdr),
            ("--baseline-only", args.baseline_only),
        )
        if enabled
    ]
    if strict_run and disabled:
        build_parser().error(
            f"{mode} forbids disabled stages: " + ", ".join(disabled)
        )
    # Diagnostic runs are still strict about executing every hardware stage,
    # but may exercise explicitly non-claiming mechanism variants.
    if claim_or_functional_run and args.saes_materialization != "representative":
        build_parser().error(
            f"{mode} requires --saes-materialization representative"
        )
    if claim_or_functional_run and args.saes_diagnostic_sweep:
        build_parser().error(f"{mode} forbids --saes-diagnostic-sweep")
    if claim_or_functional_run and args.saes_decision_semantics != "current":
        build_parser().error(
            f"{mode} forbids non-default --saes-decision-semantics"
        )
    if claim_or_functional_run and args.saes_feature_source != "pipeline":
        build_parser().error(f"{mode} forbids non-default --saes-feature-source")
    if (args.claim_run or args.functional_run) and args.fsdr_guidance_policy != (
        "paper-hamming-local-validity"
    ):
        build_parser().error(f"{mode} forbids non-default --fsdr-guidance-policy")
    if args.fsdr_only and not (args.claim_run or args.diagnostic_run):
        build_parser().error(
            "--fsdr-only requires --claim-run or --diagnostic-run"
        )
    if args.fsdr_only and args.diagnostic_run and args.image_output_policy != "none":
        build_parser().error(
            "diagnostic --fsdr-only requires --image-output-policy none"
        )
    if args.fsdr_only and any(
        (
            args.ablation,
            args.baseline_only,
            args.sensitivity_trace,
            args.calibration_trace,
            args.saes_diagnostic_sweep,
            args.tune_thresholds,
        )
    ):
        build_parser().error("--fsdr-only cannot be combined with other run modes")
    if args.saes_diagnostic_sweep and (
        args.no_saes
        or args.baseline_only
        or args.saes_materialization != "representative"
    ):
        build_parser().error(
            "--saes-diagnostic-sweep requires the sparse representative SAES path"
        )
    if args.saes_materialization_audit:
        if not args.diagnostic_run:
            build_parser().error(
                "--saes-materialization-audit requires --diagnostic-run"
            )
        if any(
            (
                args.ablation,
                args.baseline_only,
                args.fsdr_only,
                args.tune_thresholds,
                args.calibration_trace,
                args.sensitivity_trace,
                args.saes_diagnostic_sweep,
                args.saes_s3_raw_audit,
                args.saes_routing_audit,
                args.saes_hardware_audit,
            )
        ):
            build_parser().error(
                "--saes-materialization-audit cannot be combined with another run mode"
            )
    if args.saes_s3_raw_audit:
        if not args.diagnostic_run:
            build_parser().error("--saes-s3-raw-audit requires --diagnostic-run")
        if args.saes_materialization != "representative":
            build_parser().error(
                "--saes-s3-raw-audit requires representative SAES materialization"
            )
        if args.saes_feature_source != "pipeline":
            build_parser().error(
                "--saes-s3-raw-audit requires the pipeline S1 feature source"
            )
        if args.image_output_policy != "none":
            build_parser().error(
                "--saes-s3-raw-audit requires --image-output-policy none"
            )
        if any(
            (
                args.ablation,
                args.baseline_only,
                args.fsdr_only,
                args.tune_thresholds,
                args.calibration_trace,
                args.sensitivity_trace,
                args.saes_diagnostic_sweep,
                args.saes_routing_audit,
                args.saes_hardware_audit,
            )
        ):
            build_parser().error(
                "--saes-s3-raw-audit cannot be combined with another run mode"
            )
        overrides = [
            option
            for option, enabled in (
                ("--saes-fv", args.saes_fv is not None),
                ("--saes-ds", args.saes_ds is not None),
                ("--saes-cc", args.saes_cc is not None),
                ("--tile-size", args.tile_size is not None),
                ("--fsdr-cache-size", args.fsdr_cache_size is not None),
                ("--fsdr-hamming", args.fsdr_hamming is not None),
                (
                    "--fsdr-guidance-policy",
                    args.fsdr_guidance_policy != "paper-hamming-local-validity",
                ),
                ("--seed", args.seed != 0),
            )
            if enabled
        ]
        if overrides:
            build_parser().error(
                "--saes-s3-raw-audit forbids protocol overrides: "
                + ", ".join(overrides)
            )
    if args.saes_routing_audit:
        if not args.diagnostic_run:
            build_parser().error("--saes-routing-audit requires --diagnostic-run")
        if args.saes_materialization != "representative":
            build_parser().error(
                "--saes-routing-audit requires representative SAES materialization"
            )
        if args.saes_feature_source != "pipeline":
            build_parser().error(
                "--saes-routing-audit requires the pipeline S1 feature source"
            )
        if args.saes_decision_semantics != "probe-normalized-std-first-hit":
            build_parser().error(
                "--saes-routing-audit requires "
                "--saes-decision-semantics probe-normalized-std-first-hit"
            )
        if args.image_output_policy != "none":
            build_parser().error(
                "--saes-routing-audit requires --image-output-policy none"
            )
        if any(
            (
                args.ablation,
                args.baseline_only,
                args.fsdr_only,
                args.tune_thresholds,
                args.calibration_trace,
                args.sensitivity_trace,
                args.saes_diagnostic_sweep,
                args.saes_materialization_audit,
                args.saes_s3_raw_audit,
                args.saes_hardware_audit,
            )
        ):
            build_parser().error(
                "--saes-routing-audit cannot be combined with another run mode"
            )
        overrides = [
            option
            for option, enabled in (
                ("--saes-fv", args.saes_fv is not None),
                ("--saes-ds", args.saes_ds is not None),
                ("--saes-cc", args.saes_cc is not None),
                ("--tile-size", args.tile_size is not None),
                ("--fsdr-cache-size", args.fsdr_cache_size is not None),
                ("--fsdr-hamming", args.fsdr_hamming is not None),
                (
                    "--fsdr-guidance-policy",
                    args.fsdr_guidance_policy != "paper-hamming-local-validity",
                ),
                ("--seed", args.seed != 0),
            )
            if enabled
        ]
        if overrides:
            build_parser().error(
                "--saes-routing-audit forbids protocol overrides: "
                + ", ".join(overrides)
            )
    if args.saes_hardware_audit:
        if not args.diagnostic_run:
            build_parser().error("--saes-hardware-audit requires --diagnostic-run")
        if args.saes_materialization != "representative":
            build_parser().error(
                "--saes-hardware-audit requires representative SAES materialization"
            )
        if args.saes_feature_source != "pipeline":
            build_parser().error(
                "--saes-hardware-audit requires the pipeline S1 feature source"
            )
        if args.saes_decision_semantics != "current":
            build_parser().error(
                "--saes-hardware-audit requires --saes-decision-semantics current"
            )
        if args.image_output_policy != "none":
            build_parser().error(
                "--saes-hardware-audit requires --image-output-policy none"
            )
        if any(
            (
                args.ablation,
                args.baseline_only,
                args.fsdr_only,
                args.tune_thresholds,
                args.calibration_trace,
                args.sensitivity_trace,
                args.saes_diagnostic_sweep,
                args.saes_materialization_audit,
                args.saes_s3_raw_audit,
                args.saes_routing_audit,
            )
        ):
            build_parser().error(
                "--saes-hardware-audit cannot be combined with another run mode"
            )
        overrides = [
            option
            for option, enabled in (
                ("--saes-fv", args.saes_fv is not None),
                ("--saes-ds", args.saes_ds is not None),
                ("--saes-cc", args.saes_cc is not None),
                ("--tile-size", args.tile_size is not None),
                ("--fsdr-cache-size", args.fsdr_cache_size is not None),
                ("--fsdr-hamming", args.fsdr_hamming is not None),
                (
                    "--fsdr-guidance-policy",
                    args.fsdr_guidance_policy != "paper-hamming-local-validity",
                ),
                ("--seed", args.seed != 0),
            )
            if enabled
        ]
        if overrides:
            build_parser().error(
                "--saes-hardware-audit forbids protocol overrides: "
                + ", ".join(overrides)
            )
    if args.calibration_trace and not args.diagnostic_run:
        build_parser().error("--calibration-trace requires --diagnostic-run")
    if args.calibration_trace and args.sensitivity_trace:
        build_parser().error(
            "--calibration-trace cannot be combined with --sensitivity-trace"
        )
    if args.calibration_parameters is not None and not args.calibration_trace:
        build_parser().error(
            "--calibration-parameters requires --calibration-trace"
        )
    if args.calibration_trace:
        overrides = [
            option
            for option, enabled in (
                ("--ablation", args.ablation),
                ("--tune-thresholds", args.tune_thresholds),
                ("--saes-fv", args.saes_fv is not None),
                ("--saes-ds", args.saes_ds is not None),
                ("--saes-cc", args.saes_cc is not None),
                ("--tile-size", args.tile_size is not None),
                ("--fsdr-cache-size", args.fsdr_cache_size is not None),
                ("--fsdr-hamming", args.fsdr_hamming is not None),
                ("--fsdr-guidance-policy", args.fsdr_guidance_policy != "paper-hamming-local-validity"),
                ("--image-output-policy", args.image_output_policy != "none"),
                ("--seed", args.seed != 0),
            )
            if enabled
        ]
        if overrides:
            build_parser().error(
                "--calibration-trace forbids protocol overrides: " + ", ".join(overrides)
            )
    return args
