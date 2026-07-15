"""Command-line contract for the SCARF model experiment."""

from __future__ import annotations

import argparse
from pathlib import Path


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
    parser.add_argument(
        "--fsdr-only",
        action="store_true",
        help="Emit target-free FSDR Table 2 evidence without SAES or rendering",
    )
    parser.add_argument(
        "--fsdr-feature-source",
        choices=("pipeline", "depthsplat-mono"),
        default="pipeline",
        help="Select the executed feature tensor hashed by an FSDR-only diagnostic",
    )
    parser.add_argument(
        "--saes-diagnostic-sweep",
        action="store_true",
        help="Run a non-claim decision-statistic and sparse-coverage sweep",
    )

    parser.add_argument("--saes-fv", type=float)
    parser.add_argument("--saes-ds", type=float)
    parser.add_argument("--saes-cc", type=float)
    parser.add_argument(
        "--saes-materialization",
        choices=("representative", "dense-diagnostic"),
        default="representative",
        help="Use the paper-faithful sparse path or a non-claim dense diagnostic",
    )
    parser.add_argument("--tile-size", type=positive_int)
    parser.add_argument("--fsdr-cache-size", type=positive_int)
    parser.add_argument("--fsdr-hamming", type=nonnegative_int)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.sample_index >= args.num_samples:
        raise SystemExit("--sample-index must be smaller than --num-samples")
    strict_run = args.claim_run or args.functional_run
    mode = "--claim-run" if args.claim_run else "--functional-run"
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
    if strict_run and args.saes_materialization != "representative":
        build_parser().error(
            f"{mode} requires --saes-materialization representative"
        )
    if strict_run and args.saes_diagnostic_sweep:
        build_parser().error(f"{mode} forbids --saes-diagnostic-sweep")
    if args.fsdr_only and not args.claim_run:
        build_parser().error("--fsdr-only requires --claim-run")
    if args.fsdr_only and any(
        (
            args.ablation,
            args.baseline_only,
            args.sensitivity_trace,
            args.saes_diagnostic_sweep,
            args.tune_thresholds,
        )
    ):
        build_parser().error("--fsdr-only cannot be combined with other run modes")
    if args.fsdr_feature_source != "pipeline" and (
        not args.fsdr_only or args.model != "depthsplat"
    ):
        build_parser().error(
            "--fsdr-feature-source depthsplat-mono requires "
            "--model depthsplat --fsdr-only"
        )
    if args.saes_diagnostic_sweep and (
        args.no_saes
        or args.baseline_only
        or args.saes_materialization != "representative"
    ):
        build_parser().error(
            "--saes-diagnostic-sweep requires the sparse representative SAES path"
        )
    return args
