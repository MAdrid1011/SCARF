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

    parser.add_argument("--saes-fv", type=float)
    parser.add_argument("--saes-ds", type=float)
    parser.add_argument("--saes-cc", type=float)
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
    return args
