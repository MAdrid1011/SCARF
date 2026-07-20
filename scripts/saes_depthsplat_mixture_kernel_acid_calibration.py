#!/usr/bin/env python3
"""Freeze a source-only ACID 24/8 kernel-risk threshold for DepthSplat v3."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saes.depthsplat_acid_disjoint_calibration import (  # noqa: E402
    DEFAULT_MATERIALIZATION_ROOT,
    DEFAULT_PLAN_PATH,
)
from saes.depthsplat_mixture_kernel_acid_calibration import (  # noqa: E402
    KERNEL_RISK_RECORD_NAME,
)
from saes.depthsplat_mixture_kernel_acid_collector import (  # noqa: E402
    collect_mixture_kernel_risk_calibration,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan-path", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument(
        "--materialization-root", type=Path, default=DEFAULT_MATERIALIZATION_ROOT
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists():
        raise SystemExit("--output-dir must be a new directory")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("DepthSplat kernel-risk ACID calibration requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    try:
        result = collect_mixture_kernel_risk_calibration(
            device=device,
            output_directory=args.output_dir,
            plan_path=args.plan_path,
            materialization_root=args.materialization_root,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    record_path = Path(result["record_path"])
    if record_path.name != KERNEL_RISK_RECORD_NAME:
        raise RuntimeError("kernel-risk collector wrote an unexpected frozen record name")
    print(record_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
