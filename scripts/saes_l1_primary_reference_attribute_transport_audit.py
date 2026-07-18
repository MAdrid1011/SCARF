#!/usr/bin/env python3
"""Audit fixed adapter-offset SH/opacity transport without target RGB or rendering."""

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

from scripts.saes_l1_primary_reference_attribute_audit import (
    ATTRIBUTE_TRANSPORT_AUDIT_KIND,
    ATTRIBUTE_TRANSPORT_MATERIALIZATION,
    SEED,
    collect_attribute_audit,
)


MATERIALIZATION = ATTRIBUTE_TRANSPORT_MATERIALIZATION
AUDIT_KIND = ATTRIBUTE_TRANSPORT_AUDIT_KIND


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir must be a new directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    record = collect_attribute_audit(
        input_root=args.input_root,
        device=device,
        materialization=MATERIALIZATION,
        audit_kind=AUDIT_KIND,
    )
    from scripts.result_record import portable_command, write_result

    record["command"] = portable_command(
        [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_result(record, args.output_dir / "results.json")
    print(args.output_dir / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
