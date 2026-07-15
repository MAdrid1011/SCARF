#!/usr/bin/env python3
"""Generate the chunk index required by the Re10K-compatible loader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def generate(stage: Path) -> dict[str, str]:
    index = {}
    chunks = sorted(stage.glob("*.torch"))
    if not chunks:
        raise FileNotFoundError(f"no .torch chunks found in {stage}")
    for chunk_path in chunks:
        chunk = torch.load(chunk_path, map_location="cpu")
        for example in chunk:
            key = example.get("key")
            if not isinstance(key, str) or not key:
                raise ValueError(f"invalid example key in {chunk_path}")
            if key in index:
                raise ValueError(f"duplicate scene key: {key}")
            index[key] = chunk_path.name
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    index = generate(args.stage)
    (args.stage / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
