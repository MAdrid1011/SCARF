#!/usr/bin/env python3
"""Create a deterministic file manifest for one prepared dataset tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(root: Path, name: str, source: str, revision: str) -> dict:
    root = root.resolve()
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name != ".scarf-manifest.json"
    )
    if not files:
        raise ValueError(f"dataset tree is empty: {root}")
    records = {
        str(path.relative_to(root)): {"size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
    }
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": "1.0",
        "dataset": name,
        "source": source,
        "revision": revision,
        "file_count": len(records),
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    record = build(args.root, args.name, args.source, args.revision)
    output = args.root / ".scarf-manifest.json"
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
