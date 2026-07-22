"""Compile and resolve deterministic full and reviewer evidence profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REVIEWER_DOMAIN = "SCARF-AE-reviewer-v1"
REVIEWER_COUNTS = {"re10k": 512, "acid": 512, "dl3dv": 140}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ae_config import ClaimSelection, resolve_claim_selection
from scripts.compile_protocol import canonicalize_index, sha256_file


def _rank(dataset: str, scene: str) -> tuple[str, str]:
    material = f"{REVIEWER_DOMAIN}\0{dataset}\0{scene}".encode("utf-8")
    return hashlib.sha256(material).hexdigest(), scene


def _reviewer_index(source: dict[str, Any], dataset: str) -> dict[str, Any]:
    available = [
        (scene, entry)
        for scene, entry in source.items()
        if isinstance(scene, str) and scene and entry is not None
    ]
    count = REVIEWER_COUNTS[dataset]
    if len(available) < count:
        raise ValueError(f"{dataset} has fewer than {count} executable entries")
    selected = sorted(available, key=lambda item: _rank(dataset, item[0]))[:count]
    return {scene: entry for scene, entry in selected}


def compile_reviewer_profiles(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    protocol = json.loads(
        (root / "artifact/evaluation_protocol.json").read_text(encoding="utf-8")
    )
    output_root = root / "artifact/protocol/reviewer"
    output_root.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, Any] = {}
    for dataset in REVIEWER_COUNTS:
        pair = f"transplat/{dataset}"
        source_path = (root / protocol["pairs"][pair]["index_path"]).resolve()
        source = json.loads(source_path.read_text(encoding="utf-8"))
        selected = _reviewer_index(source, dataset)
        path = output_root / f"{dataset}.json"
        path.write_text(
            json.dumps(selected, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        digest = sha256_file(path)
        _, summary = canonicalize_index(path, digest)
        datasets[dataset] = {
            "index_path": str(path.relative_to(root)),
            **summary,
        }
    manifest = {
        "schema_version": "1.0",
        "kind": "reviewer_evidence_profile",
        "domain": REVIEWER_DOMAIN,
        "datasets": datasets,
    }
    manifest["selection_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return manifest


def resolve_evidence_selection(
    model: str, dataset: str, root: Path, profile: str
) -> ClaimSelection:
    if profile == "full":
        return resolve_claim_selection(model, dataset, root)
    if profile != "reviewer":
        raise ValueError(f"unsupported evidence profile: {profile}")
    root = Path(root).resolve()
    manifest_path = root / "artifact/protocol/reviewer/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "reviewer_evidence_profile":
        raise ValueError("reviewer evidence profile manifest is invalid")
    record = manifest.get("datasets", {}).get(dataset)
    if not isinstance(record, dict):
        raise ValueError(f"reviewer evidence profile has no {dataset} selection")
    path = (root / record["index_path"]).resolve()
    digest = sha256_file(path)
    _, summary = canonicalize_index(path, digest)
    for field in ("source_index_sha256", "sample_count", "sample_selection_sha256"):
        if summary[field] != record.get(field):
            raise ValueError(f"reviewer evidence profile {dataset} {field} mismatch")
    return ClaimSelection(
        model=model,
        dataset=dataset,
        index_path=path,
        source_index_sha256=digest,
        sample_count=summary["sample_count"],
        sample_selection_sha256=summary["sample_selection_sha256"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        record = compile_reviewer_profiles(args.root)
        output = args.output or args.root / "artifact/protocol/reviewer/manifest.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
