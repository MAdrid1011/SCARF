#!/usr/bin/env python3
"""Bind passing prepared-tree validations into the AE dataset contract."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEX64 = re.compile(r"[0-9a-f]{64}")
REPRESENTATIONS = {
    "re10k-native": "re10k",
    "acid-native": "acid",
    "depthsplat-native-270x480-v1": "dl3dv-native",
    "re10k-compatible-360x640-v1": "dl3dv-re10k",
}


def load_validation(path: Path, representation: str) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    digest = record.get("tree", {}).get("tree_sha256")
    if (
        record.get("kind") != "prepared_dataset_validation"
        or record.get("status") != "PASS"
        or record.get("representation") != representation
        or not isinstance(digest, str)
        or not HEX64.fullmatch(digest)
    ):
        raise ValueError(f"invalid prepared dataset validation: {path}")
    selection = record.get("selection")
    if not isinstance(selection, dict):
        raise ValueError(f"prepared dataset validation has no selection: {path}")
    source = record["tree"].get("source")
    revision = record["tree"].get("revision")
    if not isinstance(source, str) or not source or not isinstance(revision, str) or not revision:
        raise ValueError(f"prepared dataset validation has no source revision: {path}")
    return {
        "tree_sha256": digest,
        "source": source,
        "revision": revision,
        "source_index_sha256": selection.get("source_index_sha256"),
        "sample_selection_sha256": selection.get("sample_selection_sha256"),
    }


def finalize(
    protocol_path: Path,
    manifest_path: Path,
    validations: dict[str, Path],
) -> tuple[dict, dict]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validated = {
        representation: load_validation(path, representation)
        for representation, path in validations.items()
    }
    for pair, record in protocol["pairs"].items():
        representation = record["dataset_representation"]
        if representation in validated:
            validation = validated[representation]
            if validation["source_index_sha256"] != record["source_index_sha256"]:
                raise ValueError(f"{pair} validation source index SHA256 mismatch")
            if validation["sample_selection_sha256"] != record["sample_selection_sha256"]:
                raise ValueError(f"{pair} validation sample selection SHA256 mismatch")
            record["dataset_tree_sha256"] = validation["tree_sha256"]
    datasets = manifest["datasets"]
    if "re10k-native" in validated:
        if validated["re10k-native"]["source"] != datasets["re10k"]["prepared_source"]:
            raise ValueError("Re10K prepared source does not match the dataset contract")
        datasets["re10k"]["expected_tree_sha256"] = validated["re10k-native"]["tree_sha256"]
        datasets["re10k"]["prepared_source_revision"] = validated["re10k-native"]["revision"]
    if "acid-native" in validated:
        if validated["acid-native"]["source"] != datasets["acid"]["prepared_source"]:
            raise ValueError("ACID prepared source does not match the dataset contract")
        datasets["acid"]["expected_tree_sha256"] = validated["acid-native"]["tree_sha256"]
        datasets["acid"]["prepared_source_revision"] = validated["acid-native"]["revision"]
    if "depthsplat-native-270x480-v1" in validated:
        if validated["depthsplat-native-270x480-v1"]["source"] != datasets["dl3dv"]["prepared_source"]:
            raise ValueError("DL3DV native prepared source does not match the dataset contract")
        if validated["depthsplat-native-270x480-v1"]["revision"] != datasets["dl3dv"]["prepared_source_revision"]:
            raise ValueError("DL3DV native prepared revision does not match the dataset contract")
        datasets["dl3dv"]["representations"]["native"]["expected_tree_sha256"] = validated[
            "depthsplat-native-270x480-v1"
        ]["tree_sha256"]
    if "re10k-compatible-360x640-v1" in validated:
        if validated["re10k-compatible-360x640-v1"]["source"] != datasets["dl3dv"]["prepared_source"]:
            raise ValueError("DL3DV Re10K prepared source does not match the dataset contract")
        if validated["re10k-compatible-360x640-v1"]["revision"] != datasets["dl3dv"]["prepared_source_revision"]:
            raise ValueError("DL3DV Re10K prepared revision does not match the dataset contract")
        datasets["dl3dv"]["representations"]["re10k"]["expected_tree_sha256"] = validated[
            "re10k-compatible-360x640-v1"
        ]["tree_sha256"]
    return protocol, manifest


def write_json_atomic(path: Path, record: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", action="append", default=[], metavar="REPRESENTATION=PATH")
    parser.add_argument("--protocol", type=Path, default=ROOT / "artifact/evaluation_protocol.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifact/manifests/datasets.json")
    args = parser.parse_args()
    validations = {}
    for item in args.validation:
        if "=" not in item:
            parser.error("--validation requires REPRESENTATION=PATH")
        representation, path_text = item.split("=", 1)
        if representation not in REPRESENTATIONS or representation in validations:
            parser.error(f"invalid or duplicate representation: {representation}")
        validations[representation] = Path(path_text)
    if not validations:
        parser.error("at least one --validation is required")
    try:
        protocol, manifest = finalize(args.protocol, args.manifest, validations)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_json_atomic(args.protocol, protocol)
    write_json_atomic(args.manifest, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
