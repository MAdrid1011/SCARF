import hashlib
import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_seed42_projection_export_is_idempotent_and_hash_bound(tmp_path):
    pytest.importorskip("torch")
    from hardware.rtl.export_lsh_projection import export

    scala = tmp_path / "LSHProjectionROM.scala"
    manifest = tmp_path / "lsh_projection.json"
    first = export(scala, manifest)
    first_source = scala.read_bytes()
    second = export(scala, manifest)

    assert first == second
    assert scala.read_bytes() == first_source
    assert first["sha256"] == "a9d3431fca57f8408237281c60f3bf3fbe572289abd0e6be3b05824afc428397"
    values = [
        int(value, 16)
        for value in re.findall(r"0x([0-9a-f]{4})", scala.read_text())
    ]
    payload = b"".join(value.to_bytes(2, "little") for value in values)
    assert len(values) == 16 * 128
    assert hashlib.sha256(payload).hexdigest() == first["sha256"]
    assert json.loads(manifest.read_text()) == first


def test_checked_in_projection_matches_the_manifest():
    manifest = json.loads((ROOT / "artifact/lsh_projection.json").read_text())
    scala = ROOT / manifest["scala_source"]
    values = [
        int(value, 16)
        for value in re.findall(r"0x([0-9a-f]{4})", scala.read_text())
    ]
    payload = b"".join(value.to_bytes(2, "little") for value in values)

    assert len(values) == manifest["rows"] * manifest["columns"]
    assert hashlib.sha256(payload).hexdigest() == manifest["sha256"]
