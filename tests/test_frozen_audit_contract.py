from pathlib import Path

import pytest


def test_require_frozen_identity_accepts_nested_exact_fields_and_rejects_drift():
    from data.frozen_audit_contract import require_frozen_identity

    expected = {
        "tree_sha256": "a" * 64,
        "source_binding": {"index_sha256": "b" * 64},
    }
    actual = {**expected, "unrelated_schema_field": False}

    assert require_frozen_identity(actual, expected, label="fixture") == actual

    changed = {
        **actual,
        "source_binding": {"index_sha256": "c" * 64},
    }
    with pytest.raises(RuntimeError, match="source_binding.index_sha256"):
        require_frozen_identity(changed, expected, label="fixture")


def test_require_fixed_file_sha256_rejects_a_changed_file(tmp_path: Path):
    from data.frozen_audit_contract import require_fixed_file_sha256, sha256_file

    path = tmp_path / "checkpoint.bin"
    path.write_bytes(b"fixed")
    expected = sha256_file(path)

    assert (
        require_fixed_file_sha256(path, expected, label="fixture checkpoint")
        == expected
    )

    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="frozen contract"):
        require_fixed_file_sha256(path, expected, label="fixture checkpoint")
