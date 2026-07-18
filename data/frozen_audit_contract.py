"""Reusable hash and identity checks for fixed diagnostic inputs."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    """Return the SHA256 of one regular file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"frozen audit file is unavailable: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_fixed_file_sha256(path: Path, expected: str, *, label: str) -> str:
    """Fail closed unless a fixed input file matches its declared SHA256."""
    if not isinstance(expected, str) or _SHA256_PATTERN.fullmatch(expected) is None:
        raise ValueError(f"{label} has an invalid expected SHA256")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 does not match the frozen contract")
    return actual


def require_frozen_identity(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    """Require every declared frozen identity field to match recursively.

    Callers own their input schema. This helper deliberately validates only the
    declared identity fields, so a separate audit schema can bind an explicit
    frozen target-camera subset without inheriting a context-only payload rule.
    """
    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        raise ValueError(f"{label} identity must be a mapping")

    def compare(observed: Any, required: Any, path: str) -> None:
        if isinstance(required, Mapping):
            if not isinstance(observed, Mapping):
                raise RuntimeError(f"{label} identity differs at {path}")
            for key, value in required.items():
                if key not in observed:
                    raise RuntimeError(f"{label} identity is missing {path}.{key}")
                compare(observed[key], value, f"{path}.{key}")
            return
        if observed != required:
            raise RuntimeError(f"{label} identity differs at {path}")

    for key, value in expected.items():
        if key not in actual:
            raise RuntimeError(f"{label} identity is missing {key}")
        compare(actual[key], value, key)
    return dict(actual)
