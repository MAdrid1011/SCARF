#!/usr/bin/env python3
"""Validate a SCARF environment profile and emit its reproducibility digest."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
from packaging.version import InvalidVersion, Version


ROOT = Path(__file__).resolve().parents[1]
ORIN_CONTRACT_PATH = ROOT / "environments/orin/contract.json"
PROFILE_CONTRACTS = {
    "classic": {
        "python": "3.10", "torch": "2.1.2", "torchvision": "0.16.2", "cuda": "12.1"
    },
    "depthsplat": {
        "python": "3.10", "torch": "2.4.0", "torchvision": "0.19.0", "cuda": "12.1"
    },
    "orin": {"device": "Jetson Orin NX", "power_mode": "MAXN", "clocks": "recorded"},
}


def lock_versions_equal(actual: str, expected: str) -> bool:
    """Compare locked package versions using PEP 440 normalization."""
    try:
        return Version(actual) == Version(expected)
    except InvalidVersion:
        return actual == expected


def run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"command failed: {' '.join(command)}")
    return result.stdout.strip()


def file_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required environment record is missing: {path}")
    return path.read_text(encoding="utf-8", errors="replace").strip()


def normalized_freeze_lines(freeze: str) -> list[str]:
    lines = []
    for line in freeze.splitlines():
        if " @ file://" in line:
            name = line.split(" @ ", 1)[0]
            try:
                line = f"{name}=={version(name)}"
            except PackageNotFoundError as exc:
                raise RuntimeError(f"cannot normalize local package origin: {name}") from exc
        lines.append(line)
    return lines


def validate_python_profile(profile: str) -> dict:
    contract = PROFILE_CONTRACTS[profile]
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if python_version != contract["python"]:
        raise RuntimeError(f"{profile} requires Python {contract['python']}, got {python_version}")
    import torch
    import torchvision

    if not torch.__version__.startswith(contract["torch"]):
        raise RuntimeError(f"{profile} requires PyTorch {contract['torch']}, got {torch.__version__}")
    if not torchvision.__version__.startswith(contract["torchvision"]):
        raise RuntimeError(
            f"{profile} requires torchvision {contract['torchvision']}, got {torchvision.__version__}"
        )
    cuda_version = torch.version.cuda
    if cuda_version is not None and cuda_version != contract["cuda"]:
        raise RuntimeError(f"{profile} requires CUDA {contract['cuda']}, got {cuda_version}")
    freeze = run([sys.executable, "-m", "pip", "freeze", "--all"])
    nvcc_path = shutil.which("nvcc")
    nvcc_version = run([nvcc_path, "--version"]) if nvcc_path else None
    if cuda_version is not None and (
        nvcc_version is None or f"release {contract['cuda']}" not in nvcc_version
    ):
        raise RuntimeError(f"{profile} requires nvcc release {contract['cuda']}")
    compiler = run(["cc", "--version"])
    lock = ROOT / "environments" / profile / "requirements.lock"
    mismatches = []
    for line in lock.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or line.startswith("git+"):
            continue
        package, expected = line.split("==", 1)
        try:
            actual = version(package)
        except PackageNotFoundError:
            mismatches.append(f"{package}: missing (expected {expected})")
            continue
        if not lock_versions_equal(actual, expected):
            mismatches.append(f"{package}: {actual} (expected {expected})")
    rasterizer_commit = "1250c420ebb945f0dce9945086e22faab9157c92"
    if rasterizer_commit not in freeze:
        mismatches.append("diff-gaussian-rasterization: pinned commit is not installed")
    if mismatches:
        raise RuntimeError("profile lock mismatch:\n" + "\n".join(mismatches))
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda": cuda_version,
        "nvcc": nvcc_version,
        "c_compiler": compiler,
        "cuda_available": torch.cuda.is_available(),
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "pip_freeze_sha256": hashlib.sha256((freeze + "\n").encode()).hexdigest(),
        "pip_freeze": normalized_freeze_lines(freeze),
    }


def load_orin_contract(path: Path = ORIN_CONTRACT_PATH) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("status") != "runtime_autodetect":
        raise RuntimeError("Orin environment contract must use runtime autodetection")
    required = (
        "device_model_contains",
        "power_mode_contains",
        "require_jetson_clocks",
        "maximum_allowed_temperature_c",
        "timing_repetitions",
    )
    missing = [key for key in required if contract.get(key) in (None, "")]
    if missing:
        raise RuntimeError("Orin environment contract is incomplete: " + ", ".join(missing))
    return contract


def validate_orin(contract_path: Path = ORIN_CONTRACT_PATH) -> dict:
    contract = load_orin_contract(contract_path)
    model = file_text(Path("/proc/device-tree/model")).rstrip("\x00")
    if contract["device_model_contains"] not in model:
        raise RuntimeError(f"expected Jetson Orin NX, got {model}")
    l4t = file_text(Path("/etc/nv_tegra_release"))
    power = run(["nvpmodel", "-q"])
    if contract["power_mode_contains"].upper() not in power.upper():
        raise RuntimeError("Orin must be in MAXN power mode")
    clocks = run(["jetson_clocks", "--show"])
    if contract["require_jetson_clocks"] and not clocks:
        raise RuntimeError("jetson_clocks did not report the locked clock state")
    nvcc = run(["nvcc", "--version"])
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Orin PyTorch CUDA is unavailable")
    try:
        jetpack = run(
            ["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"]
        )
    except RuntimeError:
        jetpack = "not-installed-as-meta-package"
    return {
        "contract": contract,
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "device_model": model,
        "jetpack": jetpack,
        "l4t_release": l4t,
        "nvpmodel": power,
        "jetson_clocks": clocks,
        "nvcc": nvcc,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILE_CONTRACTS), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(PROFILE_CONTRACTS[args.profile], indent=2, sort_keys=True))
        return 0
    try:
        details = (
            validate_orin()
            if args.profile == "orin"
            else validate_python_profile(args.profile)
        )
    except (FileNotFoundError, RuntimeError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    record = {
        "schema_version": "1.0",
        "profile": args.profile,
        "contract": PROFILE_CONTRACTS[args.profile],
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "details": details,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
