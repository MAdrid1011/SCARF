#!/usr/bin/env python3
"""Select a CUDA-compatible host C/C++ compiler without unsafe nvcc flags."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


CUDA_GCC_MAX_MAJOR = {
    "12.1": 12,
}


def choose_cuda_host_compiler(
    *,
    cuda_release: str,
    default_cc: Path,
    default_cxx: Path,
    default_major: int,
    versioned_compilers: dict[int, tuple[Path, Path]],
) -> tuple[Path, Path]:
    maximum = CUDA_GCC_MAX_MAJOR.get(cuda_release)
    if maximum is None:
        raise RuntimeError(f"unsupported CUDA release for compiler selection: {cuda_release}")
    if default_major <= maximum:
        return default_cc, default_cxx
    candidates = [major for major in versioned_compilers if major <= maximum]
    if not candidates:
        raise RuntimeError(
            f"CUDA {cuda_release} requires a compatible GCC/G++ (major <= {maximum})"
        )
    return versioned_compilers[max(candidates)]


def _output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"command failed: {' '.join(command)}")
    return result.stdout.strip()


def _cuda_release(nvcc: Path) -> str:
    match = re.search(r"release\s+(\d+\.\d+)", _output([str(nvcc), "--version"]))
    if match is None:
        raise RuntimeError("cannot determine the CUDA release from nvcc")
    return match.group(1)


def _compiler_major(cc: Path) -> int:
    version = _output([str(cc), "-dumpfullversion"])
    try:
        return int(version.split(".", 1)[0])
    except ValueError as exc:
        raise RuntimeError(f"cannot determine GCC major version: {version}") from exc


def discover(nvcc: Path, cc: Path, cxx: Path) -> tuple[Path, Path]:
    release = _cuda_release(nvcc)
    maximum = CUDA_GCC_MAX_MAJOR.get(release)
    if maximum is None:
        raise RuntimeError(f"unsupported CUDA release for compiler selection: {release}")
    versioned = {}
    for major in range(maximum, 4, -1):
        versioned_cc = shutil.which(f"gcc-{major}")
        versioned_cxx = shutil.which(f"g++-{major}")
        if versioned_cc and versioned_cxx:
            versioned[major] = (Path(versioned_cc), Path(versioned_cxx))
    return choose_cuda_host_compiler(
        cuda_release=release,
        default_cc=cc,
        default_cxx=cxx,
        default_major=_compiler_major(cc),
        versioned_compilers=versioned,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--cc", type=Path, required=True)
    parser.add_argument("--cxx", type=Path, required=True)
    args = parser.parse_args()
    try:
        cc, cxx = discover(args.nvcc.resolve(), args.cc.resolve(), args.cxx.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(cc)
    print(cxx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
