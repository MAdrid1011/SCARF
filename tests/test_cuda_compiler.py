from pathlib import Path

import pytest


def test_cuda_12_1_keeps_a_supported_default_gcc():
    from scripts.select_cuda_compiler import choose_cuda_host_compiler

    selected = choose_cuda_host_compiler(
        cuda_release="12.1",
        default_cc=Path("/toolchain/gcc"),
        default_cxx=Path("/toolchain/g++"),
        default_major=12,
        versioned_compilers={},
    )

    assert selected == (Path("/toolchain/gcc"), Path("/toolchain/g++"))


def test_cuda_12_1_selects_installed_gcc_12_when_default_is_too_new():
    from scripts.select_cuda_compiler import choose_cuda_host_compiler

    selected = choose_cuda_host_compiler(
        cuda_release="12.1",
        default_cc=Path("/usr/bin/gcc"),
        default_cxx=Path("/usr/bin/g++"),
        default_major=13,
        versioned_compilers={
            12: (Path("/usr/bin/gcc-12"), Path("/usr/bin/g++-12")),
            11: (Path("/usr/bin/gcc-11"), Path("/usr/bin/g++-11")),
        },
    )

    assert selected == (Path("/usr/bin/gcc-12"), Path("/usr/bin/g++-12"))


def test_cuda_compiler_selection_rejects_an_unsupported_only_toolchain():
    from scripts.select_cuda_compiler import choose_cuda_host_compiler

    with pytest.raises(RuntimeError, match="compatible GCC/G\\+\\+"):
        choose_cuda_host_compiler(
            cuda_release="12.1",
            default_cc=Path("/usr/bin/gcc"),
            default_cxx=Path("/usr/bin/g++"),
            default_major=13,
            versioned_compilers={},
        )
