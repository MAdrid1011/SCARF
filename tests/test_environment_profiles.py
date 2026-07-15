import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_profile_contracts_are_describable_without_importing_torch():
    for profile in ("classic", "depthsplat", "orin"):
        result = subprocess.run(
            [sys.executable, "scripts/check_environment.py", "--profile", profile, "--describe"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)


def test_python_lockfiles_pin_rasterizer_commit():
    commit = "1250c420ebb945f0dce9945086e22faab9157c92"
    for profile in ("classic", "depthsplat"):
        lock = (ROOT / "environments" / profile / "requirements.lock").read_text()
        assert f"diff-gaussian-rasterization-modified@{commit}" in lock
        for line in lock.splitlines():
            if line and not line.startswith("git+"):
                assert "==" in line


def test_installer_exposes_separate_virtual_environment_interface():
    installer = (ROOT / "install.sh").read_text(encoding="utf-8")
    assert "--venv" in installer
    assert 'VENV}/bin/python' in installer
    assert 'CHECK_ONLY}' in installer
    assert "select_cuda_compiler.py" in installer
    assert "CUDAHOSTCXX" in installer


def test_profile_lock_version_comparison_uses_pep440_normalization():
    from scripts.check_environment import lock_versions_equal

    assert lock_versions_equal("1.1", "1.1.0")
    assert lock_versions_equal("2.4.0+cu121", "2.4.0+cu121")
    assert not lock_versions_equal("1.1", "1.2.0")
