"""Release-container and DOI metadata contracts."""

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_container_defaults_to_the_cuda_functional_quick_workflow() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    script = ROOT / "docker/run-functional.sh"
    contents = script.read_text(encoding="utf-8")

    assert "FROM nvidia/cuda:12.1.1-devel-ubuntu22.04" in dockerfile
    assert "FROM base AS source-manifest" in dockerfile
    assert "scripts/build_archive.py --source-only --require-doi" in dockerfile
    assert "release-manifest.json" in dockerfile
    assert "if [ -f release-manifest.json ]" in dockerfile
    assert "COPY --from=source-manifest /opt/release/SCARF-AE/ /opt/scarf/" in dockerfile
    assert "MPLCONFIGDIR=/tmp/matplotlib" in dockerfile
    assert "ENTRYPOINT [\"/opt/scarf/docker/run-functional.sh\"]" in dockerfile
    assert "data/download_checkpoints.sh --profile quick" in contents
    assert "scripts/run_ae.sh quick --output-root" in contents
    assert "SCARF_OUTPUT_ROOT" in contents

    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_release_metadata_and_orin_reference_table_are_available() -> None:
    from scripts.check_release import check_doi

    doi, failures = check_doi()

    assert doi == "10.5281/zenodo.22434931"
    assert failures == []
    reference = ROOT / "artifact/reference_results/orin_nx_reference.csv"
    rows = reference.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 10
    assert all("paper_reference" in row for row in rows[1:])
