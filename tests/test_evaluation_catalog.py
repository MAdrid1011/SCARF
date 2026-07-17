import ast
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "artifact/evaluation_catalog.json"
RUNNER = ROOT / "scripts/run_ae.py"


def load_catalog() -> dict:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def test_catalog_covers_every_evaluation_figure_and_table_once():
    catalog = load_catalog()
    records = catalog["results"]
    ids = [record["id"] for record in records]

    assert catalog["schema_version"] == "1.0"
    assert len(records) == 13
    assert len(ids) == len(set(ids))
    assert set(ids) == {
        *(f"figure{number}" for number in range(8, 17)),
        *(f"table{number}" for number in range(1, 5)),
    }


def test_catalog_separates_claimable_results_from_public_proxies():
    catalog = load_catalog()
    records = {record["id"]: record for record in catalog["results"]}
    classes = set(catalog["evidence_classes"])

    assert set(catalog["badge_request"]) == {
        "Artifact Available",
        "Artifacts Evaluated - Functional",
        "Results Reproduced",
    }
    for record in records.values():
        assert record["required_evidence_class"] in classes
        assert record["raw_inputs"]
        assert record["command"]
        assert record["acceptance"]

    assert records["figure8"]["claim_role"] == "mandatory_key_result"
    assert records["figure8"]["required_evidence_class"] == (
        "independent_measurement"
    )
    for result_id in ("figure9", "table4"):
        assert records[result_id]["claim_role"] == "public_proxy"
        assert records[result_id]["required_evidence_class"] == (
            "public_physical_proxy"
        )
    assert all(
        record["required_evidence_class"]
        in {"independent_measurement", "deterministic_execution"}
        for record in records.values()
        if record["claim_role"] == "mandatory_key_result"
    )


def test_runtime_modules_do_not_read_paper_expected_results():
    allowed = {
        ROOT / "scripts/build_archive.py",
        ROOT / "scripts/check_release.py",
        ROOT / "scripts/generate_report.py",
        ROOT / "scripts/validate_ae.py",
    }
    offenders = []
    roots = (
        ROOT / "scripts",
        ROOT / "fsdr",
        ROOT / "saes",
        ROOT / "encoder",
        ROOT / "depth_predictor",
        ROOT / "ggu",
        ROOT / "hardware",
        ROOT / "integration",
    )
    for directory in roots:
        for path in directory.rglob("*.py"):
            if path in allowed:
                continue
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            constants = {
                node.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
            if any("expected_results.json" in value for value in constants):
                offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []


def run_dry(tmp_path: Path, mode: str, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            mode,
            "--dry-run",
            "--output-root",
            str(tmp_path),
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def test_public_runner_exposes_complete_evaluation_interface(tmp_path):
    modes = (
        "quick",
        "quality",
        "performance",
        "worstcase",
        "mechanisms",
        "utilization",
        "sensitivity",
        "rtl",
        "dram",
        "physical",
        "scale",
        "figures",
        "validate",
        "all-eval",
    )
    for mode in modes:
        extra = ("--device", "orin") if mode == "performance" else ()
        result = run_dry(tmp_path / mode, mode, *extra)
        assert result.returncode == 0, f"{mode}: {result.stderr}"
        plan = json.loads(result.stdout)
        assert plan["mode"] == mode
        assert plan["output_root"] == str(tmp_path / mode)


def test_report_catalog_must_bind_all_results_to_evidence_classes(tmp_path):
    from scripts.generate_report import validate_figure_catalog

    catalog = load_catalog()
    generated = {
        "schema_version": "2.0",
        "results": [
            {
                "id": record["id"],
                "evidence_class": record["required_evidence_class"],
                "status": "PASS",
                "source_data": [{"path": "raw.json", "sha256": "a" * 64}],
                "exports": ["result.pdf", "result.png"],
            }
            for record in catalog["results"]
        ],
    }
    validate_figure_catalog(generated, catalog, require_key_results=True)

    generated["results"][0]["evidence_class"] = "paper_comparison_target"
    try:
        validate_figure_catalog(generated, catalog, require_key_results=True)
    except ValueError as exc:
        assert "evidence class" in str(exc)
    else:
        raise AssertionError("weaker evidence class was accepted")


def test_partial_report_catalog_allows_honest_not_run_but_strict_rejects_it():
    from scripts.generate_report import validate_figure_catalog

    catalog = load_catalog()
    generated = {
        "schema_version": "2.0",
        "results": [
            {
                "id": record["id"],
                "evidence_class": record["required_evidence_class"],
                "status": "NOT_RUN",
                "reason": "required evidence is not present",
                "source_data": [],
                "exports": [f"status/{record['id']}.json"],
            }
            for record in catalog["results"]
        ],
    }

    validate_figure_catalog(generated, catalog, require_key_results=False)
    try:
        validate_figure_catalog(generated, catalog, require_key_results=True)
    except ValueError as exc:
        assert "mandatory key result" in str(exc)
    else:
        raise AssertionError("strict validation accepted a NOT_RUN key result")
