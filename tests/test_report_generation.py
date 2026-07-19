import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_report_generation_uses_only_result_evidence_not_paper_reference(
    tmp_path, monkeypatch
):
    from scripts import generate_report

    output = tmp_path / "evidence"
    report = tmp_path / "report"
    result_path = output / "quality" / "transplat_re10k" / "results.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "evidence_class": "deterministic_execution",
                "provenance": {
                    "model": "transplat",
                    "dataset": {"name": "re10k", "paper_result_eligible": False},
                    "evaluation": {"kind": "dataset_aggregate", "sample_count": 1},
                },
                "quality": {
                    "baseline": {"psnr_db": 11.0, "ssim": 0.1, "lpips": 0.9},
                    "scarf": {"psnr_db": 10.0, "ssim": 0.2, "lpips": 0.8},
                },
            }
        ),
        encoding="utf-8",
    )

    reference_path = ROOT / "artifact" / "expected_results.json"
    real_load = generate_report.load

    def load_without_paper_reference(path):
        assert Path(path).resolve() != reference_path.resolve()
        return real_load(path)

    monkeypatch.setattr(generate_report, "load", load_without_paper_reference)

    catalog = generate_report.generate(output, report, figures="table1")

    quality_rows = (report / "table1_quality.csv").read_text(encoding="utf-8")
    assert "11.0" not in quality_rows
    table1 = next(item for item in catalog["results"] if item["id"] == "table1")
    assert table1["status"] == "FAIL"
    assert "non-claimable" in table1["reason"]
