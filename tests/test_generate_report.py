import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def result(pair: str, target: dict, mechanisms: dict, speedup: float = 2.94) -> dict:
    model, dataset = pair.split("/")
    return {
        "schema_version": "1.0",
        "provenance": {
            "model": model,
            "dataset": {"name": dataset},
            "evaluation": {"kind": "dataset_aggregate", "sample_count": 2},
        },
        "quality": {
            "baseline": dict(zip(("psnr_db", "ssim", "lpips"), target["baseline"])),
            "scarf": dict(zip(("psnr_db", "ssim", "lpips"), target["scarf"])),
            "change": {"psnr_degradation_pct": 0.05},
        },
        "performance": {"speedup": speedup},
        "ablation": {
            "asic": {"eff_total": 1590},
            "asic_fsdr": {"eff_total": 1590 / 1.35},
            "asic_saes": {"eff_total": 1590 / 1.26},
            "asic_fsdr_saes": {"eff_total": 1000},
        },
        "fsdr_saes": {
            "fsdr": {
                "guided_rate": mechanisms["guided_rate"],
                "in_window_rate": mechanisms["top1_coverage"],
            },
            "saes": {
                "level0_ratio": mechanisms["level0_rate"],
                "level1_ratio": mechanisms["level1_rate"],
                "modification_ratio": mechanisms["gaussians_saved"],
            },
            "preservation": {
                "saes_low_var_agree": mechanisms["low_variance_agreement"]
            },
        },
    }


def write_fixture(output: Path) -> None:
    expected = json.loads((ROOT / "artifact/expected_results.json").read_text())
    for pair, target in expected["table1"].items():
        record = result(pair, target, expected["mechanisms"][pair])
        directory = pair.replace("/", "_")
        for mode in ("quality", "speedup"):
            path = output / mode / directory / "results.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record), encoding="utf-8")

    runs = []
    for study, grid in expected["sensitivity_grids"].items():
        for value in grid["values"]:
            for pair in expected["table1"]:
                model, dataset = pair.split("/")
                runs.append(
                    {
                        "study": study,
                        "value": value,
                        "model": model,
                        "dataset": dataset,
                        "metrics": {
                            "performance": {"speedup": 2.0 + float(value) / (1000 if value > 5 else 100)},
                            "quality": {"change": {"psnr_degradation_pct": float(value) / 1000}},
                        },
                    }
                )
    sensitivity = output / "sensitivity/results.json"
    sensitivity.parent.mkdir(parents=True)
    sensitivity.write_text(json.dumps({"runs": runs}), encoding="utf-8")

    physical = output / "physical/asap7"
    physical.mkdir(parents=True)
    raw = {"metrics": {"area_mm2": 1.0, "total_power_w": 0.5, "max_frequency_mhz": 100.0}}
    scaled = {"scaled_metrics": {"area_mm2": 31.82, "total_power_w": 1.333, "max_frequency_mhz": 78.8}}
    (physical / "ppa.json").write_text(json.dumps(raw), encoding="utf-8")
    (physical / "ppa_28nm_estimated.json").write_text(json.dumps(scaled), encoding="utf-8")


def test_report_generator_emits_tables_vector_and_preview_figures(tmp_path):
    from scripts.generate_report import generate

    output = tmp_path / "evidence"
    report = tmp_path / "report"
    write_fixture(output)
    record = generate(output, report)

    assert (report / "reproduction_report.md").is_file()
    for name in (
        "table1_quality.csv",
        "table2_fsdr.csv",
        "table3_saes.csv",
        "hardware_comparison.csv",
        "figure_catalog.json",
    ):
        assert (report / name).stat().st_size > 0
    assert record["surface_class"] == "appendix"
    assert not record["figures"]
    markdown = (report / "reproduction_report.md").read_text(encoding="utf-8")
    assert "NOT_CLAIMED_NO_ORIN_EVIDENCE" in markdown


def test_hardware_report_handles_a_resource_downgrade_without_ppa_files(tmp_path):
    from scripts.generate_report import hardware_table

    expected = json.loads((ROOT / "artifact/expected_results.json").read_text())
    report = tmp_path / "report"
    report.mkdir()
    record = hardware_table(
        tmp_path,
        report,
        expected,
        {
            "physical_asap7": "NOT_CLAIMED_RESOURCE_LIMIT",
            "deepscale": "NOT_CLAIMED_NO_PHYSICAL_INPUT",
        },
    )

    assert record["rows"] == []
    assert record["sources"] == []
    assert (report / "hardware_comparison.csv").is_file()
