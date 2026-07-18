import json
import subprocess
import sys
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_plot_style_loads_without_invalid_color_warnings():
    import matplotlib.pyplot as plt

    stderr = StringIO()
    with redirect_stderr(stderr):
        plt.style.use(ROOT / "artifact/plot_style.mplstyle")

    assert "Bad value" not in stderr.getvalue()


def test_table4_contract_uses_only_the_final_paper_rows_in_paper_order():
    from hardware.iflow.paper_table4 import TABLE4_COMPONENTS, TABLE4_LABELS

    expected = json.loads((ROOT / "artifact/expected_results.json").read_text())
    hierarchy = expected["hardware_comparison_target"]["hierarchy"]

    assert tuple(hierarchy) == TABLE4_COMPONENTS
    assert [TABLE4_LABELS[key] for key in TABLE4_COMPONENTS] == [
        "MVU",
        "MMCU",
        "VectorALU",
        "BilinearUnit",
        "NormUnit",
        "ActivationUnit",
        "GGU Array (x32 PEs)",
        "PositionCalc",
        "CovBuilder",
        "SH_OPGenerator",
        "FSDR Subsystem",
        "LSHHashUnit",
        "CAM Array (32-entry)",
        "FSDR Controller",
        "On-chip Buffers",
        "Weight Buffer (128 KB)",
        "Feature Buffer (256 KB)",
        "Tile Buffer (64 KB)",
        "Control & Clock",
        "Control + Interconnect",
        "PLL + Clock tree",
        "I/O & PHY",
        "I/O + LPDDR4X PHY",
        "Routing / filler",
        "Total die",
    ]
    assert {"mvu_aux", "control_buffer_logic", "other_logic", "sram_proxy"}.isdisjoint(
        hierarchy
    )


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
    from hardware.iflow.paper_table4 import TABLE4_COMPONENTS, UNMODELED_COMPONENTS

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
    expected_hierarchy = expected["hardware_comparison_target"]["hierarchy"]
    hierarchy = {
        component: {
            "area_mm2": None if component in UNMODELED_COMPONENTS else 0.1,
            "dynamic_power_w": (
                None
                if component in UNMODELED_COMPONENTS
                or component == "on_chip_buffers"
                or component.startswith("on_chip_buffers_")
                or component == "routing_filler"
                or component == "total_die"
                else 0.01
            ),
            "static_power_w": (
                None
                if component in UNMODELED_COMPONENTS
                or component == "on_chip_buffers"
                or component.startswith("on_chip_buffers_")
                or component == "routing_filler"
                or component == "total_die"
                else 0.001
            ),
            "total_power_w": (
                0.5
                if component == "total_die"
                else None
                if component in UNMODELED_COMPONENTS
                or component == "on_chip_buffers"
                or component.startswith("on_chip_buffers_")
                or component == "routing_filler"
                else 0.011
            ),
        }
        for component in TABLE4_COMPONENTS
    }
    raw = {
        "physical_valid": True,
        "evidence_type": "asap7_predictive_postroute",
        "metrics": {
            "area_mm2": 1.0,
            "total_power_w": 0.5,
            "max_frequency_mhz": 100.0,
            "hierarchy": hierarchy,
        },
        "scope": {"io_phy_included": False},
    }
    scaled = {
        "raw": raw,
        "scaled_metrics": {
            "area_mm2": 31.82,
            "total_power_w": 1.333,
            "max_frequency_mhz": 78.8,
        },
        "scaled_hierarchy": {
            component: {
                metric: None if value is None else value * 2
                for metric, value in values.items()
            }
            for component, values in hierarchy.items()
        },
        "validation": {"raw_preserved": True, "foundry_measurement": False},
    }
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
        "table4_hierarchy.csv",
        "figure_catalog.json",
    ):
        assert (report / name).stat().st_size > 0
    assert record["surface_class"] == "appendix"
    assert record["schema_version"] == "2.0"
    assert {item["id"] for item in record["results"]} == {
        *(f"figure{index}" for index in range(8, 17)),
        *(f"table{index}" for index in range(1, 5)),
    }
    by_id = {item["id"]: item for item in record["results"]}
    assert by_id["figure8"]["status"] == "NOT_RUN"
    assert by_id["figure8"]["evidence_class"] == "independent_measurement"
    assert by_id["table1"]["evidence_class"] == "deterministic_execution"
    markdown = (report / "reproduction_report.md").read_text(encoding="utf-8")
    assert "CLAIMED_AWAITING_INDEPENDENT_ORIN_EVALUATION" in markdown


def test_table2_never_substitutes_continuous_in_window_rate_for_discrete_top1(tmp_path):
    from scripts.generate_report import generate

    output = tmp_path / "evidence"
    report = tmp_path / "report"
    write_fixture(output)
    generate(output, report)

    rows = (report / "table2_fsdr.csv").read_text(encoding="utf-8").splitlines()
    assert rows
    header = rows[0].split(",")
    top1_index = header.index("top1_coverage")
    assert all(row.split(",")[top1_index] == "" for row in rows[1:])


def test_report_generator_rejects_assignment_consensus_result_input(tmp_path):
    from scripts.generate_report import generate

    output = tmp_path / "evidence"
    report = tmp_path / "report"
    write_fixture(output)
    path = output / "quality" / "transplat_re10k" / "results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["provenance"]["execution_contract"] = {
        "run_class": "diagnostic",
        "saes_materialization": (
            "assignment-consensus-adapter-pseudo-descriptor-diagnostic"
        ),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="assignment-consensus pseudo descriptors"):
        generate(output, report)


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
    assert (report / "table4_hierarchy.csv").is_file()


def test_hardware_report_emits_hierarchy_and_keeps_proxy_scope_explicit(tmp_path):
    from scripts.generate_report import hardware_table

    expected = json.loads((ROOT / "artifact/expected_results.json").read_text())
    write_fixture(tmp_path)
    report = tmp_path / "report"
    report.mkdir()

    record = hardware_table(tmp_path, report, expected, {})

    rows = record["hierarchy_rows"]
    from hardware.iflow.paper_table4 import TABLE4_METRICS

    assert len(rows) == sum(len(TABLE4_METRICS[component]) for component in TABLE4_METRICS)
    excluded = [
        row
        for row in rows
        if row["component_key"] in {"io_phy", "io_lpddr4x_phy"}
    ]
    assert excluded
    assert all(row["public_proxy_scope"] == "not_modeled_in_public_proxy" for row in excluded)
    assert all(row["asap7_raw"] is None for row in excluded)
    sram_power = [
        row
        for row in rows
        if row["component_key"].startswith("on_chip_buffers")
        and row["metric"].endswith("power_w")
    ]
    assert all(row["asap7_raw"] is None for row in sram_power)
    assert all(row["deepscale_28nm_estimate"] is None for row in sram_power)
    assert all(row["comparison_is_pass_fail"] is False for row in rows)


def test_figure9_requires_activity_aware_power_and_bound_workload_dram(tmp_path):
    import hashlib
    from scripts.generate_report import hardware_table, render_efficiency_proxy

    expected = json.loads((ROOT / "artifact/expected_results.json").read_text())
    write_fixture(tmp_path)
    physical = tmp_path / "physical/asap7"
    raw_path = physical / "ppa.json"
    scaled_path = physical / "ppa_28nm_estimated.json"
    raw = json.loads(raw_path.read_text())
    vcd = physical / "activity.vcd"
    vcd.write_text("$date test $end\n", encoding="ascii")
    raw["scope"]["power_activity"] = "workload_vcd"
    raw["artifacts"] = {
        "activity_vcd": {
            "path": vcd.name,
            "path_base": "output_dir",
            "sha256": hashlib.sha256(vcd.read_bytes()).hexdigest(),
        }
    }
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    scaled = json.loads(scaled_path.read_text())
    scaled["raw"] = raw
    scaled_path.write_text(json.dumps(scaled), encoding="utf-8")

    for pair in expected["table1"]:
        model, dataset = pair.split("/")
        software_path = tmp_path / "quality" / pair.replace("/", "_") / "results.json"
        selection_hash = hashlib.sha256(pair.encode()).hexdigest()
        software = {
            "schema_version": "2.0",
            "evidence_class": "deterministic_execution",
            "provenance": {
                "model": model,
                "dataset": {"name": dataset, "paper_result_eligible": True},
                "evaluation": {
                    "kind": "dataset_aggregate",
                    "sample_selection_sha256": selection_hash,
                },
            },
            "performance": {"scarf_cycles": 1_000_000},
        }
        software_path.write_text(json.dumps(software), encoding="utf-8")
        dram_path = tmp_path / "dram/workloads" / pair.replace("/", "_") / "results.json"
        dram_path.parent.mkdir(parents=True, exist_ok=True)
        dram_path.write_text(
            json.dumps(
                {
                    "evidence_type": "public_memory_system_proxy",
                    "paper_lpddr4x_reproduced": False,
                    "metrics": {
                        "drampower_offchip_energy_per_inference_j": 2.5e-4
                    },
                    "scope": {"claim": "per_inference_workload_proxy"},
                    "workload": {
                        "model": model,
                        "dataset": dataset,
                        "sample_selection_sha256": selection_hash,
                        "software_result": {
                            "sha256": hashlib.sha256(
                                software_path.read_bytes()
                            ).hexdigest()
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

    report = tmp_path / "report"
    report.mkdir()
    hardware = hardware_table(tmp_path, report, expected, {})
    rendered = render_efficiency_proxy(tmp_path, report, hardware, expected)

    assert rendered is not None
    assert rendered["id"] == "figure9"
    evidence = json.loads((report / "figure9_public_proxy.json").read_text())
    assert len(evidence["rows"]) == 9
    assert evidence["definitions"]["paper_targets"].endswith(
        "not used for public-proxy PASS/FAIL"
    )
    assert all(
        row["paper_values_are_normalized_targets_only"] is True
        for row in evidence["rows"]
    )


def test_figure9_rejects_the_functional_dram_smoke_record(tmp_path):
    from scripts.generate_report import hardware_table, render_efficiency_proxy

    expected = json.loads((ROOT / "artifact/expected_results.json").read_text())
    write_fixture(tmp_path)
    smoke = tmp_path / "dram/results.json"
    smoke.parent.mkdir()
    smoke.write_text(
        json.dumps({"scope": {"claim": "functional public proxy only"}}),
        encoding="utf-8",
    )
    report = tmp_path / "report"
    report.mkdir()
    hardware = hardware_table(tmp_path, report, expected, {})

    assert render_efficiency_proxy(tmp_path, report, hardware, expected) is None


def test_report_cli_accepts_valid_figure_selection_and_rejects_unknown_ids(tmp_path):
    output = tmp_path / "evidence"
    report = tmp_path / "report"
    write_fixture(output)
    script = ROOT / "scripts/generate_report.py"

    selected = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(output),
            "--output-dir",
            str(report),
            "--figures",
            "table1,figure11",
        ],
        capture_output=True,
        text=True,
    )
    assert selected.returncode == 0, selected.stderr
    catalog = json.loads((report / "figure_catalog.json").read_text())
    by_id = {item["id"]: item for item in catalog["results"]}
    assert by_id["table1"]["selected"] is True
    assert by_id["figure11"]["selected"] is True
    assert by_id["figure13"]["selected"] is False

    unknown = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(output),
            "--output-dir",
            str(tmp_path / "unknown"),
            "--figures",
            "figure99",
        ],
        capture_output=True,
        text=True,
    )
    assert unknown.returncode != 0
    assert "unknown result id" in unknown.stderr


def test_worstcase_renderer_selects_by_raw_loss_and_binds_images(tmp_path):
    import hashlib
    from PIL import Image
    from scripts.generate_report import render_worstcase

    output = tmp_path / "evidence"
    report = tmp_path / "report"
    report.mkdir()
    pair = output / "quality/mvsplat_re10k"
    result = pair / "samples/sample_00003/results.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(
            {
                "ablation": {
                    name: {
                        "eff_feature": 10,
                        "eff_dp_core": 20 if name == "asic" else 12,
                        "eff_gauss_gen": 30 if name == "asic" else 18,
                    }
                    for name in ("asic", "asic_fsdr", "asic_saes")
                }
            }
        ),
        encoding="utf-8",
    )
    for kind, loss in (("fsdr", 0.4), ("saes", 0.03)):
        directory = pair / "worstcase" / kind
        directory.mkdir(parents=True)
        artifacts = {}
        for index, name in enumerate(("ground_truth", "reference", "optimized")):
            path = directory / f"{name}.png"
            Image.new("RGB", (4, 4), color=(index * 30, 20, 10)).save(path)
            artifacts[name] = {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "kind": f"{kind}_worstcase_view",
                    "loss_metric": "psnr_loss_db" if kind == "fsdr" else "lpips_increase",
                    "loss_value": loss,
                    "sample_index": 3,
                    "scene": "scene-3",
                    "target_index": 7,
                    "view_ordinal": 0,
                    "source_result": {
                        "path": "samples/sample_00003/results.json",
                        "sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
                    },
                    "artifacts": artifacts,
                }
            ),
            encoding="utf-8",
        )

    rendered = render_worstcase(output, report)

    assert rendered["id"] == "figure10"
    assert len(rendered["exports"]) == 3
    evidence = json.loads((report / "figure10_worstcase.json").read_text())
    assert evidence["selected"]["fsdr"]["loss_value"] == 0.4
    assert evidence["selected"]["saes"]["sample_index"] == 3
    assert evidence["selected"]["fsdr"]["error_map"]["maximum"] > 0


def test_utilization_renderer_requires_nine_real_slot_ratios(tmp_path):
    from scripts.generate_report import render_utilization

    expected = json.loads((ROOT / "artifact/expected_results.json").read_text())
    output = tmp_path / "evidence"
    report = tmp_path / "report"
    report.mkdir()
    for pair, targets in expected["figure12"]["utilization"].items():
        model, dataset = pair.split("/")
        path = output / "utilization" / f"{model}_{dataset}" / "results.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "provenance": {"model": model, "dataset": {"name": dataset}},
                    "performance": {
                        "stages": {
                            stage: {
                                "cycles": 10,
                                "useful_mmcu_slots": int(value * 1000),
                                "scheduled_mmcu_slots": 1000,
                                "mmcu_slots_available": True,
                                "source": "event_simulator",
                            }
                            for stage, value in targets.items()
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

    rendered = render_utilization(output, report, expected)

    assert rendered is not None
    assert rendered["id"] == "figure12"
    assert rendered["acceptance_pass"] is True
    rows = (report / "figure12_utilization.csv").read_text().splitlines()
    assert len(rows) == 28
