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


def test_report_rejects_mismatched_quality_and_mechanism_execution_bindings():
    from scripts.generate_report import (
        combined_quality_mechanism_reasons,
        quality_mechanism_binding_reason,
    )

    quality = {
        "provenance": {
            "mechanism_config_sha256": "a" * 64,
            "checkpoint": {"sha256": "b" * 64},
            "evaluation": {
                "sample_selection_sha256": "c" * 64,
                "execution_trace_set_sha256": "d" * 64,
            },
        }
    }
    mechanism = json.loads(json.dumps(quality))

    assert quality_mechanism_binding_reason(quality, mechanism) is None
    mechanism["provenance"]["checkpoint"]["sha256"] = "e" * 64
    reason = quality_mechanism_binding_reason(quality, mechanism)
    assert reason == "quality/mechanism aggregate binding mismatch: checkpoint_sha256"

    quality_reason, mechanism_reason = combined_quality_mechanism_reasons(
        quality,
        mechanism,
        None,
        "SAES S2/S3 sparse execution is not verified",
    )
    assert "cannot be combined" in quality_reason
    assert mechanism_reason == "SAES S2/S3 sparse execution is not verified"


def test_report_claim_evidence_requires_a_clean_paired_mechanism_record():
    from scripts.generate_report import (
        claim_quality_mechanism_pair_reasons,
        report_execution_evidence_reason,
    )

    quality_reason, mechanism_reason = claim_quality_mechanism_pair_reasons(
        {"quality": {}}, None, None, None
    )
    assert quality_reason == (
        "claim-quality aggregate requires a matching claimable mechanism aggregate"
    )
    assert mechanism_reason is None

    quality_reason, mechanism_reason = claim_quality_mechanism_pair_reasons(
        None, {"ablation": {}}, None, None
    )
    assert quality_reason is None
    assert mechanism_reason == (
        "claimable mechanism aggregate requires a matching claim-quality aggregate"
    )

    dirty_claim = {
        "schema_version": "2.1",
        "evidence_class": "deterministic_execution",
        "provenance": {
            "dataset": {"paper_result_eligible": True},
            "execution_contract": {
                "run_class": "claim",
                "saes_materialization": "representative",
            },
            "git_dirty": True,
        },
        "quality": {},
        "performance": {},
    }
    assert report_execution_evidence_reason(
        dirty_claim, required_evidence_class="deterministic_execution"
    ) == "claim-facing record was generated from a dirty worktree"


def test_table_builder_blocks_an_unpaired_otherwise_claimable_quality_record(
    tmp_path, monkeypatch
):
    import scripts.generate_report as report

    quality_path = tmp_path / "quality" / "transplat_re10k" / "results.json"
    quality_path.parent.mkdir(parents=True)
    quality_path.write_text(
        json.dumps(
            {
                "provenance": {
                    "model": "transplat",
                    "dataset": {"name": "re10k"},
                },
                "quality": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(report, "require_report_result", lambda *_args: None)
    monkeypatch.setattr(
        report, "report_execution_evidence_reason", lambda *_args, **_kwargs: None
    )

    report_dir = tmp_path / "report"
    report_dir.mkdir()
    tables = report.build_tables(tmp_path, report_dir)

    assert tables["quality"] == []
    assert tables["strict_quality_evidence"] == {
        "transplat/re10k": (
            "claim-quality aggregate requires a matching claimable mechanism "
            "aggregate"
        )
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


def test_figure8_rejects_bare_orin_source_records(tmp_path):
    from scripts.generate_report import generate

    output = tmp_path / "evidence"
    report = tmp_path / "report"
    expected = json.loads((ROOT / "artifact/expected_results.json").read_text())
    for pair, target in expected["table1"].items():
        record = result(pair, target, expected["mechanisms"][pair])
        record["performance"]["baseline_source"] = "orin_nx_cuda_events"
        path = output / "speedup" / pair.replace("/", "_") / "results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")

    catalog = generate(output, report, figures="figure8")
    figure8 = next(row for row in catalog["results"] if row["id"] == "figure8")

    assert figure8["status"] != "PASS"
    assert (report / "figure8_speedup.csv").read_text(encoding="utf-8").splitlines() == [
        "pair,speedup"
    ]


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


def test_figure11_requires_its_table2_and_table3_support(tmp_path):
    import scripts.generate_report as report

    output = tmp_path / "evidence"
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    raw = output / "mechanisms" / "transplat_re10k" / "results.json"
    raw.parent.mkdir(parents=True)
    raw.write_text("{}", encoding="utf-8")
    contract = json.loads((ROOT / "artifact/evaluation_catalog.json").read_text())
    rows = report.build_result_catalog(
        output=output,
        report_dir=report_dir,
        contract=contract,
        tables={
            "speedup": [],
            "strict_speedup_evidence": {},
            "quality": [],
            "strict_quality_evidence": {},
            "ablation": [
                {
                    "pair": pair,
                    "fsdr_speedup": 1.1,
                    "saes_speedup": 1.1,
                    "combined_speedup": 1.2,
                }
                for pair in report.REQUIRED_PAIRS
            ],
            "strict_mechanism_evidence": {},
            "fsdr": [],
            "saes": [],
        },
        sensitivity={"_complete": False},
        hardware={"rows": []},
        figure_exports=[],
        selected_ids={"figure11", "table2", "table3"},
        claims={"figure8": "NOT_CLAIMED"},
    )

    assert next(row for row in rows if row["id"] == "figure11")["status"] == "FAIL"


def test_analytic_or_unverified_saes_mechanisms_cannot_pass_key_tables(
    tmp_path,
):
    from scripts.generate_report import (
        generate,
        strict_saes_mechanism_evidence_reason,
    )

    output = tmp_path / "evidence"
    report = tmp_path / "report"
    pairs = (
        "transplat/re10k",
        "transplat/acid",
        "transplat/dl3dv",
        "mvsplat/re10k",
        "mvsplat/acid",
        "mvsplat/dl3dv",
        "depthsplat/re10k",
        "depthsplat/acid",
        "depthsplat/dl3dv",
    )
    for pair in pairs:
        model, dataset = pair.split("/")
        record = {
            "schema_version": "2.1",
            "evidence_class": "deterministic_execution",
            "provenance": {
                "model": model,
                "dataset": {"name": dataset, "paper_result_eligible": True},
                "execution_contract": {
                    "run_class": "claim",
                    "saes_materialization": "representative",
                },
                "evaluation": {"kind": "dataset_aggregate", "sample_count": 1},
            },
            "quality": {
                "baseline": {"psnr_db": 20.0, "ssim": 0.7, "lpips": 0.3},
                "scarf": {"psnr_db": 19.0, "ssim": 0.69, "lpips": 0.31},
            },
            "performance": {"cycle_source": "scarf_component_simulators:sample_mean"},
            "ablation": {
                "asic": {"eff_total": 1000},
                "asic_fsdr": {"eff_total": 900},
                "asic_saes": {"eff_total": 950},
                "asic_fsdr_saes": {"eff_total": 850},
            },
            "events": {
                "fsdr": {
                    "total_pixels": 20,
                    "guided_pixels": 10,
                    "guided_top1_covered": 9,
                    "guided_top1_missed": 1,
                    "discrete_top1_available": True,
                    "depth_evaluations_available": True,
                    "full_depth_evaluations": 20,
                    "executed_depth_evaluations": 10,
                    "feature_buffer_bytes_available": True,
                    "feature_buffer_bytes_baseline": 40,
                    "feature_buffer_bytes_actual": 20,
                },
                "saes": {
                    "tile_path_available": True,
                    "total_tiles": 10,
                    "level0_tiles": 2,
                    "level1_tiles": 3,
                    "full_tiles": 5,
                    "gaussian_counts_available": True,
                    "baseline_gaussians": 100,
                    "actual_gaussians": 50,
                    "s2_evaluations_available": True,
                    "full_s2_evaluations": 100,
                    "executed_s2_evaluations": 50,
                    "execution_dependency": {
                        "s2_s3_sparse_execution_verified": False
                    },
                    "s2_s3_saving": {"s2": 0.0, "s3": 0.0},
                },
            },
            "fsdr_saes": {
                "fsdr": {"guided_rate": 0.5, "in_window_rate": 0.9},
                "saes": {
                    "level0_ratio": 0.2,
                    "level1_ratio": 0.3,
                    "full_ratio": 0.5,
                    "modification_ratio": 0.5,
                    "hardware_accounting": {
                        "timing_class": "analytic_no_overlap_not_rtl_cycle_equivalent",
                        "assumptions": {"rtl_cycle_equivalent": False},
                    },
                },
                "preservation": {"saes_low_var_agree": 0.9},
            },
        }
        assert (
            strict_saes_mechanism_evidence_reason(record)
            == "SAES S2/S3 sparse execution is not verified"
        )
        for mode in ("mechanisms", "quality"):
            path = output / mode / pair.replace("/", "_") / "results.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record), encoding="utf-8")

    catalog = generate(output, report, figures="table1,figure11,table3")
    rows = {row["id"]: row for row in catalog["results"]}

    assert rows["table1"]["status"] == "FAIL"
    assert "non-claimable" in rows["table1"]["reason"]
    for result_id in ("figure11", "table3"):
        assert rows[result_id]["status"] == "FAIL"
        assert "non-claimable analytic" in rows[result_id]["reason"]


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
    from scripts.result_record import CLAIM_TIMING_CYCLE_SOURCE, build_result_record
    from scripts.saes_execution_identity import build_saes_execution_identity
    from PIL import Image
    from scripts.generate_report import render_worstcase

    output = tmp_path / "evidence"
    report = tmp_path / "report"
    report.mkdir()
    pair = output / "quality/mvsplat_re10k"
    result = pair / "samples/sample_00003/results.json"
    result.parent.mkdir(parents=True)
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    dataset_manifest = tmp_path / "dataset.manifest.json"
    dataset_manifest.write_text("{}\n", encoding="utf-8")
    timing = {
        "schema_version": "source-bound-claim-timing-v1",
        "timing_class": "rtl_cycle_equivalent_source_bound",
        "rtl_cycle_equivalent": True,
        "clock_mhz": 1000,
        "trace": {"path": "timing-trace/sample.json", "sha256": "d" * 64},
        "variants": {
            "asic": {"total_cycles": 500},
            "asic_fsdr": {"total_cycles": 400},
            "asic_saes": {"total_cycles": 375},
            "asic_fsdr_saes": {"total_cycles": 300},
        },
        "combined_stage_cycles": {"s1": 50, "s2": 100, "s3": 75, "s4": 75},
    }
    quality_view = {
        "target_index": 7,
        "baseline": {"psnr_db": 28.0, "ssim": 0.9, "lpips": 0.1},
        "scarf": {"psnr_db": 27.9, "ssim": 0.89, "lpips": 0.11},
    }
    identity = build_saes_execution_identity()
    source_record = build_result_record(
        model="mvsplat",
        dataset="re10k",
        checkpoint=checkpoint,
        checkpoint_load={
            "matched_tensors": 1,
            "matched_checkpoint_numel_fraction": 1.0,
        },
        environment={"profile": "classic", "digest_sha256": "c" * 64},
        dataset_manifest=dataset_manifest,
        dataset_representation="re10k-native",
        dataset_tree_sha256="a" * 64,
        device={"type": "cuda", "name": "test"},
        seed=0,
        quality={"baseline": quality_view["baseline"], "scarf": quality_view["scarf"]},
        quality_views=[quality_view],
        baseline_cycles=1000,
        scarf_cycles=300,
        cycles={"feature": 50, "depth": 100, "gaussian": 75, "ggu": 75},
        cycle_source=CLAIM_TIMING_CYCLE_SOURCE,
        ablation={key: {} for key in timing["variants"]},
        fsdr_saes={
            "saes": {
                "saes_execution_identity": identity,
                "route_sha256": identity["route_sha256"],
            }
        },
        command=["python", "scripts/demo.py"],
        runtime_assets={"VGG16": {"sha256": "b" * 64}},
        sample_identity={
            "scene": "scene-3",
            "context_indices": [0, 1],
            "target_indices": [7],
        },
        run_class="claim",
        paper_result_eligible=False,
        stage_records={
            stage: {
                "cycles": cycles,
                "useful_mmcu_slots": 1,
                "scheduled_mmcu_slots": 1,
                "mmcu_slots_available": True,
                "source": "rtl_trace",
            }
            for stage, cycles in timing["combined_stage_cycles"].items()
        },
        claim_timing=timing,
    )
    result.write_text(
        json.dumps(source_record),
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
    assert evidence["selected"]["fsdr"]["latency_cycles"] == {
        "no_optimization": {"End-to-end": 500.0},
        "optimized": {"End-to-end": 400.0},
    }


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
