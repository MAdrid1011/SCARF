import csv
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "tools/render_paper_reference.py"


def load_renderer():
    spec = importlib.util.spec_from_file_location("render_paper_reference", RENDERER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_reference_preview_v3_covers_structured_paper_sources(tmp_path):
    renderer = load_renderer()
    source = ROOT / "artifact/expected_results.json"
    expected = json.loads(source.read_text(encoding="utf-8"))
    output = tmp_path / "preview"
    output.mkdir()

    renderer.render(expected, source, output)

    manifest = json.loads(
        (output / renderer.REFERENCE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "2.0"
    assert manifest["artifact_class"] == renderer.REFERENCE_ONLY_MARKER
    assert manifest["source"] == {
        "id": "expected_results",
        "path": "artifact/expected_results.json",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    source_ids = {record["id"] for record in manifest["sources"]}
    assert {
        "expected_results",
        "paper_evaluation_tex",
        "figure8_speedup",
        "figure8_latency",
        "figure9_energy",
        "figure9_area",
        "figure12_utilization",
        "figure13_cache_top_tran",
        "figure13_cache_bottom_depth",
        "figure11_rendered",
        "figure14_rendered",
        "figure15_rendered",
        "figure16_rendered",
    } <= source_ids
    for record in manifest["sources"]:
        assert not Path(record["path"]).is_absolute()
        path = ROOT / record["path"]
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    expected_copy = json.loads(
        (output / "expected_results_reference.json").read_text(encoding="utf-8")
    )
    assert expected_copy["artifact_class"] == renderer.REFERENCE_ONLY_MARKER
    assert expected_copy["values"] == expected

    expected_artifacts = {
        "expected_results_reference.json",
        "table1_reference.csv",
        "table2_fsdr_reference.csv",
        "table3_saes_reference.csv",
        "figure8_speedup_reference.csv",
        "figure8_latency_reference.csv",
        "figure8_summary_reference.csv",
        "figure9_energy_efficiency_reference.csv",
        "figure9_area_throughput_reference.csv",
        "figure11_ablation_reference.csv",
        "figure12_mmcu_utilization_reference.csv",
        "figure13_fsdr_cache_reference.csv",
        "figure13_source_annotations_reference.csv",
        "figures13_16_sensitivity_reference.csv",
        "table4_hierarchy_reference.csv",
        "table4_paper_cells_reference.csv",
        "table4_metadata_reference.csv",
        "unstructured_figure_data_reference.json",
        "paper_reference.md",
    }
    assert {record["path"] for record in manifest["artifacts"]} == expected_artifacts
    for record in manifest["artifacts"]:
        path = output / record["path"]
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    for path in output.iterdir():
        content = path.read_text(encoding="utf-8")
        assert renderer.REFERENCE_ONLY_MARKER in content
        assert str(ROOT) not in content

    assert len(csv_rows(output / "figure8_speedup_reference.csv")) == 9
    assert len(csv_rows(output / "figure8_latency_reference.csv")) == 18
    assert len(csv_rows(output / "figure9_energy_efficiency_reference.csv")) == 9
    assert len(csv_rows(output / "figure9_area_throughput_reference.csv")) == 9
    assert len(csv_rows(output / "table2_fsdr_reference.csv")) == 9
    assert len(csv_rows(output / "table3_saes_reference.csv")) == 9
    assert len(csv_rows(output / "figure12_mmcu_utilization_reference.csv")) == 9
    assert len(csv_rows(output / "figure13_fsdr_cache_reference.csv")) == 90
    assert len(csv_rows(output / "figure13_source_annotations_reference.csv")) == 15
    assert len(csv_rows(output / "table4_hierarchy_reference.csv")) == 25
    assert len(csv_rows(output / "table4_paper_cells_reference.csv")) == 25

    table2 = csv_rows(output / "table2_fsdr_reference.csv")
    assert table2[0]["Depth Evals Saved in S2 (M, percent)"] == "4.54 (54.2%)"
    assert table2[0]["Feature Buffer Reduction (MB, percent)"] == "415 (51.5%)"
    table3 = csv_rows(output / "table3_saes_reference.csv")
    assert table3[0]["S2 Evals Saved (M)"] == "1.45"
    figure8 = csv_rows(output / "figure8_summary_reference.csv")
    assert figure8[0]["value"] == "1.327778942988811"
    assert figure8[1]["value"] == "2.938788764956141"

    notices = json.loads(
        (output / "unstructured_figure_data_reference.json").read_text(
            encoding="utf-8"
        )
    )["values"]
    assert notices["figure10"]["status"] == "SOURCE_ONLY_NOT_TABULATED"
    assert "plot-versus-actual" in notices["figure10"]["reason"]
    assert notices["figure15"]["status"] == "PDF_ONLY_POINT_SERIES_UNAVAILABLE"
    assert "conflicts" in notices["figure15"]["reason"]

    for filename in expected_artifacts:
        if not filename.endswith(".csv"):
            continue
        rows = csv_rows(output / filename)
        assert rows
        assert {row["artifact_class"] for row in rows} == {
            renderer.REFERENCE_ONLY_MARKER
        }
        assert all(not Path(row["source_path"]).is_absolute() for row in rows)
