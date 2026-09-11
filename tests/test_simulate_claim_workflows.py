from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_simulation_covers_full_profile_and_is_rejected_as_nonclaim(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    (root / "artifact").mkdir(parents=True)
    (root / "artifact/evaluation_protocol.json").write_text(
        json.dumps(
            {
                "pairs": {
                    f"transplat/{dataset}": {"sample_count": count}
                    for dataset, count in {"re10k": 3, "acid": 2, "dl3dv": 1}.items()
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "simulation"
    from scripts.simulate_claim_workflows import build, validate

    manifest = build(output, root=root, profile="full")
    result = validate(output)

    assert manifest["claim_eligible"] is False
    assert manifest["total_samples"] == 18
    assert result["status"] == "PASS"
    assert result["pairs"] == 9
    assert result["claim_eligible"] is False
    predictions = json.loads((output / "workflow-predictions.json").read_text())
    assert predictions["simulation_only"] is True
    assert set(predictions["quality"]) == set(manifest["pair_results"])
    assert set(predictions["mechanism"]) == set(manifest["pair_results"])
    assert set(predictions["performance"]) == set(manifest["pair_results"])


def test_simulation_output_cannot_be_written_inside_release_roots(tmp_path: Path) -> None:
    from scripts.simulate_claim_workflows import ROOT, build

    root = tmp_path / "checkout"
    (root / "artifact").mkdir(parents=True)
    (root / "artifact/evaluation_protocol.json").write_text(
        json.dumps(
            {
                "pairs": {
                    f"transplat/{dataset}": {"sample_count": 1}
                    for dataset in ("re10k", "acid", "dl3dv")
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside artifact"):
        build(root / "artifact" / "simulation-test", root=root)


def test_simulation_does_not_use_checked_in_pair_targets(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    artifact = root / "artifact"
    artifact.mkdir(parents=True)
    (artifact / "evaluation_protocol.json").write_text(
        json.dumps(
            {
                "pairs": {
                    f"transplat/{dataset}": {"sample_count": 1}
                    for dataset in ("re10k", "acid", "dl3dv")
                }
            }
        ),
        encoding="utf-8",
    )
    (artifact / "expected_results.json").write_text(
        json.dumps(
            {
                "table1": {
                    "transplat/re10k": {
                        "baseline": [31.0, 0.91, 0.11],
                        "scarf": [30.9, 0.909, 0.111],
                    }
                },
                "mechanisms": {
                    "transplat/re10k": {
                        "guided_rate": 0.72,
                        "level0_rate": 0.17,
                        "level1_rate": 0.15,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (artifact / "reference_results").mkdir()
    (artifact / "reference_results/orin_nx_reference.csv").write_text(
        "pair,orin_nx_normalized,scarf_dataflow_on_orin_nx_normalized,scarf_asic_speedup\n"
        "transplat/re10k,1.0,1.23,3.08\n",
        encoding="utf-8",
    )

    from scripts.simulate_claim_workflows import build

    output = tmp_path / "simulation"
    build(output, root=root, profile="full")
    predictions = json.loads((output / "workflow-predictions.json").read_text())
    # Simulation is a schema rehearsal, never a copy of paper targets.
    assert predictions["quality"]["transplat/re10k"]["baseline"]["psnr_db"] != 31.0
    assert predictions["mechanism"]["transplat/re10k"]["guided_rate"] != 0.72
    # The non-claim rehearsal may display the archived Orin reference row as
    # a comparison baseline; it remains simulation-only and cannot be claimed.
    assert predictions["performance"]["transplat/re10k"]["scarf_asic_speedup"] == 3.08


def test_simulation_validation_rejects_incomplete_result_without_hashes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkout"
    (root / "artifact").mkdir(parents=True)
    (root / "artifact/evaluation_protocol.json").write_text(
        json.dumps(
            {
                "pairs": {
                    f"transplat/{dataset}": {"sample_count": 1}
                    for dataset in ("re10k", "acid", "dl3dv")
                }
            }
        ),
        encoding="utf-8",
    )
    from scripts.simulate_claim_workflows import build, validate

    output = tmp_path / "simulation"
    build(output, root=root, profile="full")
    result_path = output / "results/transplat_re10k.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["sample_count"] = 2
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="sample count mismatch"):
        validate(output)


def test_file_tree_layout_materializes_sample_lineage_without_hashes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkout"
    (root / "artifact").mkdir(parents=True)
    (root / "artifact/evaluation_protocol.json").write_text(
        json.dumps(
            {
                "pairs": {
                    f"transplat/{dataset}": {"sample_count": count}
                    for dataset, count in {"re10k": 2, "acid": 1, "dl3dv": 1}.items()
                }
            }
        ),
        encoding="utf-8",
    )
    from scripts.simulate_claim_workflows import build, validate

    output = tmp_path / "simulation-tree"
    manifest = build(output, root=root, profile="full", layout="file-tree")
    result = validate(output)

    assert manifest["file_tree"]["sample_result_files"] == 36
    assert manifest["file_tree"]["timing_trace_files"] == 12
    assert result["layout"] == "file-tree"
    sample = output / "speedup/transplat_re10k/samples/sample_00001"
    assert (sample / "results.json").is_file()
    assert (sample / "orin-evidence/measurement.json").is_file()
    assert (sample / "timing-backend/manifest.json").is_symlink()
    assert (output / "input-manifest.json").is_file()
    assert (output / "FORMAT_SPEC.md").is_file()
    record = json.loads((sample / "results.json").read_text(encoding="utf-8"))
    assert record["hashes"] == "not_computed"
