from pathlib import Path

import pytest


def make_artifacts(root: Path) -> None:
    (root / "rtl").mkdir(parents=True)
    (root / "traces").mkdir()
    (root / "sbt-test.log").write_text(
        "Total number of tests run: 8\nAll tests passed.\n", encoding="utf-8"
    )
    (root / "emit.log").write_text("Done.\n", encoding="utf-8")
    (root / "verilator-lint.log").write_text(
        "%Warning-UNUSEDSIGNAL: example\n", encoding="utf-8"
    )
    (root / "rtl" / "ScarfTop.sv").write_text("module ScarfTop; endmodule\n")
    (root / "traces" / "fsdr-cache.vcd").write_text("$date test $end\n")


def test_rtl_result_preserves_lint_warning_count_and_hashes(tmp_path, monkeypatch):
    import scripts.rtl_result as rtl_result

    make_artifacts(tmp_path)
    identity = {
        "git_commit": "a" * 40,
        "git_dirty": False,
        "source": "git",
        "source_tree_sha256": "e" * 64,
        "submodules": {
            "transplat": "b" * 40,
            "mvsplat": "c" * 40,
            "depthsplat": "d" * 40,
        },
    }
    monkeypatch.setattr(rtl_result, "source_identity", lambda *_: identity)
    mechanism = {
        "status": "preregistered",
        "manifest_sha256": "f" * 64,
        "candidate_records_sha256": None,
        "evaluation_disjoint": False,
        "expected_results_accessed": False,
        "global_configuration": True,
        "mechanism_config_sha256": "1" * 64,
    }
    monkeypatch.setattr(rtl_result, "load_mechanism_config", lambda: ({}, mechanism))

    record = rtl_result.build_result(tmp_path)
    assert record["status"] == "PASS"
    assert record["rtl"]["tests_passed"] == 8
    assert record["rtl"]["verilator_warning_count"] == 1
    assert len(record["artifacts"]["systemverilog"]["sha256"]) == 64
    assert record["provenance"] == {
        "git_commit": identity["git_commit"],
        "git_dirty": False,
        "source_identity": "git",
        "source_tree_sha256": identity["source_tree_sha256"],
        "submodules": identity["submodules"],
        "mechanism_config_sha256": mechanism["mechanism_config_sha256"],
        "calibration_provenance": {
            key: value
            for key, value in mechanism.items()
            if key != "mechanism_config_sha256"
        },
        "generated_at": record["provenance"]["generated_at"],
        "commands": record["provenance"]["commands"],
    }


def test_rtl_result_rejects_lint_errors(tmp_path):
    from scripts.rtl_result import build_result

    make_artifacts(tmp_path)
    (tmp_path / "verilator-lint.log").write_text("%Error: broken\n")
    with pytest.raises(ValueError, match="Verilator reported"):
        build_result(tmp_path)
