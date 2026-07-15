import csv
import json

import pytest


def test_export_trace_splits_crossing_cachelines_and_hashes_source(tmp_path):
    from hardware.dram.export_trace import export

    source = tmp_path / "events.jsonl"
    source.write_text(
        "\n".join(
            (
                json.dumps({"cycle": 0, "op": "read", "address": "0x103f", "bytes": 2}),
                json.dumps({"cycle": 4, "op": "write", "address": 0x2000, "bytes": 64}),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "ramulator.trace"
    record = export(source, output, line_bytes=64)

    assert output.read_text().splitlines() == [
        "LD 0x0000000000001000",
        "LD 0x0000000000001040",
        "ST 0x0000000000002000",
    ]
    assert record["request_count"] == 3
    assert record["source"]["sha256"]
    assert record["trace"]["sha256"]


def test_export_trace_uses_relocatable_paths_for_archived_evidence(tmp_path):
    from hardware.dram.export_trace import export

    source = tmp_path / "memory-events.jsonl"
    source.write_text(
        '{"cycle":0,"op":"read","address":0,"bytes":32}\n',
        encoding="utf-8",
    )
    record = export(source, tmp_path / "ramulator.trace", evidence_root=tmp_path)

    assert record["source"]["path"] == "memory-events.jsonl"
    assert record["trace"]["path"] == "ramulator.trace"
    assert record["source"]["path_base"] == "output_dir"


def test_export_trace_rejects_invalid_or_nonmonotonic_events(tmp_path):
    from hardware.dram.export_trace import export

    source = tmp_path / "events.jsonl"
    source.write_text(
        '{"cycle":2,"op":"read","address":0,"bytes":64}\n'
        '{"cycle":1,"op":"read","address":64,"bytes":64}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="nondecreasing"):
        export(source, tmp_path / "trace")


def test_convert_ramulator_lpddr5_commands_is_explicit_and_drampower_ordered(tmp_path):
    from hardware.dram.convert_commands import convert

    source = tmp_path / "ramulator.csv.ch0"
    source.write_text(
        "clock,command,Channel,Rank,BankGroup,Bank,Row,Column,type,source\n"
        "0,ACT1,0,0,2,3,10,0,0,0\n"
        "1,ACT2,0,0,2,3,10,0,0,0\n"
        "2,CAS_RD,0,0,2,3,10,8,0,0\n"
        "3,RD,0,0,2,3,10,8,0,0\n"
        "9,PREpb,0,0,2,3,10,0,0,0\n",
        encoding="utf-8",
    )
    output = tmp_path / "drampower.csv"
    summary = convert(source, output)
    rows = list(csv.reader(output.open()))

    assert rows[0] == ["1", "ACT", "0", "2", "3", "10", "0"]
    assert rows[1][:7] == ["3", "RD", "0", "2", "3", "10", "8"]
    assert rows[1][7] == "0x0000000000000000"
    assert rows[-1] == ["10", "END", "0", "0", "0", "0", "0"]
    assert summary == {"converted_commands": 3, "ignored_ca_phases": 2}


def test_convert_rejects_unmapped_command(tmp_path):
    from hardware.dram.convert_commands import convert

    source = tmp_path / "bad.csv"
    source.write_text(
        "clock,command,Channel,Rank,BankGroup,Bank,Row,Column,type,source\n"
        "0,UNKNOWN,0,0,0,0,0,0,0,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported"):
        convert(source, tmp_path / "out.csv")


def test_ramulator_validation_rejects_undrained_or_zero_latency():
    from hardware.dram.run_ramulator import validate_metrics

    valid = {
        "memory_cycles": 40,
        "submitted_requests": 2,
        "completed_requests": 2,
        "read_requests": 1,
        "write_requests": 1,
        "average_read_latency_cycles": 12.0,
    }
    validate_metrics(valid, "1,RD,0\n2,WR,0\n")

    with pytest.raises(RuntimeError, match="fully drained"):
        validate_metrics({**valid, "completed_requests": 1}, "1,RD,0\n")
    with pytest.raises(RuntimeError, match="zero read latency"):
        validate_metrics({**valid, "average_read_latency_cycles": 0}, "1,RD,0\n2,WR,0\n")


def test_drampower_memspec_adaptation_is_explicit_and_hashed(tmp_path):
    from hardware.dram.run_drampower import (
        LPDDR5_TIMING_COMPATIBILITY_FIELDS,
        materialize_memspec,
        sha256_file,
    )

    root = tmp_path / "DRAMPower"
    source = root / "tests/tests_drampower/resources/lpddr5.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps({"memspec": {"memtimingspec": {"tCK": 1.0}}}) + "\n",
        encoding="utf-8",
    )
    original_hash = sha256_file(source)
    original, derived = materialize_memspec(
        root, tmp_path / "output", {"lpddr5_memspec_sha256": original_hash}
    )

    assert original == source
    assert sha256_file(original) == original_hash
    timing = json.loads(derived.read_text())["memspec"]["memtimingspec"]
    assert {field: timing[field] for field in LPDDR5_TIMING_COMPATIBILITY_FIELDS} == {
        field: 0 for field in LPDDR5_TIMING_COMPATIBILITY_FIELDS
    }
    assert sha256_file(derived) != original_hash
