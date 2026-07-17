from pathlib import Path

import pytest


def test_resource_guard_rejects_active_vivado():
    from hardware.iflow.resource_guard import check_resources

    with pytest.raises(RuntimeError, match="Vivado"):
        check_resources(available_bytes=60 * 1024**3, active_processes=["vivado"])
    with pytest.raises(RuntimeError, match="Vivado"):
        check_resources(
            available_bytes=60 * 1024**3,
            active_processes=["/opt/Xilinx/2025.2/Vivado/bin/vrs -j 7"],
        )


def test_resource_guard_requires_48_gib_available():
    from hardware.iflow.resource_guard import check_resources

    with pytest.raises(RuntimeError, match="48 GiB"):
        check_resources(available_bytes=47 * 1024**3, active_processes=[])
    check = check_resources(available_bytes=48 * 1024**3, active_processes=[])
    assert check["pass"]
    assert check["minimum_available_bytes"] == 48 * 1024**3


def test_resource_guard_allows_an_explicit_low_memory_attempt():
    from hardware.iflow.resource_guard import check_resources

    check = check_resources(
        available_bytes=12 * 1024**3,
        active_processes=[],
        allow_low_memory_attempt=True,
    )

    assert check["pass"]
    assert check["mode"] == "low_memory_attempt"
    assert check["available_memory_threshold_met"] is False

    with pytest.raises(RuntimeError, match="Vivado"):
        check_resources(
            available_bytes=60 * 1024**3,
            active_processes=["vivado"],
            allow_low_memory_attempt=True,
        )


def test_iflow_run_wires_resource_guard_before_rtl_and_materialization():
    script = (Path(__file__).resolve().parents[1] / "hardware/iflow/run.sh").read_text()

    guard = script.index("resource_guard.py")
    assert guard < script.index('RTL_DIR="${OUTPUT_DIR}/rtl-validation"')
    assert guard < script.index("materialize.py")


def test_iflow_preflights_the_detached_runtime_not_dirty_source():
    script = (Path(__file__).resolve().parents[1] / "hardware/iflow/run.sh").read_text()

    actual_run = script.index('RUNTIME="${OUTPUT_DIR}/runtime/iflow"')
    worktree = script.index("worktree add --detach", actual_run)
    preflight = script.index('preflight.py', worktree)
    materialize = script.index('materialize.py', preflight)
    assert worktree < preflight < materialize


def test_iflow_records_per_stage_time_and_logs():
    script = (Path(__file__).resolve().parents[1] / "hardware/iflow/run.sh").read_text()

    assert "/usr/bin/time -v" in script
    assert 'time-${flow_stage}.log' in script
    assert 'iflow-${flow_stage}.log' in script
    assert 'stage-${flow_stage}-validation.json' in script
    assert 'resource-before-${flow_stage}.json' in script
    assert 'resource-after-${flow_stage}.json' in script


def test_iflow_exposes_the_explicit_low_memory_attempt_flag():
    script = (Path(__file__).resolve().parents[1] / "hardware/iflow/run.sh").read_text()

    assert "--allow-low-memory-attempt" in script
    assert "ALLOW_LOW_MEMORY_ATTEMPT" in script


def test_iflow_executes_the_preflight_resolved_image_id():
    script = (Path(__file__).resolve().parents[1] / "hardware/iflow/run.sh").read_text()

    assert 'RESOLVED_IFLOW_IMAGE="$(python3 -c' in script
    assert 'SCARF_IFLOW_IMAGE="${RESOLVED_IFLOW_IMAGE}"' in script
    assert '-w /opt/iFlow "${RESOLVED_IFLOW_IMAGE}"' in script
