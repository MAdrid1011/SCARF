import pytest


def _route_counts(**overrides):
    from saes.stage_event_simulator import SAESRouteCounters

    values = {
        "total_tiles": 3,
        "level0_tiles": 1,
        "level1_tiles": 1,
        "full_tiles": 1,
        "primary_probe_events": 3,
        "secondary_probe_events": 2,
        "full_execution_events": 1,
        "full_probe_replay_events": 0,
    }
    values.update(overrides)
    return SAESRouteCounters(**values)


def _route_events(**overrides):
    from saes.stage_event_simulator import StageEvent, StageRole

    values = {
        "primary": StageEvent(
            name="primary_probe",
            count=3,
            cycles_per_event=2,
            resource="probe",
            role=StageRole.PRIMARY_PROBE,
        ),
        "secondary": StageEvent(
            name="secondary_probe",
            count=2,
            cycles_per_event=3,
            resource="probe",
            depends_on=("primary_probe",),
            role=StageRole.SECONDARY_PROBE,
        ),
        "full": StageEvent(
            name="full_execute",
            count=1,
            cycles_per_event=4,
            resource="full",
            depends_on=("secondary_probe",),
            role=StageRole.FULL_EXECUTION,
        ),
    }
    values.update(overrides)
    return tuple(values.values())


def _resources():
    from saes.stage_event_simulator import ResourceConfig

    return (
        ResourceConfig(name="probe", lanes=1),
        ResourceConfig(name="full", lanes=1),
        ResourceConfig(name="merge", lanes=1),
    )


def test_stage_event_schedule_is_explicitly_non_rtl_and_reuses_full_probes():
    from saes.stage_event_simulator import (
        NON_RTL_TIMING_CLASS,
        StageEvent,
        simulate_stage_events,
    )

    events = _route_events() + (
        StageEvent(
            name="merge",
            count=2,
            cycles_per_event=5,
            resource="merge",
            depends_on=("secondary_probe",),
        ),
    )

    result = simulate_stage_events(events, _resources(), route_counts=_route_counts())

    assert result.timing_class == NON_RTL_TIMING_CLASS
    assert result.rtl_cycle_equivalent is False
    assert result.event_count == 8
    assert result.stages["primary_probe"].start_cycle == 0
    assert result.stages["primary_probe"].end_cycle == 6
    assert result.stages["secondary_probe"].start_cycle == 6
    assert result.stages["secondary_probe"].end_cycle == 12
    assert result.stages["full_execute"].start_cycle == 12
    assert result.stages["full_execute"].end_cycle == 16
    assert result.stages["merge"].end_cycle == 22
    assert result.total_cycles == 22


def test_shared_resource_queue_and_dependency_wait_are_reported():
    from saes.stage_event_simulator import (
        ResourceConfig,
        StageEvent,
        simulate_stage_events,
    )

    result = simulate_stage_events(
        (
            StageEvent(
                name="first",
                count=2,
                cycles_per_event=3,
                resource="compute",
            ),
            StageEvent(
                name="dependent",
                count=1,
                cycles_per_event=2,
                resource="compute",
                depends_on=("first",),
            ),
        ),
        (ResourceConfig(name="compute", lanes=1),),
    )

    assert result.stages["first"].end_cycle == 6
    assert result.stages["first"].queue_delay_cycles == 3
    assert result.stages["dependent"].dependency_ready_cycle == 6
    assert result.stages["dependent"].start_cycle == 6
    assert result.stages["dependent"].queue_delay_cycles == 0
    assert result.resources["compute"].busy_cycles == 8
    assert result.total_cycles == 8


def test_route_conservation_and_full_fallback_probe_replay_fail_closed():
    from saes.stage_event_simulator import StageEvent, StageRole, simulate_stage_events

    with pytest.raises(ValueError, match="route tile conservation"):
        simulate_stage_events(
            _route_events(), _resources(), route_counts=_route_counts(total_tiles=4)
        )

    with pytest.raises(
        ValueError,
        match="Full fallback must not repeat primary or secondary probe work",
    ):
        simulate_stage_events(
            _route_events(),
            _resources(),
            route_counts=_route_counts(full_probe_replay_events=1),
        )

    with pytest.raises(
        ValueError, match="Full execution must reuse secondary probe work"
    ):
        simulate_stage_events(
            _route_events(
                full=StageEvent(
                    name="full_execute",
                    count=1,
                    cycles_per_event=4,
                    resource="full",
                    depends_on=("primary_probe",),
                    role=StageRole.FULL_EXECUTION,
                )
            ),
            _resources(),
            route_counts=_route_counts(),
        )


def test_negative_event_cycles_are_rejected_before_scheduling():
    from saes.stage_event_simulator import (
        ResourceConfig,
        StageEvent,
        simulate_stage_events,
    )

    with pytest.raises(
        ValueError, match="cycles_per_event must be a nonnegative integer"
    ):
        simulate_stage_events(
            (
                StageEvent(
                    name="invalid",
                    count=1,
                    cycles_per_event=-1,
                    resource="compute",
                ),
            ),
            (ResourceConfig(name="compute", lanes=1),),
        )
