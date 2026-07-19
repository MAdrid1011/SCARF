"""Explicit, non-RTL scheduling for declared SAES stage events.

This module is intentionally separate from the analytic hardware ledger.  It
does not inspect model tensors, infer work from paper values, or describe an
RTL microarchitecture.  Callers declare every event count, per-event cycle
cost, resource, and dependency.  The resulting schedule is useful for
auditing event conservation and abstract resource contention only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


SIMULATOR_VERSION = "saes-stage-event-simulator-v1"
NON_RTL_TIMING_CLASS = "stage_event_schedule_not_rtl_cycle_equivalent"


class StageRole(str, Enum):
    """Semantic labels used only when validating an SAES route trace."""

    GENERIC = "generic"
    PRIMARY_PROBE = "primary_probe"
    SECONDARY_PROBE = "secondary_probe"
    FULL_EXECUTION = "full_execution"


@dataclass(frozen=True)
class ResourceConfig:
    """One abstract resource with an explicit number of independent lanes."""

    name: str
    lanes: int


@dataclass(frozen=True)
class StageEvent:
    """A homogeneous batch of work with explicit timing and dependencies.

    ``cycles_per_event`` is supplied by the caller.  It is not an RTL latency
    or a value inferred from model data.  A stage dependency gates the entire
    dependent batch, which makes this a conservative stage-level model rather
    than an event-accurate pipeline implementation.
    """

    name: str
    count: int
    cycles_per_event: int
    resource: str
    depends_on: tuple[str, ...] = ()
    role: StageRole | str = StageRole.GENERIC


@dataclass(frozen=True)
class SAESRouteCounters:
    """Explicit routing counters required to audit one L0/L1/Full trace.

    The controller evaluates a primary probe for every tile.  Every primary
    miss (L1 or Full) evaluates one secondary probe, and Full then reuses those
    probe results.  ``full_probe_replay_events`` is deliberately explicit so a
    caller cannot hide duplicate probe work in a Full fallback.
    """

    total_tiles: int
    level0_tiles: int
    level1_tiles: int
    full_tiles: int
    primary_probe_events: int
    secondary_probe_events: int
    full_execution_events: int
    full_probe_replay_events: int = 0


@dataclass(frozen=True)
class ScheduledStage:
    """The aggregate schedule and waiting attribution for one stage batch."""

    name: str
    role: StageRole
    resource: str
    event_count: int
    cycles_per_event: int
    dependency_ready_cycle: int
    start_cycle: int
    end_cycle: int
    work_cycles: int
    queue_delay_cycles: int


@dataclass(frozen=True)
class ResourceSchedule:
    """Summary of explicit work and queue delay on one abstract resource."""

    name: str
    lanes: int
    busy_cycles: int
    queue_delay_cycles: int
    last_completion_cycle: int


@dataclass(frozen=True)
class StageEventSimulation:
    """Result of a declared event schedule, explicitly not RTL evidence."""

    simulator_version: str
    timing_class: str
    rtl_cycle_equivalent: bool
    event_count: int
    total_cycles: int
    stages: Mapping[str, ScheduledStage]
    resources: Mapping[str, ResourceSchedule]


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _name(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _normalise_role(value: object, stage_name: str) -> StageRole:
    try:
        return StageRole(value)
    except (TypeError, ValueError) as error:
        allowed = ", ".join(role.value for role in StageRole)
        raise ValueError(
            f"stage {stage_name!r} has an invalid role; expected one of {allowed}"
        ) from error


def _validate_resources(
    resources: Sequence[ResourceConfig],
) -> dict[str, ResourceConfig]:
    resource_map: dict[str, ResourceConfig] = {}
    for resource in resources:
        if not isinstance(resource, ResourceConfig):
            raise ValueError("resources must contain ResourceConfig values")
        name = _name(resource.name, "resource name")
        _positive_int(resource.lanes, f"resource {name!r} lanes")
        if name in resource_map:
            raise ValueError(f"resource {name!r} is declared more than once")
        resource_map[name] = resource
    return resource_map


def _validate_events(
    events: Sequence[StageEvent], resources: Mapping[str, ResourceConfig]
) -> tuple[
    dict[str, StageEvent], dict[str, StageRole], dict[str, tuple[str, ...]], list[str]
]:
    event_map: dict[str, StageEvent] = {}
    roles: dict[str, StageRole] = {}
    dependencies: dict[str, tuple[str, ...]] = {}
    order: list[str] = []

    for event in events:
        if not isinstance(event, StageEvent):
            raise ValueError("events must contain StageEvent values")
        name = _name(event.name, "stage name")
        if name in event_map:
            raise ValueError(f"stage {name!r} is declared more than once")
        _nonnegative_int(event.count, f"stage {name!r} count")
        _nonnegative_int(event.cycles_per_event, "cycles_per_event")
        resource = _name(event.resource, f"stage {name!r} resource")
        if resource not in resources:
            raise ValueError(f"stage {name!r} references unknown resource {resource!r}")
        if isinstance(event.depends_on, str):
            raise ValueError(
                f"stage {name!r} depends_on must be an iterable of stage names"
            )
        try:
            event_dependencies = tuple(event.depends_on)
        except TypeError as error:
            raise ValueError(
                f"stage {name!r} depends_on must be an iterable of stage names"
            ) from error
        if len(set(event_dependencies)) != len(event_dependencies):
            raise ValueError(f"stage {name!r} declares a dependency more than once")
        for dependency in event_dependencies:
            _name(dependency, f"stage {name!r} dependency")
            if dependency == name:
                raise ValueError(f"stage {name!r} cannot depend on itself")
        event_map[name] = event
        roles[name] = _normalise_role(event.role, name)
        dependencies[name] = event_dependencies
        order.append(name)

    for name, event_dependencies in dependencies.items():
        for dependency in event_dependencies:
            if dependency not in event_map:
                raise ValueError(
                    f"stage {name!r} depends on unknown stage {dependency!r}"
                )
    return event_map, roles, dependencies, order


def _validate_route_counters(counters: SAESRouteCounters) -> None:
    if not isinstance(counters, SAESRouteCounters):
        raise ValueError("route_counts must be SAESRouteCounters")
    for field in (
        "total_tiles",
        "level0_tiles",
        "level1_tiles",
        "full_tiles",
        "primary_probe_events",
        "secondary_probe_events",
        "full_execution_events",
        "full_probe_replay_events",
    ):
        _nonnegative_int(getattr(counters, field), field)

    if counters.total_tiles != (
        counters.level0_tiles + counters.level1_tiles + counters.full_tiles
    ):
        raise ValueError("route tile conservation does not match total_tiles")
    if counters.primary_probe_events != counters.total_tiles:
        raise ValueError("primary_probe_events must equal total_tiles")
    if counters.secondary_probe_events != (counters.level1_tiles + counters.full_tiles):
        raise ValueError("secondary_probe_events must equal level1_tiles + full_tiles")
    if counters.full_execution_events != counters.full_tiles:
        raise ValueError("full_execution_events must equal full_tiles")
    if counters.full_probe_replay_events != 0:
        raise ValueError(
            "Full fallback must not repeat primary or secondary probe work"
        )


def _transitively_depends_on(
    stage_name: str,
    ancestor: str,
    dependencies: Mapping[str, tuple[str, ...]],
) -> bool:
    pending = list(dependencies[stage_name])
    visited: set[str] = set()
    while pending:
        dependency = pending.pop()
        if dependency == ancestor:
            return True
        if dependency not in visited:
            visited.add(dependency)
            pending.extend(dependencies[dependency])
    return False


def _validate_route_stages(
    event_map: Mapping[str, StageEvent],
    roles: Mapping[str, StageRole],
    dependencies: Mapping[str, tuple[str, ...]],
    counters: SAESRouteCounters | None,
) -> None:
    special_roles = {role for role in roles.values() if role is not StageRole.GENERIC}
    if counters is None:
        if special_roles:
            raise ValueError("route_counts are required for SAES route stage roles")
        return

    _validate_route_counters(counters)
    expected_counts = {
        StageRole.PRIMARY_PROBE: counters.primary_probe_events,
        StageRole.SECONDARY_PROBE: counters.secondary_probe_events,
        StageRole.FULL_EXECUTION: counters.full_execution_events,
    }
    role_stages: dict[StageRole, str] = {}
    for role, expected_count in expected_counts.items():
        matching = [name for name, event_role in roles.items() if event_role is role]
        if len(matching) != 1:
            raise ValueError(
                f"route trace requires exactly one {role.value} stage, got {len(matching)}"
            )
        stage_name = matching[0]
        if event_map[stage_name].count != expected_count:
            raise ValueError(
                f"{role.value} stage count does not match declared route counters"
            )
        role_stages[role] = stage_name

    primary = role_stages[StageRole.PRIMARY_PROBE]
    secondary = role_stages[StageRole.SECONDARY_PROBE]
    full = role_stages[StageRole.FULL_EXECUTION]
    if not _transitively_depends_on(secondary, primary, dependencies):
        raise ValueError("secondary probe stage must depend on primary probe work")
    if not _transitively_depends_on(full, primary, dependencies):
        raise ValueError("Full execution must reuse primary probe work")
    if not _transitively_depends_on(full, secondary, dependencies):
        raise ValueError("Full execution must reuse secondary probe work")


def simulate_stage_events(
    events: Sequence[StageEvent],
    resources: Sequence[ResourceConfig],
    *,
    route_counts: SAESRouteCounters | None = None,
) -> StageEventSimulation:
    """Schedule declared stage batches on explicit abstract resources.

    A dependency waits for the predecessor batch to finish.  Batches with no
    dependency are selected by their declared input order when they become
    ready at the same cycle.  Work within a batch takes the earliest available
    resource lane, so ``queue_delay_cycles`` reports aggregate lane waiting.

    No default resource width, cycle cost, model shape, or paper result is
    supplied here.  The returned ``total_cycles`` is an abstract schedule
    length and must not be presented as RTL-cycle-equivalent timing.
    """
    resource_map = _validate_resources(resources)
    event_map, roles, dependencies, order = _validate_events(events, resource_map)
    _validate_route_stages(event_map, roles, dependencies, route_counts)

    remaining = set(order)
    scheduled: dict[str, ScheduledStage] = {}
    lane_availability = {
        name: [0 for _ in range(resource.lanes)]
        for name, resource in resource_map.items()
    }
    resource_busy_cycles = {name: 0 for name in resource_map}
    resource_queue_delay = {name: 0 for name in resource_map}

    while remaining:
        ready: list[tuple[int, int, str]] = []
        for position, name in enumerate(order):
            if name not in remaining:
                continue
            if all(dependency in scheduled for dependency in dependencies[name]):
                dependency_ready = max(
                    (
                        scheduled[dependency].end_cycle
                        for dependency in dependencies[name]
                    ),
                    default=0,
                )
                ready.append((dependency_ready, position, name))
        if not ready:
            raise ValueError("stage dependencies contain a cycle")

        dependency_ready, _, name = min(ready)
        event = event_map[name]
        resource = event.resource
        work_cycles = event.count * event.cycles_per_event
        if event.count == 0 or event.cycles_per_event == 0:
            start_cycle = dependency_ready
            end_cycle = dependency_ready
            queue_delay_cycles = 0
        else:
            starts: list[int] = []
            ends: list[int] = []
            queue_delay_cycles = 0
            lanes = lane_availability[resource]
            for _ in range(event.count):
                lane_index = min(
                    range(len(lanes)), key=lambda index: (lanes[index], index)
                )
                start = max(dependency_ready, lanes[lane_index])
                end = start + event.cycles_per_event
                lanes[lane_index] = end
                starts.append(start)
                ends.append(end)
                queue_delay_cycles += start - dependency_ready
            start_cycle = min(starts)
            end_cycle = max(ends)
            resource_busy_cycles[resource] += work_cycles
            resource_queue_delay[resource] += queue_delay_cycles

        scheduled[name] = ScheduledStage(
            name=name,
            role=roles[name],
            resource=resource,
            event_count=event.count,
            cycles_per_event=event.cycles_per_event,
            dependency_ready_cycle=dependency_ready,
            start_cycle=start_cycle,
            end_cycle=end_cycle,
            work_cycles=work_cycles,
            queue_delay_cycles=queue_delay_cycles,
        )
        remaining.remove(name)

    resource_schedules = {
        name: ResourceSchedule(
            name=name,
            lanes=resource.lanes,
            busy_cycles=resource_busy_cycles[name],
            queue_delay_cycles=resource_queue_delay[name],
            last_completion_cycle=max(lane_availability[name], default=0),
        )
        for name, resource in resource_map.items()
    }
    return StageEventSimulation(
        simulator_version=SIMULATOR_VERSION,
        timing_class=NON_RTL_TIMING_CLASS,
        rtl_cycle_equivalent=False,
        event_count=sum(event.count for event in event_map.values()),
        total_cycles=max((stage.end_cycle for stage in scheduled.values()), default=0),
        stages=scheduled,
        resources=resource_schedules,
    )
