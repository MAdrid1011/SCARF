"""Incremental same-weight execution for a spatial two-convolution raw head.

This producer is deliberately narrower than whole-pipeline SAES.  It executes
only requested final raw-head positions, caches the first-convolution closure,
and records source-bound primary, secondary, and Full dispatches.  It does not
make S1/S2/refinement sparse, and a mixed Full tile is patch replay rather than
a native whole-view dispatch.  Consumers must therefore treat its output as
S3-head-only evidence until they implement an explicit packed adapter path.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

import torch
import torch.nn as nn

from saes.selected_output_replay import (
    _apply_conv_to_patches,
    _conv_macs_per_position,
    _gather_padded_patches,
    _linear_to_coordinates,
    _same_conv3_closure,
    unpack_two_conv_head,
)


INCREMENTAL_HEAD_EXECUTION_VERSION = "saes-incremental-head-execution-v1"
RAW_HEAD_EXECUTION_CONTRACT = "native-dense-head-closure-selected-packet-v2"
NATIVE_DENSE_HEAD_EXECUTION_EVIDENCE_VERSION = (
    "saes-native-dense-head-execution-evidence-v1"
)
_PHASE_ORDER = {
    "primary": 0,
    "secondary": 1,
    "full": 2,
    "full_extension": 3,
}


@dataclass(frozen=True)
class IncrementalSelectedOutputReplay:
    """Sparse raw-head map plus the source-bound execution ledger."""

    values: torch.Tensor
    computed_mask: torch.Tensor
    events: dict[str, Any]


@dataclass
class IncrementalSelectedOutputExecutionTrace:
    """One scoped raw-head invocation backed by an incremental producer."""

    primary_mask: torch.Tensor
    secondary_mask: torch.Tensor
    full_mask: torch.Tensor
    defer_full_extension: bool = False
    invocations: list[dict[str, Any]] = field(default_factory=list)
    replay: IncrementalSelectedOutputReplay | None = None
    _producer: IncrementalSelectedOutputProducer | None = field(
        default=None, init=False, repr=False
    )
    _initial_events: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )
    _full_extension_mask: torch.Tensor | None = field(
        default=None, init=False, repr=False
    )

    @property
    def initial_selection_mask(self) -> torch.Tensor:
        """The route-plan union before a post-guard Full extension."""
        return self.primary_mask | self.secondary_mask | self.full_mask

    @property
    def selection_mask(self) -> torch.Tensor:
        if self._full_extension_mask is None:
            return self.initial_selection_mask
        return self.initial_selection_mask.to(self._full_extension_mask.device) | self._full_extension_mask

    @property
    def initial_events(self) -> dict[str, Any]:
        """Return the completed three-phase ledger available at Adapter entry."""
        if self._initial_events is None:
            raise RuntimeError("incremental head execution has not reached the Adapter boundary")
        return dict(self._initial_events)

    @property
    def live_values(self) -> torch.Tensor:
        """Return the live raw-head map while a deferred route is open.

        This reference is intentionally not a clone.  A same-invocation Adapter
        boundary can append a guarded Full request and observe the new values
        without invoking the native head a second time.
        """
        if self._producer is None:
            raise RuntimeError("incremental head execution has not started")
        return self._producer.values

    def append_full_extension(self, selection_mask: torch.Tensor) -> dict[str, Any]:
        """Execute one disjoint, guard-requested Full extension in this session."""
        if not self.defer_full_extension:
            raise RuntimeError("this scoped head execution does not permit a Full extension")
        if self._producer is None:
            raise RuntimeError("Full extension requires an active raw-head invocation")
        if self.replay is not None:
            raise RuntimeError("Full extension must be appended before finalizing the producer")
        if self._full_extension_mask is not None:
            raise RuntimeError("incremental head execution permits only one Full extension")
        extension = self._producer.validated_mask(selection_mask)
        if not bool(extension.any()):
            raise ValueError("Full extension must request at least one new raw-head position")
        initial_mask = self.initial_selection_mask.to(extension.device)
        if bool((extension & initial_mask).any()):
            raise ValueError("Full extension must be disjoint from the initial route plan")
        event = self._producer.execute("full_extension", extension)
        self._full_extension_mask = extension.detach().clone()
        return dict(event)

    def finalize(self) -> IncrementalSelectedOutputReplay:
        """Seal the scoped producer after all allowed guarded work has run."""
        if self.replay is not None:
            return self.replay
        if self._producer is None:
            raise RuntimeError("incremental head execution has not started")
        replay = self._producer.finalize(self.selection_mask)
        events = dict(replay.events)
        events["head_forward_invocations"] = 1
        events["head_execution_mode"] = (
            "scoped_native_dense_head_selected_packet_no_s3_saving"
            if events["second_conv_native_kernel_aligned"]
            else "scoped_incremental_patch_replay_fp32_equivalence_only"
        )
        events["deferred_full_extension_enabled"] = self.defer_full_extension
        self.replay = replay
        self.invocations.append(events)
        return replay

    @property
    def events(self) -> dict[str, Any]:
        if len(self.invocations) != 1:
            raise RuntimeError(
                "incremental head execution must be finalized after its one raw-head call"
            )
        return dict(self.invocations[0])


def _sha256_tensor(value: torch.Tensor) -> str:
    if not torch.is_tensor(value):
        raise TypeError("tensor digest requires a tensor")
    detached = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(json.dumps(list(detached.shape), separators=(",", ":")).encode("ascii"))
    digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _sha256_module_state(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if not torch.is_tensor(value):
            raise TypeError("head state contains a non-tensor entry")
        digest.update(name.encode("utf-8"))
        digest.update(_sha256_tensor(value).encode("ascii"))
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _require_event_count(value: Any, *, label: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"native dense head evidence has invalid {label}")
    if positive and value == 0:
        raise ValueError(f"native dense head evidence requires positive {label}")
    return value


def _require_event_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"native dense head evidence has invalid {label}")
    return value


def native_dense_head_execution_evidence(events: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and summarize a fully native dense raw-head closure ledger.

    This is deliberately stricter than a string version check.  V16 can use
    this contract only when the source-weight first and second convolutions
    both executed once at native dense shape, while the packet still exposes
    only route-selected outputs.
    """
    if not isinstance(events, Mapping):
        raise TypeError("native dense head evidence requires an execution mapping")
    if events.get("raw_head_execution_contract") != RAW_HEAD_EXECUTION_CONTRACT:
        raise ValueError("native dense head execution contract changed")
    if events.get("head_forward_invocations") != 1:
        raise ValueError("native dense head evidence requires one head invocation")
    if events.get("head_execution_mode") != (
        "scoped_native_dense_head_selected_packet_no_s3_saving"
    ):
        raise ValueError("native dense head execution mode changed")
    if (
        events.get("source_bound") is not True
        or events.get("execution_scope") != "s3_raw_gaussian_head_only"
        or events.get("claim_scope") != "raw_gaussian_head_only"
        or events.get("whole_pipeline_s2_s3_sparse_execution_verified") is not False
        or events.get("upstream_s2_saving") != 0.0
        or events.get("full_execution_mode")
        != "native_dense_head_closure_selected_packet_no_s3_saving"
        or events.get("full_tile_native_identity_verified") is not False
        or not isinstance(events.get("execution_finalized"), bool)
    ):
        raise ValueError("native dense head evidence cannot claim Full tile identity")

    dense_positions = _require_event_count(
        events.get("dense_head_positions"), label="dense head positions", positive=True
    )
    dense_macs = _require_event_count(
        events.get("dense_head_macs"), label="dense head MACs", positive=True
    )
    if (
        events.get("first_conv_positions_executed") != dense_positions
        or events.get("native_dense_first_conv_positions_executed") != dense_positions
        or events.get("second_conv_positions_executed") != dense_positions
        or events.get("native_dense_second_conv_positions_executed") != dense_positions
        or events.get("first_conv_execution_mode") != "native_dense_closure"
        or events.get("second_conv_execution_mode") != "native_dense_closure"
        or events.get("first_conv_native_kernel_aligned") is not True
        or events.get("second_conv_native_kernel_aligned") is not True
        or events.get("actual_head_macs") != dense_macs
        or events.get("head_mac_delta") != 0
    ):
        raise ValueError("native dense head execution does not conserve dense work")

    phases = events.get("phases")
    if not isinstance(phases, list) or len(phases) not in (3, 4):
        raise ValueError("native dense head evidence has an invalid phase trace")
    expected_phase_names = ["primary", "secondary", "full"]
    if len(phases) == 4:
        expected_phase_names.append("full_extension")
    if [phase.get("phase") if isinstance(phase, Mapping) else None for phase in phases] != (
        expected_phase_names
    ):
        raise ValueError("native dense head evidence phase order changed")
    if events.get("phase_trace_sha256") != _canonical_sha256(phases):
        raise ValueError("native dense head evidence phase trace digest changed")

    flattened_tiles: list[dict[str, Any]] = []
    phase_evidence: list[dict[str, Any]] = []
    for phase_index, phase in enumerate(phases):
        assert isinstance(phase, Mapping)
        phase_name = expected_phase_names[phase_index]
        tile_trace = phase.get("per_tile")
        if not isinstance(tile_trace, list):
            raise ValueError(f"native dense head {phase_name} tile trace is invalid")
        if phase.get("tile_trace_sha256") != _canonical_sha256(tile_trace):
            raise ValueError(f"native dense head {phase_name} tile trace digest changed")
        first_executed = _require_event_count(
            phase.get("first_conv_positions_executed"),
            label=f"{phase_name} first-convolution positions",
        )
        native_first = _require_event_count(
            phase.get("native_dense_first_conv_positions_executed"),
            label=f"{phase_name} native first-convolution positions",
        )
        second_executed = _require_event_count(
            phase.get("second_conv_positions_executed"),
            label=f"{phase_name} second-convolution positions",
        )
        native_second = _require_event_count(
            phase.get("native_dense_second_conv_positions_executed"),
            label=f"{phase_name} native second-convolution positions",
        )
        expected_positions = dense_positions if phase_name == "primary" else 0
        expected_mode = (
            "native_dense_closure"
            if phase_name == "primary"
            else "native_dense_closure_reuse"
        )
        if (
            first_executed != expected_positions
            or native_first != expected_positions
            or second_executed != expected_positions
            or native_second != expected_positions
            or phase.get("first_conv_execution_mode") != expected_mode
            or phase.get("second_conv_execution_mode") != expected_mode
            or phase.get("first_conv_native_kernel_aligned") is not True
            or phase.get("second_conv_native_kernel_aligned") is not True
        ):
            raise ValueError(f"native dense head {phase_name} work changed")
        for tile in tile_trace:
            if not isinstance(tile, Mapping):
                raise ValueError(f"native dense head {phase_name} tile is invalid")
            for key in (
                "first_conv_positions_executed",
                "native_dense_first_conv_positions_executed",
                "second_conv_positions_executed",
                "native_dense_second_conv_positions_executed",
            ):
                _require_event_count(tile.get(key), label=f"{phase_name} tile {key}")
            flattened_tiles.append(dict(tile))
        if (
            sum(int(tile["first_conv_positions_executed"]) for tile in tile_trace)
            != first_executed
            or sum(
                int(tile["native_dense_first_conv_positions_executed"])
                for tile in tile_trace
            )
            != native_first
            or sum(int(tile["second_conv_positions_executed"]) for tile in tile_trace)
            != second_executed
            or sum(
                int(tile["native_dense_second_conv_positions_executed"])
                for tile in tile_trace
            )
            != native_second
        ):
            raise ValueError(f"native dense head {phase_name} tile work does not conserve")
        phase_evidence.append(
            {
                "phase": phase_name,
                "mask_sha256": _require_event_sha256(
                    phase.get("mask_sha256"), label=f"{phase_name} mask"
                ),
                "tile_trace_sha256": _require_event_sha256(
                    phase.get("tile_trace_sha256"), label=f"{phase_name} tile trace"
                ),
                "first_conv_positions_executed": first_executed,
                "native_dense_first_conv_positions_executed": native_first,
                "second_conv_positions_executed": second_executed,
                "native_dense_second_conv_positions_executed": native_second,
            }
        )
    if (
        events.get("per_tile") != flattened_tiles
        or events.get("tile_trace_records") != len(flattened_tiles)
        or events.get("tile_trace_sha256") != _canonical_sha256(flattened_tiles)
    ):
        raise ValueError("native dense head aggregate tile trace changed")
    return {
        "schema_version": NATIVE_DENSE_HEAD_EXECUTION_EVIDENCE_VERSION,
        "raw_head_execution_contract": RAW_HEAD_EXECUTION_CONTRACT,
        "head_weight_sha256": _require_event_sha256(
            events.get("head_weight_sha256"), label="head weights"
        ),
        "head_input_sha256": _require_event_sha256(
            events.get("head_input_sha256"), label="head input"
        ),
        "phase_trace_sha256": _require_event_sha256(
            events.get("phase_trace_sha256"), label="phase trace"
        ),
        "tile_trace_sha256": _require_event_sha256(
            events.get("tile_trace_sha256"), label="tile trace"
        ),
        "execution_finalized": events["execution_finalized"],
        "dense_head_positions": dense_positions,
        "dense_head_macs": dense_macs,
        "actual_head_macs": dense_macs,
        "head_mac_delta": 0,
        "phases": phase_evidence,
    }


def validate_native_dense_head_execution_evidence(
    evidence: Mapping[str, Any], *, expected_phase_count: int | None = None
) -> dict[str, Any]:
    """Validate the persisted, compact form of native dense execution evidence."""
    if not isinstance(evidence, Mapping):
        raise TypeError("persisted native dense head evidence requires a mapping")
    required = {
        "schema_version",
        "raw_head_execution_contract",
        "head_weight_sha256",
        "head_input_sha256",
        "phase_trace_sha256",
        "tile_trace_sha256",
        "execution_finalized",
        "dense_head_positions",
        "dense_head_macs",
        "actual_head_macs",
        "head_mac_delta",
        "phases",
    }
    if set(evidence) != required:
        raise ValueError("persisted native dense head evidence has unexpected fields")
    if (
        evidence.get("schema_version") != NATIVE_DENSE_HEAD_EXECUTION_EVIDENCE_VERSION
        or evidence.get("raw_head_execution_contract") != RAW_HEAD_EXECUTION_CONTRACT
    ):
        raise ValueError("persisted native dense head evidence contract changed")
    if not isinstance(evidence.get("execution_finalized"), bool):
        raise ValueError("persisted native dense head finalization state is invalid")
    dense_positions = _require_event_count(
        evidence.get("dense_head_positions"), label="persisted dense head positions", positive=True
    )
    dense_macs = _require_event_count(
        evidence.get("dense_head_macs"), label="persisted dense head MACs", positive=True
    )
    if (
        evidence.get("actual_head_macs") != dense_macs
        or evidence.get("head_mac_delta") != 0
    ):
        raise ValueError("persisted native dense head MAC accounting changed")
    phases = evidence.get("phases")
    if not isinstance(phases, list) or len(phases) not in (3, 4):
        raise ValueError("persisted native dense head phase evidence is invalid")
    if expected_phase_count is not None and len(phases) != expected_phase_count:
        raise ValueError("persisted native dense head phase count changed")
    expected_phase_names = ["primary", "secondary", "full"]
    if len(phases) == 4:
        expected_phase_names.append("full_extension")
    normalized_phases: list[dict[str, Any]] = []
    for position, (phase, phase_name) in enumerate(zip(phases, expected_phase_names)):
        if not isinstance(phase, Mapping) or set(phase) != {
            "phase",
            "mask_sha256",
            "tile_trace_sha256",
            "first_conv_positions_executed",
            "native_dense_first_conv_positions_executed",
            "second_conv_positions_executed",
            "native_dense_second_conv_positions_executed",
        }:
            raise ValueError("persisted native dense head phase has unexpected fields")
        expected_positions = dense_positions if position == 0 else 0
        for key in (
            "first_conv_positions_executed",
            "native_dense_first_conv_positions_executed",
            "second_conv_positions_executed",
            "native_dense_second_conv_positions_executed",
        ):
            if _require_event_count(phase.get(key), label=f"persisted {phase_name} {key}") != expected_positions:
                raise ValueError("persisted native dense head phase work changed")
        if phase.get("phase") != phase_name:
            raise ValueError("persisted native dense head phase order changed")
        normalized_phases.append(
            {
                "phase": phase_name,
                "mask_sha256": _require_event_sha256(
                    phase.get("mask_sha256"), label=f"persisted {phase_name} mask"
                ),
                "tile_trace_sha256": _require_event_sha256(
                    phase.get("tile_trace_sha256"),
                    label=f"persisted {phase_name} tile trace",
                ),
                "first_conv_positions_executed": expected_positions,
                "native_dense_first_conv_positions_executed": expected_positions,
                "second_conv_positions_executed": expected_positions,
                "native_dense_second_conv_positions_executed": expected_positions,
            }
        )
    return {
        "schema_version": NATIVE_DENSE_HEAD_EXECUTION_EVIDENCE_VERSION,
        "raw_head_execution_contract": RAW_HEAD_EXECUTION_CONTRACT,
        "head_weight_sha256": _require_event_sha256(
            evidence.get("head_weight_sha256"), label="persisted head weights"
        ),
        "head_input_sha256": _require_event_sha256(
            evidence.get("head_input_sha256"), label="persisted head input"
        ),
        "phase_trace_sha256": _require_event_sha256(
            evidence.get("phase_trace_sha256"), label="persisted phase trace"
        ),
        "tile_trace_sha256": _require_event_sha256(
            evidence.get("tile_trace_sha256"), label="persisted tile trace"
        ),
        "execution_finalized": evidence["execution_finalized"],
        "dense_head_positions": dense_positions,
        "dense_head_macs": dense_macs,
        "actual_head_macs": dense_macs,
        "head_mac_delta": 0,
        "phases": normalized_phases,
    }


class IncrementalSelectedOutputProducer:
    """Execute a head in primary -> secondary -> Full phases without replaying work.

    A request mask represents positions demanded by that phase.  Positions that
    were already produced by an earlier phase are recorded as reuse rather than
    dispatched again.  The first 3x3 convolution's hidden closure is cached at
    individual spatial positions, so a Full fallback computes only its missing
    closure positions.  The result intentionally carries a false whole-pipeline
    verification flag: this class cannot make the upstream S2/refinement graph
    sparse or provide native mixed-tile Full identity.
    """

    def __init__(
        self,
        head: nn.Module,
        head_input: torch.Tensor,
        *,
        tile_size: int | None = None,
    ) -> None:
        if not isinstance(head, nn.Module):
            raise TypeError("incremental execution requires an nn.Module head")
        if not torch.is_tensor(head_input) or head_input.ndim != 4:
            raise ValueError("incremental execution requires head_input [N,C,H,W]")
        first, activation, second = unpack_two_conv_head(head)
        if head_input.shape[1] != first.in_channels:
            raise ValueError("head_input channel count does not match head[0]")
        batch, _channels, height, width = head_input.shape
        if tile_size is not None and (
            isinstance(tile_size, bool)
            or not isinstance(tile_size, int)
            or tile_size <= 0
            or height % tile_size
            or width % tile_size
        ):
            raise ValueError("tile_size must divide the raw-head spatial dimensions")
        self._head = head
        self._first = first
        self._activation = activation
        self._second = second
        self._head_input = head_input
        self._batch = batch
        self._height = height
        self._width = width
        self._tile_size = tile_size or max(height, width)
        self._hidden_values = torch.zeros(
            (batch, first.out_channels, height, width),
            dtype=head_input.dtype,
            device=head_input.device,
        )
        self._hidden_mask = torch.zeros(
            (batch, height, width), dtype=torch.bool, device=head_input.device
        )
        self._values = torch.zeros(
            (batch, second.out_channels, height, width),
            dtype=head_input.dtype,
            device=head_input.device,
        )
        self._computed_mask = torch.zeros(
            (batch, height, width), dtype=torch.bool, device=head_input.device
        )
        self._phase_events: list[dict[str, Any]] = []
        self._seen_phases: set[str] = set()
        self._last_phase = -1
        self._sealed = False
        self._native_dense_first_closure = False
        self._native_dense_second_values: torch.Tensor | None = None
        self._head_weight_sha256 = _sha256_module_state(head)
        self._head_input_sha256 = _sha256_tensor(head_input)

    def _validated_mask(self, selection_mask: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(selection_mask) or selection_mask.dtype != torch.bool:
            raise ValueError("selection_mask must be a bool tensor")
        if selection_mask.ndim == 2:
            selection_mask = selection_mask.unsqueeze(0).expand(self._batch, -1, -1)
        if selection_mask.shape != (self._batch, self._height, self._width):
            raise ValueError("selection_mask must have shape [N,H,W]")
        return selection_mask.to(device=self._head_input.device)

    def validated_mask(self, selection_mask: torch.Tensor) -> torch.Tensor:
        """Validate a public phase mask against this producer's head layout."""
        return self._validated_mask(selection_mask)

    @property
    def values(self) -> torch.Tensor:
        """Live raw-head values for a same-invocation deferred consumer."""
        return self._values

    def _validate_phase(self, phase: str) -> None:
        if self._sealed:
            raise RuntimeError("incremental producer has been finalized")
        if phase not in _PHASE_ORDER:
            raise ValueError("phase must be primary, secondary, full, or full_extension")
        phase_index = _PHASE_ORDER[phase]
        if phase_index < self._last_phase:
            raise ValueError(
                "execution phases must be ordered primary, secondary, full, full_extension"
            )
        if phase in self._seen_phases:
            raise ValueError(f"incremental execution phase {phase} may run only once")
        if phase == "secondary" and "primary" not in self._seen_phases:
            raise ValueError("secondary phase requires a primary phase")
        if phase == "full" and "secondary" not in self._seen_phases:
            raise ValueError("full phase requires a secondary phase")
        if phase == "full_extension" and "full" not in self._seen_phases:
            raise ValueError("Full extension requires the initial full phase")

    def _tile_origins(self) -> list[tuple[int, int]]:
        return [
            (tile_y, tile_x)
            for tile_y in range(0, self._height, self._tile_size)
            for tile_x in range(0, self._width, self._tile_size)
        ]

    def _primary_closure_is_dense(self, selection_mask: torch.Tensor) -> bool:
        """Return whether the primary final-output closure needs every hidden site."""
        for batch_item in range(self._batch):
            coordinates = selection_mask[batch_item].nonzero(as_tuple=False)
            if coordinates.numel() == 0:
                return False
            closure = _same_conv3_closure(
                coordinates, height=self._height, width=self._width
            )
            if int(closure.numel()) != self._height * self._width:
                return False
        return True

    def _maybe_execute_native_dense_head_closure(
        self, phase: str, selection_mask: torch.Tensor
    ) -> int:
        """Use the source head when the primary closure already needs it all.

        The four primary probes in every T=4 tile often make the first 3x3
        closure dense. The remaining L1 omissions are then too sparse to
        justify a separate patch kernel whose FP32 reduction differs from the
        source head. This branch charges both convolutions as dense work and
        exposes only route-selected values to the packet.
        """
        if (
            phase != "primary"
            or self._native_dense_first_closure
            or bool(self._hidden_mask.any())
            or not self._primary_closure_is_dense(selection_mask)
        ):
            return 0
        native_hidden = self._activation(self._first(self._head_input))
        if native_hidden.shape != self._hidden_values.shape:
            raise RuntimeError("native dense first convolution returned an invalid shape")
        if native_hidden.dtype != self._hidden_values.dtype:
            raise RuntimeError("native dense first convolution changed the source dtype")
        native_values = self._second(native_hidden)
        if native_values.shape != self._values.shape:
            raise RuntimeError("native dense second convolution returned an invalid shape")
        if native_values.dtype != self._values.dtype:
            raise RuntimeError("native dense second convolution changed the source dtype")
        self._hidden_values.copy_(native_hidden)
        self._hidden_mask.fill_(True)
        self._native_dense_second_values = native_values
        self._native_dense_first_closure = True
        return self._batch * self._height * self._width

    def _execute_tile(
        self,
        *,
        batch_item: int,
        tile_y: int,
        tile_x: int,
        requested_local: torch.Tensor,
    ) -> dict[str, int]:
        local_coordinates = requested_local.nonzero(as_tuple=False)
        requested = int(local_coordinates.shape[0])
        if requested == 0:
            return {
                "head_final_positions_requested": 0,
                "head_final_positions_reused": 0,
                "head_final_positions_executed": 0,
                "first_conv_positions_reused": 0,
                "first_conv_positions_executed": 0,
                "native_dense_first_conv_positions_executed": 0,
                "second_conv_positions_executed": 0,
                "native_dense_second_conv_positions_executed": 0,
            }
        coordinates = local_coordinates.clone()
        coordinates[:, 0] += tile_y
        coordinates[:, 1] += tile_x
        previously_computed = self._computed_mask[
            batch_item, coordinates[:, 0], coordinates[:, 1]
        ]
        new_coordinates = coordinates[~previously_computed]
        reused_final = int(previously_computed.sum().item())
        executed_final = int(new_coordinates.shape[0])
        if executed_final == 0:
            return {
                "head_final_positions_requested": requested,
                "head_final_positions_reused": reused_final,
                "head_final_positions_executed": 0,
                "first_conv_positions_reused": 0,
                "first_conv_positions_executed": 0,
                "native_dense_first_conv_positions_executed": 0,
                "second_conv_positions_executed": 0,
                "native_dense_second_conv_positions_executed": 0,
            }

        hidden_linear = _same_conv3_closure(
            new_coordinates, height=self._height, width=self._width
        )
        hidden_coordinates = _linear_to_coordinates(hidden_linear, width=self._width)
        hidden_known = self._hidden_mask[
            batch_item, hidden_coordinates[:, 0], hidden_coordinates[:, 1]
        ]
        new_hidden_coordinates = hidden_coordinates[~hidden_known]
        reused_hidden = int(hidden_known.sum().item())
        executed_hidden = int(new_hidden_coordinates.shape[0])
        if executed_hidden:
            first_patches = _gather_padded_patches(
                self._head_input[batch_item : batch_item + 1],
                new_hidden_coordinates,
                padding_mode=self._first.padding_mode,
            )
            hidden_values = self._activation(
                _apply_conv_to_patches(self._first, first_patches)
            )[0]
            self._hidden_values[
                batch_item,
                :,
                new_hidden_coordinates[:, 0],
                new_hidden_coordinates[:, 1],
            ] = hidden_values.transpose(0, 1)
            self._hidden_mask[
                batch_item, new_hidden_coordinates[:, 0], new_hidden_coordinates[:, 1]
            ] = True

        required_hidden = _same_conv3_closure(
            new_coordinates, height=self._height, width=self._width
        )
        required_coordinates = _linear_to_coordinates(required_hidden, width=self._width)
        if not bool(
            self._hidden_mask[
                batch_item, required_coordinates[:, 0], required_coordinates[:, 1]
            ].all()
        ):
            raise RuntimeError("incremental first-convolution closure is incomplete")
        if self._native_dense_second_values is None:
            second_patches = _gather_padded_patches(
                self._hidden_values[batch_item : batch_item + 1],
                new_coordinates,
                padding_mode=self._second.padding_mode,
            )
            final_values = _apply_conv_to_patches(self._second, second_patches)[0]
        else:
            final_values = self._native_dense_second_values[
                batch_item, :, new_coordinates[:, 0], new_coordinates[:, 1]
            ].transpose(0, 1)
        self._values[
            batch_item, :, new_coordinates[:, 0], new_coordinates[:, 1]
        ] = final_values.transpose(0, 1)
        self._computed_mask[
            batch_item, new_coordinates[:, 0], new_coordinates[:, 1]
        ] = True
        return {
            "head_final_positions_requested": requested,
            "head_final_positions_reused": reused_final,
            "head_final_positions_executed": executed_final,
            "first_conv_positions_reused": reused_hidden,
            "first_conv_positions_executed": executed_hidden,
            "native_dense_first_conv_positions_executed": 0,
            "second_conv_positions_executed": (
                0 if self._native_dense_second_values is not None else executed_final
            ),
            "native_dense_second_conv_positions_executed": 0,
        }

    def _allocate_native_dense_phase_work(
        self, per_tile: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Attach the physical dense closure to its spatial tile ledger.

        The selected-output rows still describe only exposed descriptors.  The
        native first and second convolutions, however, cover every spatial
        tile and must remain reconstructible from the same ledger rather than
        appearing only in a phase-level counter.
        """
        rows = {
            (entry["batch_item"], entry["tile_y"], entry["tile_x"]): dict(entry)
            for entry in per_tile
        }
        if len(rows) != len(per_tile):
            raise RuntimeError("native dense head phase has duplicate tile records")
        allocated: list[dict[str, Any]] = []
        for batch_item in range(self._batch):
            for tile_y, tile_x in self._tile_origins():
                tile_height = min(self._tile_size, self._height - tile_y)
                tile_width = min(self._tile_size, self._width - tile_x)
                key = (batch_item, tile_y // self._tile_size, tile_x // self._tile_size)
                entry = rows.pop(
                    key,
                    {
                        "phase": "primary",
                        "batch_item": batch_item,
                        "tile_y": key[1],
                        "tile_x": key[2],
                        "head_final_positions_requested": 0,
                        "head_final_positions_reused": 0,
                        "head_final_positions_executed": 0,
                        "first_conv_positions_reused": 0,
                        "first_conv_positions_executed": 0,
                        "native_dense_first_conv_positions_executed": 0,
                        "second_conv_positions_executed": 0,
                        "native_dense_second_conv_positions_executed": 0,
                    },
                )
                if (
                    entry["first_conv_positions_executed"] != 0
                    or entry["native_dense_first_conv_positions_executed"] != 0
                    or entry["second_conv_positions_executed"] != 0
                    or entry["native_dense_second_conv_positions_executed"] != 0
                ):
                    raise RuntimeError("native dense head closure mixed with patch work")
                positions = tile_height * tile_width
                entry["first_conv_positions_executed"] = positions
                entry["native_dense_first_conv_positions_executed"] = positions
                entry["second_conv_positions_executed"] = positions
                entry["native_dense_second_conv_positions_executed"] = positions
                allocated.append(entry)
        if rows:
            raise RuntimeError("native dense head phase has an out-of-range tile record")
        return allocated

    def execute(self, phase: str, selection_mask: torch.Tensor) -> dict[str, Any]:
        """Execute one route phase and return its source-bound per-tile ledger."""
        self._validate_phase(phase)
        mask = self._validated_mask(selection_mask)
        if phase == "full_extension" and bool((mask & self._computed_mask).any()):
            raise ValueError("Full extension may not replay an already produced position")
        native_dense_head_positions = self._maybe_execute_native_dense_head_closure(
            phase, mask
        )
        per_tile: list[dict[str, Any]] = []
        totals = {
            "head_final_positions_requested": 0,
            "head_final_positions_reused": 0,
            "head_final_positions_executed": 0,
            "first_conv_positions_reused": 0,
            "first_conv_positions_executed": 0,
            "native_dense_first_conv_positions_executed": 0,
            "second_conv_positions_executed": 0,
            "native_dense_second_conv_positions_executed": 0,
        }
        for batch_item in range(self._batch):
            for tile_y, tile_x in self._tile_origins():
                tile_height = min(self._tile_size, self._height - tile_y)
                tile_width = min(self._tile_size, self._width - tile_x)
                values = self._execute_tile(
                    batch_item=batch_item,
                    tile_y=tile_y,
                    tile_x=tile_x,
                    requested_local=mask[
                        batch_item,
                        tile_y : tile_y + tile_height,
                        tile_x : tile_x + tile_width,
                    ],
                )
                if values["head_final_positions_requested"]:
                    per_tile.append(
                        {
                            "phase": phase,
                            "batch_item": batch_item,
                            "tile_y": tile_y // self._tile_size,
                            "tile_x": tile_x // self._tile_size,
                            **values,
                        }
                    )
                    for key in totals:
                        totals[key] += values[key]
        if native_dense_head_positions:
            per_tile = self._allocate_native_dense_phase_work(per_tile)
        for key, expected in totals.items():
            observed = sum(int(entry[key]) for entry in per_tile)
            if key.startswith("native_dense_") and native_dense_head_positions:
                expected = native_dense_head_positions
            elif key in {
                "first_conv_positions_executed",
                "second_conv_positions_executed",
            } and native_dense_head_positions:
                expected = native_dense_head_positions
            if observed != expected:
                raise RuntimeError(f"incremental head {phase} tile ledger does not conserve {key}")
        event = {
            "schema_version": INCREMENTAL_HEAD_EXECUTION_VERSION,
            "phase": phase,
            "mask_sha256": _sha256_tensor(mask.to(dtype=torch.uint8)),
            "head_weight_sha256": self._head_weight_sha256,
            "head_input_sha256": self._head_input_sha256,
            "head_final_positions_requested": totals["head_final_positions_requested"],
            "head_final_positions_reused": totals["head_final_positions_reused"],
            "head_final_positions_executed": totals["head_final_positions_executed"],
            "first_conv_positions_reused": totals["first_conv_positions_reused"],
            "first_conv_positions_executed": sum(
                int(entry["first_conv_positions_executed"]) for entry in per_tile
            ),
            "native_dense_first_conv_positions_executed": sum(
                int(entry["native_dense_first_conv_positions_executed"])
                for entry in per_tile
            ),
            "first_conv_execution_mode": (
                "native_dense_closure"
                if native_dense_head_positions
                else "native_dense_closure_reuse"
                if self._native_dense_first_closure
                else "incremental_patch_closure"
            ),
            "first_conv_native_kernel_aligned": self._native_dense_first_closure,
            "native_dense_second_conv_positions_executed": sum(
                int(entry["native_dense_second_conv_positions_executed"])
                for entry in per_tile
            ),
            "second_conv_execution_mode": (
                "native_dense_closure"
                if native_dense_head_positions
                else "native_dense_closure_reuse"
                if self._native_dense_second_values is not None
                else "incremental_patch_selected_outputs"
            ),
            "second_conv_native_kernel_aligned": self._native_dense_second_values is not None,
            "second_conv_positions_executed": sum(
                int(entry["second_conv_positions_executed"]) for entry in per_tile
            ),
            "full_tile_native_identity_verified": False,
            "full_execution_mode": (
                "native_dense_head_closure_selected_packet_no_s3_saving"
                if phase == "full" and self._native_dense_second_values is not None
                else "native_dense_head_closure_reuse_no_s3_saving"
                if phase == "full_extension" and self._native_dense_second_values is not None
                else "incremental_patch_replay_fp32_equivalence_only"
                if phase == "full"
                else "guard_appended_incremental_patch_replay_fp32_equivalence_only"
                if phase == "full_extension"
                else "not_applicable"
            ),
            "per_tile": per_tile,
            "tile_trace_sha256": _canonical_sha256(per_tile),
        }
        self._phase_events.append(event)
        self._seen_phases.add(phase)
        self._last_phase = _PHASE_ORDER[phase]
        return dict(event)

    def _build_replay(
        self, required_mask: torch.Tensor | None, *, execution_finalized: bool
    ) -> IncrementalSelectedOutputReplay:
        required = (
            self._computed_mask
            if required_mask is None
            else self._validated_mask(required_mask)
        )
        missing = required & ~self._computed_mask
        if bool(missing.any()):
            raise ValueError("required head positions have not been executed")
        first_positions = int(self._hidden_mask.sum().item())
        final_positions = int(self._computed_mask.sum().item())
        first_macs = _conv_macs_per_position(self._first)
        second_macs = _conv_macs_per_position(self._second)
        dense_positions = self._batch * self._height * self._width
        second_positions = (
            dense_positions
            if self._native_dense_second_values is not None
            else final_positions
        )
        phase_events = [dict(event) for event in self._phase_events]
        per_tile = [
            dict(entry)
            for event in phase_events
            for entry in event["per_tile"]
        ]
        extension_events = [
            event for event in self._phase_events if event["phase"] == "full_extension"
        ]
        events = {
            "schema_version": INCREMENTAL_HEAD_EXECUTION_VERSION,
            "available": True,
            "producer_source": "same_weight_incremental_two_conv_head",
            "source_bound": True,
            "execution_scope": "s3_raw_gaussian_head_only",
            "claim_scope": "raw_gaussian_head_only",
            "whole_pipeline_s2_s3_sparse_execution_verified": False,
            "upstream_s2_saving": 0.0,
            "head_weight_sha256": self._head_weight_sha256,
            "head_input_sha256": self._head_input_sha256,
            "computed_mask_sha256": _sha256_tensor(
                self._computed_mask.to(dtype=torch.uint8)
            ),
            "execution_finalized": execution_finalized,
            "tile_size": self._tile_size,
            "batch_size": self._batch,
            "dense_head_positions": dense_positions,
            "first_conv_positions_executed": first_positions,
            "native_dense_first_conv_positions_executed": (
                dense_positions if self._native_dense_first_closure else 0
            ),
            "raw_head_execution_contract": RAW_HEAD_EXECUTION_CONTRACT,
            "first_conv_execution_mode": (
                "native_dense_closure"
                if self._native_dense_first_closure
                else "incremental_patch_closure"
            ),
            "first_conv_native_kernel_aligned": self._native_dense_first_closure,
            "head_final_positions_executed": final_positions,
            "second_conv_positions_executed": second_positions,
            "native_dense_second_conv_positions_executed": (
                dense_positions if self._native_dense_second_values is not None else 0
            ),
            "second_conv_execution_mode": (
                "native_dense_closure"
                if self._native_dense_second_values is not None
                else "incremental_patch_selected_outputs"
            ),
            "second_conv_native_kernel_aligned": self._native_dense_second_values is not None,
            "dense_head_macs": dense_positions * (first_macs + second_macs),
            "actual_head_macs": first_positions * first_macs + second_positions * second_macs,
            "head_mac_delta": dense_positions * (first_macs + second_macs)
            - (first_positions * first_macs + second_positions * second_macs),
            "full_tile_native_identity_verified": False,
            "full_execution_mode": (
                "native_dense_head_closure_selected_packet_no_s3_saving"
                if self._native_dense_second_values is not None
                else "incremental_patch_replay_fp32_equivalence_only"
            ),
            "full_extension_dispatched": bool(extension_events),
            "full_extension_positions_executed": sum(
                int(event["head_final_positions_executed"])
                for event in extension_events
            ),
            "full_extension_mask_sha256": (
                extension_events[0]["mask_sha256"] if extension_events else None
            ),
            "phases": phase_events,
            "phase_trace_sha256": _canonical_sha256(phase_events),
            "per_tile": per_tile,
            "tile_trace_records": len(per_tile),
            "tile_trace_sha256": _canonical_sha256(per_tile),
        }
        return IncrementalSelectedOutputReplay(
            values=self._values.clone(),
            computed_mask=self._computed_mask.clone(),
            events=events,
        )

    def snapshot(
        self, required_mask: torch.Tensor | None = None
    ) -> IncrementalSelectedOutputReplay:
        """Capture a complete-but-unsealed route ledger for an Adapter guard."""
        if self._sealed:
            raise RuntimeError("incremental producer has already been finalized")
        return self._build_replay(required_mask, execution_finalized=False)

    def finalize(
        self, required_mask: torch.Tensor | None = None
    ) -> IncrementalSelectedOutputReplay:
        """Return the produced map after proving every required position exists."""
        if self._sealed:
            raise RuntimeError("incremental producer has already been finalized")
        replay = self._build_replay(required_mask, execution_finalized=True)
        self._sealed = True
        return replay


def _validate_phase_masks(
    primary_mask: torch.Tensor,
    secondary_mask: torch.Tensor,
    full_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masks = (primary_mask, secondary_mask, full_mask)
    if any(not torch.is_tensor(mask) for mask in masks):
        raise ValueError("incremental head execution requires tensor phase masks")
    if any(mask.ndim != 3 or mask.dtype != torch.bool for mask in masks):
        raise ValueError(
            "incremental head execution requires [N,H,W] bool phase masks"
        )
    if primary_mask.shape != secondary_mask.shape or primary_mask.shape != full_mask.shape:
        raise ValueError("incremental head execution phase masks must share one shape")
    return tuple(mask.detach().clone() for mask in masks)  # type: ignore[return-value]


@contextmanager
def incremental_selected_output_head_execution(
    head: nn.Module,
    primary_mask: torch.Tensor,
    secondary_mask: torch.Tensor,
    full_mask: torch.Tensor,
    *,
    tile_size: int | None = None,
    defer_full_extension: bool = False,
) -> Iterator[IncrementalSelectedOutputExecutionTrace]:
    """Replace one raw-head call with ordered primary, secondary, and Full work.

    The supplied masks must use the native flattened head-batch order.  For
    TranSplat's B=1 context pass this is the canonical view order; B>1 callers
    must convert ``[B,V,H,W]`` to the model's ``(v b)`` batch layout before
    entering the scope.  The returned map is zero-filled outside the union, so
    only a source-bound packed consumer may read it.  With
    ``defer_full_extension=True``, the caller must finalize the trace after an
    Adapter-bound guard has either appended one disjoint Full extension or
    decided that no extension is necessary.
    """
    if not isinstance(head, nn.Module):
        raise TypeError("incremental head execution requires an nn.Module head")
    primary, secondary, full = _validate_phase_masks(
        primary_mask, secondary_mask, full_mask
    )
    original_forward = head.forward
    had_instance_forward = "forward" in head.__dict__
    trace = IncrementalSelectedOutputExecutionTrace(
        primary, secondary, full, defer_full_extension=defer_full_extension
    )

    def incremental_forward(head_input: torch.Tensor) -> torch.Tensor:
        if trace._producer is not None:
            raise RuntimeError(
                "incremental head execution requires exactly one raw-head call"
            )
        expected_shape = (head_input.shape[0], *head_input.shape[-2:])
        if primary.shape != expected_shape:
            raise ValueError(
                "incremental phase masks do not match the native raw-head input layout"
            )
        producer = IncrementalSelectedOutputProducer(
            head, head_input, tile_size=tile_size
        )
        producer.execute("primary", primary)
        producer.execute("secondary", secondary)
        producer.execute("full", full)
        trace._producer = producer
        initial = producer.snapshot(trace.initial_selection_mask)
        initial_events = dict(initial.events)
        initial_events["head_forward_invocations"] = 1
        initial_events["head_execution_mode"] = (
            "scoped_native_dense_head_selected_packet_no_s3_saving"
            if initial_events["second_conv_native_kernel_aligned"]
            else "scoped_incremental_patch_replay_initial_route_only"
            if defer_full_extension
            else "scoped_incremental_patch_replay_fp32_equivalence_only"
        )
        initial_events["deferred_full_extension_enabled"] = defer_full_extension
        trace._initial_events = initial_events
        if defer_full_extension:
            return producer.values
        return trace.finalize().values

    head.forward = incremental_forward  # type: ignore[method-assign]
    try:
        yield trace
    finally:
        if trace._producer is not None and trace.replay is None:
            trace.finalize()
        if had_instance_forward:
            head.forward = original_forward  # type: ignore[method-assign]
        else:
            delattr(head, "forward")
