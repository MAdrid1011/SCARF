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
from typing import Any, Iterator

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
            "scoped_incremental_patch_replay_fp32_equivalence_only"
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
        second_patches = _gather_padded_patches(
            self._hidden_values[batch_item : batch_item + 1],
            new_coordinates,
            padding_mode=self._second.padding_mode,
        )
        final_values = _apply_conv_to_patches(self._second, second_patches)[0]
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
        }

    def execute(self, phase: str, selection_mask: torch.Tensor) -> dict[str, Any]:
        """Execute one route phase and return its source-bound per-tile ledger."""
        self._validate_phase(phase)
        mask = self._validated_mask(selection_mask)
        if phase == "full_extension" and bool((mask & self._computed_mask).any()):
            raise ValueError("Full extension may not replay an already produced position")
        per_tile: list[dict[str, Any]] = []
        totals = {
            "head_final_positions_requested": 0,
            "head_final_positions_reused": 0,
            "head_final_positions_executed": 0,
            "first_conv_positions_reused": 0,
            "first_conv_positions_executed": 0,
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
            "first_conv_positions_executed": totals["first_conv_positions_executed"],
            "second_conv_positions_executed": totals["head_final_positions_executed"],
            "full_tile_native_identity_verified": False,
            "full_execution_mode": (
                "incremental_patch_replay_fp32_equivalence_only"
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
            "head_final_positions_executed": final_positions,
            "second_conv_positions_executed": final_positions,
            "dense_head_macs": dense_positions * (first_macs + second_macs),
            "actual_head_macs": first_positions * first_macs + final_positions * second_macs,
            "head_mac_delta": dense_positions * (first_macs + second_macs)
            - (first_positions * first_macs + final_positions * second_macs),
            "full_tile_native_identity_verified": False,
            "full_execution_mode": "incremental_patch_replay_fp32_equivalence_only",
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
            "scoped_incremental_patch_replay_initial_route_only"
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
