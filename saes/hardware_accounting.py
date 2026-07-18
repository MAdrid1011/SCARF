"""Deterministic accounting for the SAES control and merge path.

The S2/S3 bypass fraction alone is not a hardware cost model: every early tile
still performs probe statistics, a first-hit decision, probe assignment, and
moment matching.  This module turns the *executed* tile-path counters emitted
by :mod:`saes.progressive_saes` into an explicit, target-free event ledger.

The ledger is deliberately conservative about visibility rather than claiming
RTL timing.  Its ``serialized_accounting_cycles`` assumes no overlap between
VectorALU work, controller transitions, and charged retained-descriptor buffer
traffic.  It is suitable for an analytic ablation penalty, but it must not be
reported as RTL-cycle-equivalent evidence until the corresponding datapath and
memory schedule are implemented and co-simulated.
"""

from __future__ import annotations

from math import ceil, log2
from typing import Any, Mapping

from saes.probe_layout import (
    compute_lightweight_positions,
    compute_probe_positions,
)


LEDGER_VERSION = "saes-event-ledger-v1"
DEFAULT_VECTOR_WIDTH = 64
DEFAULT_STORAGE_BEAT_BYTES = 16


def _count(stats: Mapping[str, Any], field: str, default: int = 0) -> int:
    """Read one nonnegative integral runtime counter."""
    value = stats.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _descriptor_layout_bytes(sh_degree: int) -> dict[str, int]:
    """Return the compact retained-Gaussian layout used by the event model.

    Geometry follows the paper's precision-sensitive policy (FP32 means and
    upper-triangular covariance); SH and opacity use the FP16 data path.  This
    is a storage accounting convention, not an assertion about a physical SRAM
    macro or final wire format.
    """
    _positive_int(sh_degree + 1, "sh_degree + 1")
    sh_coefficients = 3 * (sh_degree + 1) ** 2
    return {
        "mean_fp32": 3 * 4,
        "covariance_upper_fp32": 6 * 4,
        "harmonics_fp16": sh_coefficients * 2,
        "opacity_fp16": 2,
    }


def retained_descriptor_storage_layout(
    sh_degree: int,
    *,
    storage_beat_bytes: int = DEFAULT_STORAGE_BEAT_BYTES,
) -> dict[str, Any]:
    """Return the packed retained-descriptor storage contract.

    This target-free helper is shared by the analytic event ledger and the
    staged Chisel descriptor-buffer contract.  It describes logical packed
    128-bit beats only; it does not claim an SRAM macro, an RTL schedule, or a
    cycle-equivalent transfer time.
    """
    if isinstance(sh_degree, bool) or not isinstance(sh_degree, int) or sh_degree < 0:
        raise ValueError("sh_degree must be a nonnegative integer")
    storage_beat_bytes = _positive_int(
        storage_beat_bytes, "storage_beat_bytes"
    )
    descriptor_parts = _descriptor_layout_bytes(sh_degree)
    descriptor_bytes = sum(descriptor_parts.values())
    return {
        "descriptor_layout_bytes": descriptor_parts,
        "descriptor_bytes": descriptor_bytes,
        "storage_beat_bytes": storage_beat_bytes,
        "storage_beats": _ceil_div(descriptor_bytes, storage_beat_bytes),
    }


def _validated_anchor_count(
    stats: Mapping[str, Any],
    *,
    field: str,
    expected: int,
    active_tiles: int,
) -> int:
    """Validate an emitted retained-anchor count against the declared path."""
    value = _count(stats, field, expected)
    if active_tiles and value != expected:
        raise ValueError(
            f"{field} is inconsistent with the declared SAES tile path: "
            f"got {value}, expected {expected}"
        )
    if not active_tiles and value != 0:
        raise ValueError(f"{field} is nonzero without matching tiles")
    return value


def build_saes_event_ledger(
    saes_stats: Mapping[str, Any],
    *,
    feature_dim: int,
    tile_size: int,
    sh_degree: int,
    primitives_per_pixel: int = 1,
    vector_width: int = DEFAULT_VECTOR_WIDTH,
    storage_beat_bytes: int = DEFAULT_STORAGE_BEAT_BYTES,
) -> dict[str, Any]:
    """Build an explicit control/merge ledger from one executed SAES trace.

    Args:
        saes_stats: Runtime counters from ``ProgressiveSAES.process_all_tiles``.
        feature_dim: Width of the exact S1 feature tensor used for routing.
        tile_size: Declared square tile side length.
        sh_degree: Runtime Gaussian SH degree.
        primitives_per_pixel: Native primitives emitted per pixel position.
        vector_width: VectorALU width (64 in the submitted RTL configuration).
        storage_beat_bytes: Logical accounting beat.  The submitted design's
            public AXI interface is 128 bit; this value is only used for a
            no-overlap analytic charge, not a SRAM timing claim.

    The event formula intentionally uses only runtime path counts and fixed
    model dimensions.  It does not receive images, target RGB, target metrics,
    expected paper results, or withheld non-anchor descriptors.
    """
    feature_dim = _positive_int(feature_dim, "feature_dim")
    tile_size = _positive_int(tile_size, "tile_size")
    primitives_per_pixel = _positive_int(
        primitives_per_pixel, "primitives_per_pixel"
    )
    vector_width = _positive_int(vector_width, "vector_width")
    storage_beat_bytes = _positive_int(storage_beat_bytes, "storage_beat_bytes")
    if sh_degree < 0:
        raise ValueError("sh_degree must be nonnegative")

    total_tiles = _count(saes_stats, "total_tiles_processed")
    l0_tiles = _count(saes_stats, "level0_tiles")
    l1_tiles = _count(saes_stats, "level1_tiles")
    full_tiles = _count(saes_stats, "full_tiles")
    if l0_tiles + l1_tiles + full_tiles != total_tiles:
        raise ValueError("SAES tile-path counters do not cover total tiles")

    primary_probe_count = len(compute_probe_positions(tile_size))
    l1_anchor_count = len(compute_lightweight_positions(tile_size))
    tile_positions = tile_size * tile_size
    if primary_probe_count > tile_positions or l1_anchor_count > tile_positions:
        raise ValueError("SAES probe construction exceeds tile capacity")

    l0_anchors = _validated_anchor_count(
        saes_stats,
        field="l0_representatives",
        expected=l0_tiles * primary_probe_count * primitives_per_pixel,
        active_tiles=l0_tiles,
    )
    l1_anchors = _validated_anchor_count(
        saes_stats,
        field="l1_lightweight_anchors",
        expected=l1_tiles * l1_anchor_count * primitives_per_pixel,
        active_tiles=l1_tiles,
    )
    full_stage3 = _validated_anchor_count(
        saes_stats,
        field="full_stage3_gaussians",
        expected=full_tiles * tile_positions * primitives_per_pixel,
        active_tiles=full_tiles,
    )

    l0_nonanchors = l0_tiles * (tile_positions - primary_probe_count) * primitives_per_pixel
    l1_nonanchors = l1_tiles * (tile_positions - l1_anchor_count) * primitives_per_pixel
    total_nonanchors = l0_nonanchors + l1_nonanchors
    retained_anchors = l0_anchors + l1_anchors
    attribute_transport_pairs = _count(
        saes_stats, "adapter_offset_attribute_transport_uses"
    )
    expected_attribute_transport_pairs = (
        l0_nonanchors * primary_probe_count
        + l1_nonanchors * l1_anchor_count
    )
    if (
        attribute_transport_pairs
        and attribute_transport_pairs != expected_attribute_transport_pairs
    ):
        raise ValueError(
            "adapter_offset_attribute_transport_uses is inconsistent with "
            "the declared SAES tile path"
        )

    guard_enabled = saes_stats.get("materialization_guard_enabled", False)
    if not isinstance(guard_enabled, bool):
        raise ValueError("materialization_guard_enabled must be boolean")
    l0_guard_checks = _count(saes_stats, "l0_guard_checks")
    l1_guard_checks = _count(saes_stats, "l1_guard_checks")
    l0_guard_rejections = _count(saes_stats, "l0_guard_rejections")
    l1_guard_rejections = _count(saes_stats, "l1_guard_rejections")
    guard_attribute_reads = _count(saes_stats, "guard_anchor_attribute_reads")
    guard_nonprobe_reads = _count(saes_stats, "guard_nonprobe_s3_attribute_reads")
    if guard_nonprobe_reads != 0:
        raise ValueError("SAES materialization guard read a non-probe S3 attribute")
    if l0_guard_rejections > l0_guard_checks or l1_guard_rejections > l1_guard_checks:
        raise ValueError("SAES materialization guard rejections exceed checks")
    expected_guard_attribute_reads = 3 * primitives_per_pixel * (
        l0_guard_checks * primary_probe_count + l1_guard_checks * l1_anchor_count
    )
    if guard_enabled and guard_attribute_reads != expected_guard_attribute_reads:
        raise ValueError("SAES materialization guard anchor-read accounting is inconsistent")
    if not guard_enabled and any(
        (l0_guard_checks, l1_guard_checks, l0_guard_rejections, l1_guard_rejections, guard_attribute_reads)
    ):
        raise ValueError("disabled SAES materialization guard has event activity")

    # S3 path selection: two sufficient-statistic reductions (sum and square)
    # over K probe feature vectors.  A reduction tree contributes one cycle per
    # level for every 64-channel vector chunk.  L1's depth standard deviation
    # runs only after a failed L0 decision.
    feature_chunks = _ceil_div(feature_dim, vector_width)
    probe_elements = primary_probe_count * feature_dim
    feature_stat_vector_cycles_per_tile = 2 * _ceil_div(probe_elements, vector_width)
    feature_stat_reduce_cycles_per_tile = (
        2 * feature_chunks * int(ceil(log2(primary_probe_count)))
    )
    feature_stat_cycles = total_tiles * (
        feature_stat_vector_cycles_per_tile + feature_stat_reduce_cycles_per_tile
    )
    depth_stat_vector_cycles_per_tile = 2 * _ceil_div(primary_probe_count, vector_width)
    depth_stat_reduce_cycles_per_tile = 2 * int(ceil(log2(primary_probe_count)))
    depth_stat_cycles = (l1_tiles + full_tiles) * (
        depth_stat_vector_cycles_per_tile + depth_stat_reduce_cycles_per_tile
    )

    # The submitted SAESController has Idle -> CheckL0 -> Result for L0 and
    # Idle -> CheckL0 -> CheckL1 -> Result for an L1/Full result.  Its exposed
    # ``decisionCycles`` RTL event reports the same accepted-start-to-done
    # counts; Chisel tests cover all three outcome paths.
    controller_l0_cycles = 2 * l0_tiles
    controller_l1_full_cycles = 3 * (l1_tiles + full_tiles)
    controller_cycles = controller_l0_cycles + controller_l1_full_cycles

    descriptor_storage = retained_descriptor_storage_layout(
        sh_degree, storage_beat_bytes=storage_beat_bytes
    )
    descriptor_parts = descriptor_storage["descriptor_layout_bytes"]
    descriptor_bytes = descriptor_storage["descriptor_bytes"]
    descriptor_elements = 3 + 6 + 3 * (sh_degree + 1) ** 2 + 1
    descriptor_chunks = _ceil_div(descriptor_elements, vector_width)
    attribute_transport_elements = 3 * (sh_degree + 1) ** 2 + 1
    attribute_transport_chunks = _ceil_div(
        attribute_transport_elements, vector_width
    )
    guard_anchor_descriptors = primitives_per_pixel * (
        l0_guard_checks * primary_probe_count + l1_guard_checks * l1_anchor_count
    )
    # Covariance, SH, and opacity checks are a Control operation over already
    # selected native descriptors. Charge the descriptor fetch and one vector
    # comparison chunk per anchor rather than treating the check as free.
    guard_control_cycles = guard_anchor_descriptors * descriptor_chunks

    def _assignment_cycles(nonanchors: int, anchors_per_tile: int) -> int:
        # Feature distance: subtract/square plus reduce for every anchor.
        feature_distance = nonanchors * anchors_per_tile * (2 * feature_chunks)
        # The VectorALU softmax datapath consumes one score per cycle in both
        # its exp/sum and normalize phases, after one start cycle.
        softmax = nonanchors * (1 + 2 * anchors_per_tile)
        return feature_distance + softmax

    def _moment_cycles(nonanchors: int, anchors_per_tile: int, anchors: int) -> int:
        # Each pseudo descriptor updates every selected anchor's first/second
        # moments, SH, and opacity.  Final anchor normalization is separately
        # charged once per retained descriptor.
        update = nonanchors * anchors_per_tile * descriptor_chunks
        finalize = anchors * descriptor_chunks
        return update + finalize

    l0_assignment_cycles = _assignment_cycles(l0_nonanchors, primary_probe_count)
    l1_assignment_cycles = _assignment_cycles(l1_nonanchors, l1_anchor_count)
    l0_moment_cycles = _moment_cycles(l0_nonanchors, primary_probe_count, l0_anchors)
    l1_moment_cycles = _moment_cycles(l1_nonanchors, l1_anchor_count, l1_anchors)
    assignment_cycles = l0_assignment_cycles + l1_assignment_cycles
    moment_cycles = l0_moment_cycles + l1_moment_cycles
    # The standard moment charge includes the receiving-anchor update. The
    # adapter-offset attribute diagnostic first reconstructs the skipped SH and
    # opacity convex estimate from selected anchors, so charge that additional
    # fixed-function reduction rather than treating it as free.
    attribute_transport_cycles = (
        attribute_transport_pairs * attribute_transport_chunks
    )

    # Traffic is reported even where data can reuse an already-resident S1/S2
    # tile buffer.  Only the retained-descriptor read/rewrite and route record
    # transfer receive an explicit no-overlap cycle charge below; treating all
    # feature/depth reads as incremental would double count baseline S1/S2 IO.
    route_feature_read_bytes = total_tiles * primary_probe_count * feature_dim * 2
    assignment_feature_read_bytes = total_nonanchors * feature_dim * 2
    probe_depth_read_bytes = (l1_tiles + full_tiles) * primary_probe_count * 2
    retained_descriptor_read_bytes = retained_anchors * descriptor_bytes
    retained_descriptor_write_bytes = retained_anchors * descriptor_bytes
    guard_descriptor_read_bytes = guard_anchor_descriptors * descriptor_bytes
    attribute_transport_read_bytes = attribute_transport_pairs * (
        descriptor_parts["harmonics_fp16"] + descriptor_parts["opacity_fp16"]
    )
    route_record_write_bytes = total_tiles
    charged_storage_bytes = (
        retained_descriptor_read_bytes
        + retained_descriptor_write_bytes
        + guard_descriptor_read_bytes
        + attribute_transport_read_bytes
        + route_record_write_bytes
    )
    storage_transfer_cycles = _ceil_div(charged_storage_bytes, storage_beat_bytes)

    decision_cycles = (
        feature_stat_cycles + depth_stat_cycles + controller_cycles + guard_control_cycles
    )
    serialized_accounting_cycles = (
        decision_cycles
        + assignment_cycles
        + moment_cycles
        + attribute_transport_cycles
        + storage_transfer_cycles
    )

    return {
        "ledger_version": LEDGER_VERSION,
        "timing_class": "analytic_no_overlap_not_rtl_cycle_equivalent",
        "target_rgb_accessed": False,
        "inputs": {
            "feature_dim": feature_dim,
            "tile_size": tile_size,
            "sh_degree": sh_degree,
            "primitives_per_pixel": primitives_per_pixel,
            "vector_width": vector_width,
            "storage_beat_bytes": storage_beat_bytes,
            "primary_probe_count": primary_probe_count,
            "l1_anchor_count": l1_anchor_count,
            "descriptor_layout_bytes": descriptor_parts,
            "descriptor_bytes": descriptor_bytes,
            "descriptor_storage_beats": descriptor_storage["storage_beats"],
            "descriptor_elements": descriptor_elements,
        },
        "events": {
            "total_tiles": total_tiles,
            "l0_tiles": l0_tiles,
            "l1_tiles": l1_tiles,
            "full_tiles": full_tiles,
            "l0_retained_anchors": l0_anchors,
            "l1_retained_anchors": l1_anchors,
            "full_stage3_gaussians": full_stage3,
            **(
                {
                    "adapter_offset_attribute_transport_pairs": (
                        attribute_transport_pairs
                    ),
                }
                if attribute_transport_pairs
                else {}
            ),
            **(
                {
                    "l0_guard_checks": l0_guard_checks,
                    "l1_guard_checks": l1_guard_checks,
                    "l0_guard_rejections": l0_guard_rejections,
                    "l1_guard_rejections": l1_guard_rejections,
                    "guard_anchor_descriptors": guard_anchor_descriptors,
                    "guard_nonprobe_s3_attribute_reads": guard_nonprobe_reads,
                }
                if guard_enabled
                else {}
            ),
            "l0_nonanchors": l0_nonanchors,
            "l1_nonanchors": l1_nonanchors,
            "total_nonanchors": total_nonanchors,
        },
        "cycles": {
            "feature_stat_vector": feature_stat_vector_cycles_per_tile * total_tiles,
            "feature_stat_reduce": feature_stat_reduce_cycles_per_tile * total_tiles,
            "feature_stat_total": feature_stat_cycles,
            "depth_stat_vector": depth_stat_vector_cycles_per_tile * (l1_tiles + full_tiles),
            "depth_stat_reduce": depth_stat_reduce_cycles_per_tile * (l1_tiles + full_tiles),
            "depth_stat_total": depth_stat_cycles,
            "controller_l0": controller_l0_cycles,
            "controller_l1_full": controller_l1_full_cycles,
            "controller_total": controller_cycles,
            "materialization_guard": guard_control_cycles,
            "decision_total": decision_cycles,
            "l0_assignment": l0_assignment_cycles,
            "l1_assignment": l1_assignment_cycles,
            "assignment_total": assignment_cycles,
            "l0_moment_matching": l0_moment_cycles,
            "l1_moment_matching": l1_moment_cycles,
            "moment_matching_total": moment_cycles,
            "adapter_offset_attribute_reconstruction": attribute_transport_cycles,
            "storage_transfer": storage_transfer_cycles,
            "serialized_accounting_cycles": serialized_accounting_cycles,
        },
        "traffic_bytes": {
            "route_feature_read": route_feature_read_bytes,
            "assignment_feature_read": assignment_feature_read_bytes,
            "probe_depth_read": probe_depth_read_bytes,
            "retained_descriptor_read": retained_descriptor_read_bytes,
            "retained_descriptor_write": retained_descriptor_write_bytes,
            "materialization_guard_descriptor_read": guard_descriptor_read_bytes,
            "adapter_offset_attribute_transport_read": attribute_transport_read_bytes,
            "route_record_write": route_record_write_bytes,
            "charged_storage_total": charged_storage_bytes,
            "observed_total": (
                route_feature_read_bytes
                + assignment_feature_read_bytes
                + probe_depth_read_bytes
                + charged_storage_bytes
            ),
        },
        "assumptions": {
            "feature_and_probe_depth_reads": "reported as required traffic; not charged as incremental baseline IO",
            "retained_descriptor_buffer": "one conservative read plus one merged rewrite per retained anchor",
            "storage_cycle_model": "one logical 128-bit accounting beat per cycle without overlap",
            "controller_cycles": "matches submitted SAESController.decisionCycles traces",
            "materialization_guard": "probe-only Control comparison and descriptor reads are charged without adding an S2/S3 saving claim",
            "adapter_offset_attribute_transport": (
                "selected-anchor SH/opacity reconstruction is charged as "
                "conservative FP16 descriptor reads plus VectorALU reduction; "
                "it does not create an S2/S3 saving claim"
            ),
            "rtl_cycle_equivalent": False,
        },
    }
