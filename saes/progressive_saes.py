"""
SAES: Scene-Adaptive Early Sparsification — Multi-Level v4 Simulator

Progressive Adaptive Early Sparsification for SCARF with 2-level tile classification
aligned with the SCARF Dataflow specification (L0 + L1 only):

  Level 0 (Feature Pre-Filter): Tiles with low feature variance after S1.
    Process K(T) probes through S2+S3 and merge assigned non-probe Gaussians
    with first/second-moment matching. Set non-probe opacity=0.
    Effective Gaussian count reduced to K(T).

  Level 1 (Depth-Based): Tiles that pass L0 but have uniform primary-probe
    depths. Use the paper's probe-constrained decision and depth-reliability
    term, then execute a deterministic 2K(T) native-anchor lightweight path.
    Set the remaining non-anchor opacity=0. At T=4 this retains 8 of 16
    Gaussians, making L1 less compressive than the 4-of-16 L0 path.

  Full: Tiles that fail both checks — no SAES savings.

The L0 path reads Stage-3 attributes only for the K(T) declared probe
Gaussians. L1 routing adds primary-probe depth evidence and a depth-reliability
factor; after that routing decision it executes the declared deterministic
2K(T) native anchors. Non-anchor assignment uses S1 features, tile coordinates,
and only those executed-anchor values; it never reads the remaining tile
positions.

K(T) probe count formula (Spec §Stage 3):
    K(T) = 4 + ceil(2·log₂(T/4))   for T ≥ 4
    e.g. T=4 → 4 probes, T=8 → 6, T=16 → 8
    Additional M=K(T)−4 probes selected from a G×G uniform subgrid (G=⌈√M⌉)
    of the non-corner interior, one probe per cell, nearest to cell centre.
"""

import math
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Mapping, Tuple

from saes.probe_layout import (
    compute_lightweight_positions as _compute_lightweight_positions,
)
from saes.probe_layout import compute_probe_positions as _compute_probe_positions
from saes.guard_policy import (
    CONTEXT_GUARD_MAX_CENTER_MAHALANOBIS,
    CONTEXT_GUARD_MAX_FOOTPRINT_RATIO,
    CONTEXT_GUARD_MAX_RELATIVE_DEPTH_SPAN,
    MATERIALIZATION_GUARD_MAX_OPACITY_DISTANCE,
    MATERIALIZATION_GUARD_MIN_COVARIANCE_COSINE,
    MATERIALIZATION_GUARD_MIN_HARMONIC_COSINE,
)


LEGACY_L1_PRIMARY_DEPTH_REFERENCE_MATERIALIZATION = (
    "l1-primary-depth-reference-diagnostic"
)

DELETION_CERTIFICATE_POLICY = "exact-source-opacity-zero-v1"
"""The only currently sound source-only representative deletion certificate."""

DELETION_CERTIFICATE_SOURCE_KIND = "s2-density-adapter-opacity-v1"
"""Opacity came directly from the TranSplat/MVSplat S2 density mapping."""

PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS = (
    "paper-probe-normalized-feature-first-hit"
)
"""Explicit engineering interpretation of the paper's unspecified feature scale.

The literal ``paper-probe-feature-variance-first-hit`` diagnostic remains raw.
This distinct mode compares ``tau_f`` with the standard deviation of
unit-normalized probe feature vectors, then squares that score only when the
bilateral assignment needs a variance.
"""

def paper_assignment_weights(
    spatial_distances: torch.Tensor,
    feature_distances: torch.Tensor,
    *,
    feature_variance: float,
    beta_x: float,
    beta_f: float,
    level: str,
    probe_depths: torch.Tensor | None = None,
    depth_reference_depths: torch.Tensor | None = None,
    beta_d: float | None = None,
) -> torch.Tensor:
    """Compute the Section 3 bilateral L0/L1 probe-assignment weights.

    The L0 spatial term is explicitly modulated by the tile's probe-feature
    variance.  L1 reuses that bilateral term and multiplies it by the paper's
    probe-depth reliability factor, expressed in log space before softmax.
    """
    if (
        spatial_distances.ndim != 1
        or feature_distances.shape != spatial_distances.shape
    ):
        raise ValueError("assignment distances must be equal-length vectors")
    if not math.isfinite(feature_variance) or feature_variance < 0.0:
        raise ValueError("feature_variance must be finite and nonnegative")
    if level not in {"L0", "L1"}:
        raise ValueError("assignment level must be L0 or L1")
    if beta_x <= 0.0 or beta_f <= 0.0:
        raise ValueError("assignment bandwidths must be positive")
    if (
        not torch.isfinite(spatial_distances).all()
        or not torch.isfinite(feature_distances).all()
    ):
        raise ValueError("assignment distances must be finite")

    logits = -(
        float(feature_variance) * spatial_distances / (beta_x**2 + 1e-8)
        + feature_distances / (beta_f**2 + 1e-8)
    )
    if level == "L1":
        if probe_depths is None or probe_depths.shape != spatial_distances.shape:
            raise ValueError("L1 assignment requires one probe depth per weight")
        if beta_d is None or beta_d <= 0.0 or not torch.isfinite(probe_depths).all():
            raise ValueError("L1 assignment requires finite depths and positive beta_d")
        reference_depths = (
            probe_depths
            if depth_reference_depths is None
            else depth_reference_depths.reshape(-1).to(
                device=probe_depths.device, dtype=probe_depths.dtype
            )
        )
        if reference_depths.numel() < 1 or not torch.isfinite(reference_depths).all():
            raise ValueError("L1 depth reference must contain finite values")
        depth_mean = reference_depths.mean()
        depth_std = reference_depths.std(unbiased=False)
        logits = logits - (probe_depths - depth_mean).abs() / (
            beta_d * depth_std + 1e-8
        )
    return torch.softmax(logits, dim=0)


class ProgressiveSAES:
    """
    Progressive Adaptive Early Sparsification for SCARF (v4: Dataflow-aligned).

    Key changes vs v3:
      - K(T) adaptive probe count replaces fixed 4-corner probes.
      - Weighted moment matching (L0/L1) absorbs non-anchor Gaussians into
        the selected outputs.
      - Single-gate update (L2) blends toward selected-anchor means.
      - Non-anchor pixels have opacity set to 0 after an early stop. L0 retains
        K(T) representatives and the less-compressive L1 path retains 2K(T).
    """

    # ------------------------------------------------------------------ #
    # Static helpers                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_probe_positions(tile_size: int) -> List[Tuple[int, int]]:
        """
        Compute K(T) adaptive probe positions inside a T×T tile.

        Spec: K(T) = 4 + ceil(2·log₂(T/4)), minimum 4 for T=4.
        Layout:
          - 4 corner probes: (0,0), (0,T-1), (T-1,0), (T-1,T-1)
          - M = K(T)-4 additional probes from a G×G uniform subgrid of the
            non-corner interior (rows 1..T-2, cols 1..T-2), with G=ceil(√M).
            For each cell one probe is placed at the pixel nearest to the cell
            centre; at most one probe per cell (spatial dispersion guarantee).
        """
        return _compute_probe_positions(tile_size)

    @staticmethod
    def compute_lightweight_positions(tile_size: int) -> List[Tuple[int, int]]:
        """Return deterministic L1 anchors with twice the L0 probe capacity.

        The paper leaves the lightweight retained-output count unspecified.
        This declared engineering detail keeps the primary probes first and
        uses farthest-point sampling for the additional anchors.  The route
        itself still depends only on the primary probes; after an L1 decision
        the 2K positions are executed as native S2/S3 outputs.
        """
        return _compute_lightweight_positions(tile_size)

    @staticmethod
    def _validate_camera_geometry(
        context_extrinsics: torch.Tensor,
        context_intrinsics: torch.Tensor,
        *,
        view_count: int,
        ray_depth_mode: str,
    ) -> None:
        if not torch.is_tensor(context_extrinsics) or not torch.is_tensor(
            context_intrinsics
        ):
            raise ValueError("context camera geometry must be tensors")
        if context_extrinsics.shape != (1, view_count, 4, 4):
            raise ValueError("context extrinsics must have shape [1, views, 4, 4]")
        if context_intrinsics.shape != (1, view_count, 3, 3):
            raise ValueError("context intrinsics must have shape [1, views, 3, 3]")
        if context_extrinsics.device != context_intrinsics.device:
            raise ValueError("context camera geometry must share one device")
        if ray_depth_mode not in {"euclidean", "z"}:
            raise ValueError("ray depth mode must be euclidean or z")

    @staticmethod
    def camera_world_point(
        context_extrinsics: torch.Tensor,
        context_intrinsics: torch.Tensor,
        *,
        view_index: int,
        row: int,
        column: int,
        height: int,
        width: int,
        depth: torch.Tensor,
        ray_depth_mode: str,
    ) -> torch.Tensor:
        """Lift one pixel-depth pair with the upstream normalized C2W convention."""
        ProgressiveSAES._validate_camera_geometry(
            context_extrinsics,
            context_intrinsics,
            view_count=context_extrinsics.shape[1]
            if torch.is_tensor(context_extrinsics) and context_extrinsics.ndim >= 2
            else 0,
            ray_depth_mode=ray_depth_mode,
        )
        if (
            not isinstance(view_index, int)
            or not 0 <= view_index < context_extrinsics.shape[1]
        ):
            raise ValueError("camera view index is out of range")
        if (
            height <= 0
            or width <= 0
            or not 0 <= row < height
            or not 0 <= column < width
        ):
            raise ValueError("camera pixel coordinate is out of range")
        if not torch.is_tensor(depth) or depth.numel() != 1:
            raise ValueError("camera depth must be a scalar tensor")

        dtype = context_extrinsics.dtype
        device = context_extrinsics.device
        coordinate = torch.tensor(
            ((column + 0.5) / width, (row + 0.5) / height, 1.0),
            device=device,
            dtype=dtype,
        )
        camera_direction = torch.linalg.solve(
            context_intrinsics[0, view_index], coordinate
        )
        if ray_depth_mode == "euclidean":
            camera_direction = camera_direction / camera_direction.norm().clamp_min(
                1e-8
            )
        else:
            denominator = camera_direction[2]
            if denominator.abs() < 1e-8:
                denominator = torch.full_like(denominator, 1e-8)
            camera_direction = camera_direction / denominator
        transform = context_extrinsics[0, view_index]
        world_direction = transform[:3, :3] @ camera_direction
        return transform[:3, 3] + world_direction * depth.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------ #
    # Initialisation                                                        #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        H: int,
        W: int,
        initial_tile_size: int = 4,
        feature_var_threshold: float = None,
        depth_std_threshold: float = None,
        cross_check_threshold: float = 0.02,
        view_count: int = 1,
        primitives_per_pixel: int = 1,
        materialization: str = "representative",
        decision_semantics: str = "current",
        beta_x: float = 0.50,
        beta_f: float = 0.10,
        beta_d: float = 1.00,
        num_depth_candidates: int = 1,
        context_extrinsics: torch.Tensor | None = None,
        context_intrinsics: torch.Tensor | None = None,
        ray_depth_mode: str = "euclidean",
        materialization_guard: bool = True,
        context_safety_guard: bool = False,
        require_deletion_certificate: bool = False,
        joint_calibrator: Any | None = None,
        joint_calibrator_model: str | None = None,
        joint_calibrator_selected_head: Mapping[str, Any] | None = None,
        offline_joint_calibration_capture: Any | None = None,
    ):
        self.H = H
        self.W = W
        self.initial_tile_size = initial_tile_size
        # L0: tighter default so only truly flat/uniform regions early-stop.
        self.feature_var_threshold = (
            feature_var_threshold if feature_var_threshold is not None else 0.004
        )
        # L1: 4% relative depth std allows depth-uniform tiles to be caught.
        self.depth_std_threshold = (
            depth_std_threshold if depth_std_threshold is not None else 0.04
        )
        self.cross_check_threshold = cross_check_threshold
        if (
            isinstance(cross_check_threshold, bool)
            or not isinstance(cross_check_threshold, (int, float))
            or not math.isfinite(float(cross_check_threshold))
            or cross_check_threshold < 0.0
        ):
            raise ValueError("cross_check_threshold must be a finite nonnegative scalar")
        # A zero threshold has historically represented the diagnostic route
        # without this registered materialization check.  The frozen SAES
        # execution identity supplies 0.015, so its guarded route is always
        # enabled while legacy/direct callers retain an explicit opt-out.
        self.probe_cross_check_enabled = bool(
            materialization_guard and cross_check_threshold > 0.0
        )
        if view_count < 1 or primitives_per_pixel < 1:
            raise ValueError("view_count and primitives_per_pixel must be positive")
        self.view_count = view_count
        self.primitives_per_pixel = primitives_per_pixel
        if materialization not in (
            "representative",
            "dense-diagnostic",
            "probe-spread-diagnostic",
            "transmittance-diagnostic",
            "virtual-reconstruction-diagnostic",
            "assignment-consensus-adapter-pseudo-descriptor-diagnostic",
            "same-budget-dense-oracle-diagnostic",
            LEGACY_L1_PRIMARY_DEPTH_REFERENCE_MATERIALIZATION,
            "conditional-anchor-transport-diagnostic",
            "conditional-adapter-offset-transport-diagnostic",
            "conditional-adapter-offset-attribute-transport-diagnostic",
            "conditional-optical-mass-diagnostic",
            "conditional-projected-optical-mass-diagnostic",
            "multicontext-tangent-plane-diagnostic",
        ):
            raise ValueError(f"unsupported SAES materialization: {materialization}")
        # The former diagnostic name represented the paper's actual L1
        # semantics. Keep it parseable for historical invocation records, but
        # normalize it to the ordinary representative implementation so it
        # cannot remain a divergent execution mode.
        if materialization == LEGACY_L1_PRIMARY_DEPTH_REFERENCE_MATERIALIZATION:
            materialization = "representative"
        self.materialization = materialization
        if not isinstance(materialization_guard, bool):
            raise ValueError("materialization_guard must be boolean")
        self.materialization_guard = materialization_guard
        if not isinstance(context_safety_guard, bool):
            raise ValueError("context_safety_guard must be boolean")
        if context_safety_guard and not materialization_guard:
            raise ValueError(
                "context_safety_guard requires materialization_guard to be enabled"
            )
        self.context_safety_guard = context_safety_guard
        if not isinstance(require_deletion_certificate, bool):
            raise ValueError("require_deletion_certificate must be boolean")
        self.require_deletion_certificate = require_deletion_certificate
        self.joint_calibrator = None
        self._joint_calibrator_contract: dict[str, Any] | None = None
        self._offline_joint_calibration_capture = None
        if offline_joint_calibration_capture is not None:
            if joint_calibrator is not None:
                raise ValueError(
                    "offline joint-calibration capture cannot run with a calibrator"
                )
            if materialization != "representative":
                raise ValueError(
                    "offline joint-calibration capture only supports representative materialization"
                )
            if not callable(offline_joint_calibration_capture):
                raise TypeError("offline joint-calibration capture must be callable")
            # This is an author-side cache compiler hook, never a runtime
            # correction path. It can observe only the already selected-only
            # descriptor and ordinary representative output below.
            self._offline_joint_calibration_capture = offline_joint_calibration_capture
        if joint_calibrator is None:
            if (
                joint_calibrator_model is not None
                or joint_calibrator_selected_head is not None
            ):
                raise ValueError(
                    "joint calibrator model and selected-head events require a calibrator"
                )
        else:
            from saes.joint_materialization_calibrator import (
                ASSET_SCHEMA_VERSION,
                FP16_PARAMETER_BYTES,
                NETWORK_MACS_PER_DESCRIPTOR,
                JointMaterializationCalibrator,
            )

            if not isinstance(joint_calibrator, JointMaterializationCalibrator):
                raise TypeError(
                    "joint_calibrator must be a JointMaterializationCalibrator"
                )
            if materialization != "representative":
                raise ValueError(
                    "joint calibrator only extends representative materialization"
                )
            if joint_calibrator_model not in {"transplat", "mvsplat", "depthsplat"}:
                raise ValueError(
                    "joint calibrator requires a model-specific selected-head replay contract"
                )
            if not isinstance(joint_calibrator_selected_head, Mapping):
                raise ValueError("joint calibrator selected-head events are required")
            selected_head = dict(joint_calibrator_selected_head)
            if selected_head.get("model") != joint_calibrator_model:
                raise ValueError("joint calibrator selected-head model does not match")
            if selected_head.get("contract_version") != "saes-selected-output-replay-v1":
                raise ValueError("joint calibrator selected-head contract is invalid")
            for name in ("dense_head_macs", "replayed_head_macs"):
                value = selected_head.get(name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        f"joint calibrator selected-head {name} is invalid"
                    )
            if selected_head["dense_head_macs"] <= selected_head["replayed_head_macs"]:
                raise ValueError("joint calibrator selected-head trace has no skipped MACs")
            asset_sha256 = getattr(joint_calibrator, "runtime_asset_sha256", None)
            if (
                not isinstance(asset_sha256, str)
                or len(asset_sha256) != 64
                or any(character not in "0123456789abcdef" for character in asset_sha256)
            ):
                raise ValueError("joint calibrator requires a hash-pinned asset")
            cost = joint_calibrator.cost_contract()
            if (
                not isinstance(cost, Mapping)
                or cost.get("schema_version") != ASSET_SCHEMA_VERSION
                or cost.get("network_macs_per_descriptor")
                != NETWORK_MACS_PER_DESCRIPTOR
                or cost.get("fp16_parameter_bytes") != FP16_PARAMETER_BYTES
            ):
                raise ValueError("joint calibrator cost contract is invalid")
            self.joint_calibrator = joint_calibrator.eval()
            self._joint_calibrator_contract = {
                "schema_version": ASSET_SCHEMA_VERSION,
                "asset_sha256": asset_sha256,
                "model": joint_calibrator_model,
                "calls": 0,
                "l0_calls": 0,
                "l1_calls": 0,
                "full_calls": 0,
                "selected_descriptor_reads": 0,
                "network_macs_per_call": NETWORK_MACS_PER_DESCRIPTOR,
                # Biases are executable parameters too, so this is the total
                # FP16 parameter footprint rather than matrix weights alone.
                "weight_bytes": FP16_PARAMETER_BYTES,
                "selected_head": selected_head,
            }
        self.multicontext_tangent_enabled = (
            materialization == "multicontext-tangent-plane-diagnostic"
        )
        # The paper's soft assignment associates a skipped position with its
        # receiving probe.  The historical implementation first formed an
        # unconditional mixture over *all* probes and then applied the same
        # assignment again while updating each probe.  This diagnostic instead
        # transports each selected anchor conditionally to its assigned target
        # position, avoiding an undocumented r_i,p * r_i,q cross-anchor term
        # for geometry.  The attribute-transport diagnostic retains that
        # geometry, but reconstructs skipped SH/opacity from the declared
        # bilateral assignments before the receiving-anchor update.
        self.merge_semantics = (
            "same-budget-dense-oracle"
            if materialization == "same-budget-dense-oracle-diagnostic"
            else "assignment-consensus-adapter-pseudo-descriptor"
            if materialization
            == "assignment-consensus-adapter-pseudo-descriptor-diagnostic"
            else "conditional-adapter-offset-attribute-transport"
            if materialization
            in (
                "conditional-adapter-offset-attribute-transport-diagnostic",
                "multicontext-tangent-plane-diagnostic",
            )
            else "conditional-adapter-offset-transport"
            if materialization == "conditional-adapter-offset-transport-diagnostic"
            else "conditional-projected-optical-mass"
            if materialization == "conditional-projected-optical-mass-diagnostic"
            else "conditional-optical-mass"
            if materialization == "conditional-optical-mass-diagnostic"
            else "conditional-anchor-transport"
            if materialization == "conditional-anchor-transport-diagnostic"
            else "assignment-mixture"
        )
        # The paper defines L1 depth reliability over the K primary routing
        # probes. The declared 2K native-anchor expansion may supply extra
        # descriptors after routing, but it must not redefine that reference.
        self.l1_depth_reference = "primary-probes"
        if decision_semantics not in (
            "current",
            "probe-vector-first-hit",
            "probe-channel-variance-first-hit",
            "probe-normalized-std-first-hit",
            "paper-probe-feature-variance-first-hit",
            PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS,
        ):
            raise ValueError(
                f"unsupported SAES decision semantics: {decision_semantics}"
            )
        self.decision_semantics = decision_semantics
        for label, value in (
            ("beta_x", beta_x),
            ("beta_f", beta_f),
            ("beta_d", beta_d),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{label} must be finite and positive")
        self.beta_x = float(beta_x)
        self.beta_f = float(beta_f)
        self.beta_d = float(beta_d)
        if (
            isinstance(num_depth_candidates, bool)
            or not isinstance(num_depth_candidates, int)
            or num_depth_candidates <= 0
        ):
            raise ValueError("num_depth_candidates must be positive")
        self.num_depth_candidates = num_depth_candidates
        if (context_extrinsics is None) != (context_intrinsics is None):
            raise ValueError(
                "both context extrinsics and intrinsics are required for camera geometry"
            )
        self.context_extrinsics = context_extrinsics
        self.context_intrinsics = context_intrinsics
        self.ray_depth_mode = ray_depth_mode
        self._camera_origins = None
        self._camera_directions = None
        if context_extrinsics is not None:
            self._validate_camera_geometry(
                context_extrinsics,
                context_intrinsics,
                view_count=view_count,
                ray_depth_mode=ray_depth_mode,
            )
            self._camera_origins, self._camera_directions = self._build_camera_rays()

        # --- Adaptive probe positions (K(T) formula) ---
        T = initial_tile_size
        self.probe_positions: List[Tuple[int, int]] = self.compute_probe_positions(T)
        self.lightweight_positions: List[Tuple[int, int]] = (
            self.compute_lightweight_positions(T)
        )
        _probe_set = set(self.probe_positions)
        self.non_probe_positions: List[Tuple[int, int]] = [
            (r, c) for r in range(T) for c in range(T) if (r, c) not in _probe_set
        ]

        # Backward-compatible class-level aliases (read by external tooling)
        self.__class__.PROBE_POSITIONS = self.probe_positions
        self.__class__.NON_PROBE_POSITIONS = self.non_probe_positions

        # Statistics
        self.stats: Dict = {
            "decision_semantics": self.decision_semantics,
            "cross_check_threshold": self.cross_check_threshold,
            "l1_depth_reference": self.l1_depth_reference,
            "merge_semantics": self.merge_semantics,
            "feature_statistic": (
                "raw-probe-mean-channel-variance"
                if self.decision_semantics
                in {
                    "probe-channel-variance-first-hit",
                    "paper-probe-feature-variance-first-hit",
                }
                else "raw-probe-vector-variance"
                if self.decision_semantics == "probe-vector-first-hit"
                else "normalized-probe-vector-standard-deviation"
                if self.decision_semantics
                in {
                    "probe-normalized-std-first-hit",
                    PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS,
                }
                else "normalized-probe-total-variance"
            ),
            "camera_aware_moment_matching": context_extrinsics is not None,
            "total_tiles_processed": 0,
            "level0_tiles": 0,
            "level1_tiles": 0,
            "full_tiles": 0,
            "level0_pixels": 0,
            "level1_pixels": 0,
            "pixels_original": 0,
            "total_modified_pixels": 0,
            "effective_gaussians": 0,
            "zeroed_gaussians": 0,
            "l0_representatives": 0,
            "l1_lightweight_anchors": 0,
            "full_stage3_gaussians": 0,
            "assignment_weight_sum_error_max": 0.0,
            "opacity_transmittance_error_max": 0.0,
            # Only the diagnostic optical-depth path uses this identity.  It
            # measures arithmetic agreement with the selected-anchor
            # surrogate, not agreement with withheld Stage-3 outputs.
            "optical_depth_assignment_error_max": 0.0,
            # Virtual reconstruction retains a descriptor at every pixel but
            # obtains its non-anchor descriptor from fixed-function anchor
            # interpolation rather than an S3 network evaluation.
            "virtual_reconstructed_gaussians": 0,
            # Assignment-consensus pseudo descriptors are virtual skipped
            # outputs only. They do not modify selected anchors or establish a
            # sparse-execution claim, so their geometry work is tracked apart
            # from the historical SH/opacity attribute transport diagnostic.
            "assignment_consensus_pseudo_outputs": 0,
            "assignment_consensus_anchor_pairs": 0,
            "assignment_consensus_offset_recoveries": 0,
            "assignment_consensus_target_lifts": 0,
            "assignment_consensus_fallback_tiles": 0,
            "assignment_consensus_l0_fallback_tiles": 0,
            "assignment_consensus_l1_fallback_tiles": 0,
            # This post-hoc diagnostic may read every dense Stage-3 descriptor
            # in a candidate tile, but it must still render only the paper's
            # K/2K retained outputs.  It is deliberately non-runtime and
            # cannot support a paper result or an S2/S3 saving claim.
            "same_budget_dense_oracle_tiles": 0,
            "same_budget_dense_oracle_l0_tiles": 0,
            "same_budget_dense_oracle_l1_tiles": 0,
            "same_budget_dense_oracle_fallback_tiles": 0,
            "same_budget_dense_oracle_full_stage3_reads": 0,
            "same_budget_dense_oracle_output_gaussians": 0,
            "same_budget_dense_oracle_failure_reasons": {},
            # These verify the local first-order construction only. Renderer
            # fidelity is measured separately by the committed image run.
            "same_budget_dense_oracle_mass_construction_error_max": 0.0,
            "same_budget_dense_oracle_projected_moment_construction_error_max": 0.0,
            "same_budget_dense_oracle_required_footprint_expansion_max": 1.0,
            "same_budget_dense_oracle_runtime_eligible": False,
            # This candidate changes only a retained representative covariance
            # after the ordinary selected-anchor construction. It is a
            # target-free geometry diagnostic, not a router or saving claim.
            "multicontext_tangent_enabled": self.multicontext_tangent_enabled,
            "multicontext_tangent_attempts": 0,
            "multicontext_tangent_accepted": 0,
            "multicontext_tangent_local_fallbacks": 0,
            "multicontext_tangent_context_camera_reads": 0,
            "multicontext_tangent_constraint_count": 0,
            "multicontext_tangent_contributor_count": 0,
            "multicontext_tangent_residual_max": 0.0,
            "multicontext_tangent_fallback_reasons": {},
            "multicontext_tangent_runtime_eligible": False,
            "covariance_psd_violations": 0,
            # Conditional optical-density transport is diagnostic until it
            # passes the target-free and image-quality gates.  These counters
            # make its mass accounting and any Full fallback explicit.
            "conditional_mass_conservation_error_max": 0.0,
            "conditional_assignment_uses": 0,
            "conditional_mass_fallback_tiles": 0,
            "conditional_range_fallback_tiles": 0,
            # This diagnostic reconstructs the producer's bounded image-plane
            # offset from a selected anchor and applies it before normalizing
            # the target C2W ray.  It is fail-closed when that reconstruction
            # cannot be justified from selected-anchor/context inputs alone.
            "adapter_offset_transport_uses": 0,
            "adapter_offset_transport_fallback_tiles": 0,
            # The attribute-transport diagnostic reconstructs skipped SH and
            # opacity from the existing bilateral assignment matrix. This
            # counts selected-anchor attribute pairs for the conservative
            # analytic ledger; it is not an S2/S3 saving counter.
            "adapter_offset_attribute_transport_uses": 0,
            # This is Control logic for the paper's existing probe-only
            # materialization path. It inspects only the selected native
            # anchors before their moments are merged; it is not a fourth
            # routing level or a Table 4 module.
            "l0_guard_checks": 0,
            "l0_guard_rejections": 0,
            "l1_guard_checks": 0,
            "l1_guard_rejections": 0,
            "l1_guard_attempts_after_l0_rejection": 0,
            # The registered leave-one-out probe consistency check is a
            # materialization guard, not another feature/depth routing tier.
            # It reads only the primary anchors already requested for an L0
            # or L1 candidate and fails closed to Full on disagreement.
            "probe_cross_check_enabled": self.probe_cross_check_enabled,
            "probe_cross_check_l0_checks": 0,
            "probe_cross_check_l1_checks": 0,
            "probe_cross_check_l0_rejections": 0,
            "probe_cross_check_l1_rejections": 0,
            "guard_anchor_attribute_reads": 0,
            "guard_nonprobe_s3_attribute_reads": 0,
            "materialization_guard_enabled": self.materialization_guard,
            "context_safety_guard_enabled": self.context_safety_guard,
            "deletion_certificate_required": self.require_deletion_certificate,
            "deletion_certificate_policy": DELETION_CERTIFICATE_POLICY,
            "deletion_certificate_source_kind": None,
            "deletion_certificate_source_status": "not_checked",
            "deletion_certificate_checks": 0,
            "deletion_certificate_accepted_tiles": 0,
            "deletion_certificate_rejected_tiles": 0,
            "deletion_certificate_zero_opacity_gaussians": 0,
            "deletion_certificate_anchor_s3_opacity_reads": 0,
            "deletion_certificate_nonprobe_s3_attribute_reads": 0,
            "deletion_certificate_rejection_reasons": {},
            "uncertified_deletion_fallback_tiles": 0,
            "context_guard_checks": 0,
            "context_guard_rejections": 0,
            "context_guard_missing_geometry": 0,
            "context_guard_invalid_footprint": 0,
            "context_guard_coverage_rejections": 0,
            "context_guard_occlusion_rejections": 0,
            "context_guard_center_separation_rejections": 0,
            "offline_joint_calibration_capture_calls": 0,
        }
        if self._joint_calibrator_contract is not None:
            self.stats["joint_calibrator"] = {
                **self._joint_calibrator_contract,
                "selected_head": dict(
                    self._joint_calibrator_contract["selected_head"]
                ),
            }

    def _build_camera_rays(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build C2W rays once for the resident model/sample session."""
        if self.context_extrinsics is None or self.context_intrinsics is None:
            raise RuntimeError("camera geometry was not configured")
        device = self.context_extrinsics.device
        dtype = self.context_extrinsics.dtype
        rows = (torch.arange(self.H, device=device, dtype=dtype) + 0.5) / self.H
        columns = (torch.arange(self.W, device=device, dtype=dtype) + 0.5) / self.W
        grid_y, grid_x = torch.meshgrid(rows, columns, indexing="ij")
        coordinates = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=-1)
        camera_directions = torch.einsum(
            "vij,hwj->vhwi",
            torch.linalg.inv(self.context_intrinsics[0]),
            coordinates,
        )
        if self.ray_depth_mode == "euclidean":
            camera_directions = camera_directions / camera_directions.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
        else:
            denominator = camera_directions[..., 2:3]
            denominator = torch.where(
                denominator.abs() < 1e-8,
                torch.full_like(denominator, 1e-8),
                denominator,
            )
            camera_directions = camera_directions / denominator
        world_directions = torch.einsum(
            "vij,vhwj->vhwi", self.context_extrinsics[0, :, :3, :3], camera_directions
        )
        return self.context_extrinsics[0, :, :3, 3], world_directions

    def _camera_world_point(
        self, view_index: int, row: int, column: int, depth: torch.Tensor
    ) -> torch.Tensor:
        if self._camera_origins is None or self._camera_directions is None:
            raise RuntimeError("camera geometry was not configured")
        return self._camera_origins[view_index] + self._camera_directions[
            view_index, row, column
        ] * depth.to(
            device=self._camera_directions.device,
            dtype=self._camera_directions.dtype,
        )

    def _camera_aware_interpolated_mean(
        self,
        source_means: torch.Tensor,
        source_depths: torch.Tensor,
        source_positions: List[Tuple[int, int]],
        *,
        view_index: int,
        target_row: int,
        target_column: int,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate a probe Gaussian at a new pixel without discarding offsets.

        Gaussian adaptors predict a sub-pixel image-plane offset in addition to
        depth.  A C2W ray reconstructed only from a target pixel centre loses
        that probe-produced geometric attribute.  Preserve it by transporting
        the assignment-weighted residual between each probe mean and its
        depth-only C2W ray to the target ray.  All inputs are probe outputs or
        S1/S2-derived quantities; this helper never needs a non-probe S3 value.
        """
        if self._camera_directions is None:
            raise RuntimeError("camera geometry was not configured")
        count = len(source_positions)
        if (
            source_means.shape != (count, 3)
            or source_depths.numel() != count
            or weights.numel() != count
        ):
            raise ValueError("camera interpolation inputs must align with probes")
        source_depths = source_depths.reshape(-1).to(
            device=source_means.device, dtype=source_means.dtype
        )
        weights = weights.reshape(-1).to(
            device=source_means.device, dtype=source_means.dtype
        )
        source_rays = torch.stack(
            [
                self._camera_world_point(view_index, row, column, depth)
                for (row, column), depth in zip(source_positions, source_depths)
            ],
            dim=0,
        )
        residuals = source_means - source_rays
        target_depth = torch.dot(weights, source_depths)
        target_ray = self._camera_world_point(
            view_index, target_row, target_column, target_depth
        )
        return target_ray + torch.einsum("n,ni->i", weights, residuals)

    def _anchor_conditioned_transport_means(
        self,
        source_mean: torch.Tensor,
        source_depth: torch.Tensor,
        source_position: Tuple[int, int],
        target_positions: List[Tuple[int, int]],
        *,
        view_index: int,
    ) -> torch.Tensor:
        """Transport one selected anchor to a batch of assigned positions.

        This is the conditional counterpart of the existing mixture transport:
        ``g_{i|p}`` uses only anchor ``p``'s depth and adaptor residual.  It
        never reads a skipped S2/S3 value and deliberately does not mix another
        anchor into p before the paper assignment weight is applied.
        """
        if source_mean.shape != (3,) or source_depth.numel() != 1:
            raise ValueError("anchor-conditioned transport requires one 3D source")
        if not target_positions:
            return source_mean.new_empty((0, 3))
        if self._camera_directions is None or self._camera_origins is None:
            return source_mean.unsqueeze(0).expand(len(target_positions), -1)
        source_row, source_column = source_position
        source_ray = self._camera_world_point(
            view_index, source_row, source_column, source_depth.reshape(())
        ).to(device=source_mean.device, dtype=source_mean.dtype)
        residual = source_mean - source_ray
        rows = torch.tensor(
            [row for row, _ in target_positions],
            device=self._camera_directions.device,
            dtype=torch.long,
        )
        columns = torch.tensor(
            [column for _, column in target_positions],
            device=self._camera_directions.device,
            dtype=torch.long,
        )
        directions = self._camera_directions[view_index, rows, columns].to(
            device=source_mean.device, dtype=source_mean.dtype
        )
        origin = self._camera_origins[view_index].to(
            device=source_mean.device, dtype=source_mean.dtype
        )
        return (
            origin.unsqueeze(0)
            + directions
            * source_depth.reshape(1, 1).to(
                device=source_mean.device, dtype=source_mean.dtype
            )
            + residual.unsqueeze(0)
        )

    def _recover_adapter_offset(
        self,
        source_mean: torch.Tensor,
        source_depth: torch.Tensor,
        source_position: Tuple[int, int],
        *,
        view_index: int,
    ) -> torch.Tensor | None:
        """Recover one selected anchor's bounded TranSplat image-plane offset."""
        if source_mean.shape != (3,) or source_depth.numel() != 1:
            raise ValueError("adapter-offset recovery requires one 3D source")
        if (
            self.context_extrinsics is None
            or self.context_intrinsics is None
            or self._camera_origins is None
            or self.ray_depth_mode != "euclidean"
            or not 0 <= view_index < self.view_count
        ):
            return None

        source_row, source_column = source_position
        if not 0 <= source_row < self.H or not 0 <= source_column < self.W:
            return None

        device = source_mean.device
        dtype = source_mean.dtype
        if not torch.is_floating_point(source_mean):
            return None
        depth = source_depth.reshape(()).to(device=device, dtype=dtype)
        if not bool(torch.isfinite(source_mean).all()) or not bool(
            torch.isfinite(depth)
        ):
            return None
        tiny = torch.as_tensor(torch.finfo(dtype).eps, device=device, dtype=dtype)
        if bool(depth <= tiny):
            return None

        extrinsic = self.context_extrinsics[0, view_index].to(
            device=device, dtype=dtype
        )
        intrinsic = self.context_intrinsics[0, view_index].to(
            device=device, dtype=dtype
        )
        if not bool(torch.isfinite(extrinsic).all()) or not bool(
            torch.isfinite(intrinsic).all()
        ):
            return None
        try:
            intrinsic_determinant = torch.linalg.det(intrinsic)
        except RuntimeError:
            return None
        if not bool(torch.isfinite(intrinsic_determinant)) or bool(
            intrinsic_determinant.abs() <= tiny
        ):
            return None

        rotation_c2w = extrinsic[:3, :3]
        origin = extrinsic[:3, 3]
        camera_displacement = rotation_c2w.mT @ (source_mean - origin)
        source_distance = camera_displacement.norm()
        if not bool(torch.isfinite(source_distance)) or bool(source_distance <= tiny):
            return None
        depth_tolerance = torch.maximum(
            depth.abs() * 1e-4,
            torch.as_tensor(1e-6, device=device, dtype=dtype),
        )
        if bool((source_distance - depth).abs() > depth_tolerance):
            return None
        camera_direction = camera_displacement / source_distance
        source_homogeneous = intrinsic @ camera_direction
        if not bool(torch.isfinite(source_homogeneous).all()) or bool(
            source_homogeneous[2].abs() <= tiny
        ):
            return None
        source_coordinate = source_homogeneous[:2] / source_homogeneous[2]
        source_centre = torch.tensor(
            ((source_column + 0.5) / self.W, (source_row + 0.5) / self.H),
            device=device,
            dtype=dtype,
        )
        offset = source_coordinate - source_centre
        offset_limit = torch.tensor(
            (0.5 / self.W, 0.5 / self.H), device=device, dtype=dtype
        )
        offset_tolerance = torch.as_tensor(1e-6, device=device, dtype=dtype)
        if not bool(torch.isfinite(offset).all()) or bool(
            (offset.abs() > offset_limit + offset_tolerance).any()
        ):
            return None
        return offset

    def _lift_adapter_offsets(
        self,
        target_positions: List[Tuple[int, int]],
        target_depths: torch.Tensor,
        target_offsets: torch.Tensor,
        *,
        view_index: int,
    ) -> torch.Tensor | None:
        """Lift bounded adapter offsets at target pixels before ray normalization."""
        if not target_positions:
            return target_depths.new_empty((0, 3))
        if (
            self.context_extrinsics is None
            or self.context_intrinsics is None
            or self._camera_origins is None
            or self.ray_depth_mode != "euclidean"
            or not 0 <= view_index < self.view_count
        ):
            return None
        if any(
            not 0 <= row < self.H or not 0 <= column < self.W
            for row, column in target_positions
        ):
            return None
        if not torch.is_tensor(target_depths) or not torch.is_tensor(target_offsets):
            raise ValueError("adapter-offset lift requires tensor depths and offsets")
        target_depths = target_depths.reshape(-1)
        if target_depths.numel() != len(target_positions):
            raise ValueError(
                "adapter-offset lift depth count must match target positions"
            )
        if target_offsets.shape != (len(target_positions), 2):
            raise ValueError("adapter-offset lift offsets must match target positions")
        if not torch.is_floating_point(target_depths):
            return None

        device = target_depths.device
        dtype = target_depths.dtype
        target_offsets = target_offsets.to(device=device, dtype=dtype)
        tiny = torch.as_tensor(torch.finfo(dtype).eps, device=device, dtype=dtype)
        if (
            not bool(torch.isfinite(target_depths).all())
            or not bool(torch.isfinite(target_offsets).all())
            or bool((target_depths <= tiny).any())
        ):
            return None
        offset_limit = torch.tensor(
            (0.5 / self.W, 0.5 / self.H), device=device, dtype=dtype
        )
        offset_tolerance = torch.as_tensor(1e-6, device=device, dtype=dtype)
        if bool((target_offsets.abs() > offset_limit + offset_tolerance).any()):
            return None

        extrinsic = self.context_extrinsics[0, view_index].to(
            device=device, dtype=dtype
        )
        intrinsic = self.context_intrinsics[0, view_index].to(
            device=device, dtype=dtype
        )
        if not bool(torch.isfinite(extrinsic).all()) or not bool(
            torch.isfinite(intrinsic).all()
        ):
            return None
        try:
            intrinsic_determinant = torch.linalg.det(intrinsic)
        except RuntimeError:
            return None
        if not bool(torch.isfinite(intrinsic_determinant)) or bool(
            intrinsic_determinant.abs() <= tiny
        ):
            return None

        rows = torch.tensor(
            [row for row, _ in target_positions], device=device, dtype=dtype
        )
        columns = torch.tensor(
            [column for _, column in target_positions], device=device, dtype=dtype
        )
        target_centres = torch.stack(
            ((columns + 0.5) / self.W, (rows + 0.5) / self.H), dim=-1
        )
        target_homogeneous = torch.cat(
            (
                target_centres + target_offsets,
                torch.ones((len(target_positions), 1), device=device, dtype=dtype),
            ),
            dim=-1,
        )
        try:
            camera_directions = torch.linalg.solve(intrinsic, target_homogeneous.mT).mT
        except RuntimeError:
            return None
        direction_norms = camera_directions.norm(dim=-1, keepdim=True)
        if (
            not bool(torch.isfinite(camera_directions).all())
            or not bool(torch.isfinite(direction_norms).all())
            or bool((direction_norms <= tiny).any())
        ):
            return None
        camera_directions = camera_directions / direction_norms
        world_directions = torch.einsum(
            "ij,nj->ni", extrinsic[:3, :3], camera_directions
        )
        transported = extrinsic[:3, 3].unsqueeze(
            0
        ) + world_directions * target_depths.unsqueeze(1)
        return transported if bool(torch.isfinite(transported).all()) else None

    def _adapter_offset_transport_means(
        self,
        source_mean: torch.Tensor,
        source_depth: torch.Tensor,
        source_position: Tuple[int, int],
        target_positions: List[Tuple[int, int]],
        *,
        view_index: int,
    ) -> torch.Tensor | None:
        """Transport one anchor with TranSplat's image-plane-offset convention.

        The Gaussian adapter first shifts a pixel centre by its predicted
        sub-pixel image-plane offset, then unprojects and normalizes that ray.
        Stage 3 exposes the resulting mean rather than the raw offset, so this
        diagnostic recovers the offset from one *selected* anchor mean/depth
        and the producer context camera.  It applies that same bounded offset
        to each assigned pixel centre before normalizing the target ray.

        This deliberately accepts no target-view data and no skipped S3
        descriptor.  ``None`` is a fail-closed signal: a caller must keep the
        tile Full rather than approximate the adapter with a world-space
        residual when camera geometry, depth convention, or offset bounds do
        not match the upstream adapter contract.
        """
        if source_mean.shape != (3,) or source_depth.numel() != 1:
            raise ValueError("adapter-offset transport requires one 3D source")
        if not target_positions:
            return source_mean.new_empty((0, 3))
        offset = self._recover_adapter_offset(
            source_mean, source_depth, source_position, view_index=view_index
        )
        if offset is None:
            return None
        depth = source_depth.reshape(()).to(
            device=source_mean.device, dtype=source_mean.dtype
        )
        return self._lift_adapter_offsets(
            target_positions,
            depth.expand(len(target_positions)),
            offset.unsqueeze(0).expand(len(target_positions), -1),
            view_index=view_index,
        )

    def _commit_assignment_consensus_plan(self, gaussians_full, plan: Dict) -> None:
        """Commit one already-validated virtual-output plan.

        Planning is deliberately separate from writing so every primitive slot
        can validate from selected S2/S3 inputs before any skipped descriptor
        is touched. This preserves whole-tile fail-closed behavior without
        snapshotting raw skipped S3 attributes.
        """
        output_indices = plan.get("output_indices")
        consensus_means = plan.get("means")
        consensus_covariances = plan.get("covariances")
        consensus_harmonics = plan.get("harmonics")
        consensus_opacities = plan.get("opacities")
        harmonic_shape = plan.get("harmonic_shape")
        opacity_shape = plan.get("opacity_shape")
        anchor_count = plan.get("anchor_count")
        target_count = len(output_indices) if isinstance(output_indices, tuple) else -1
        if (
            target_count < 0
            or not isinstance(anchor_count, int)
            or anchor_count < 1
            or consensus_means.shape != (target_count, 3)
            or consensus_covariances.shape != (target_count, 3, 3)
            or consensus_harmonics.shape[0] != target_count
            or consensus_opacities.shape[0] != target_count
            or not isinstance(harmonic_shape, torch.Size)
            or not isinstance(opacity_shape, torch.Size)
        ):
            raise ValueError("invalid assignment-consensus virtual-output plan")
        for pseudo_index, output_index in enumerate(output_indices):
            gaussians_full.means[0, output_index] = consensus_means[pseudo_index]
            gaussians_full.covariances[0, output_index] = consensus_covariances[
                pseudo_index
            ]
            gaussians_full.harmonics[0, output_index] = consensus_harmonics[
                pseudo_index
            ].reshape(harmonic_shape)
            gaussians_full.opacities[0, output_index] = consensus_opacities[
                pseudo_index
            ].reshape(opacity_shape)
        self.stats["assignment_consensus_pseudo_outputs"] += target_count
        self.stats["assignment_consensus_anchor_pairs"] += target_count * anchor_count
        self.stats["assignment_consensus_offset_recoveries"] += anchor_count
        self.stats["assignment_consensus_target_lifts"] += target_count * (
            anchor_count + 1
        )

    def _assignment_consensus_adapter_pseudo_geometry(
        self,
        selected_means: torch.Tensor,
        selected_covariances: torch.Tensor,
        selected_depths: torch.Tensor,
        selected_positions: List[Tuple[int, int]],
        assignment_matrix: torch.Tensor,
        target_positions: List[Tuple[int, int]],
        *,
        view_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Build virtual skipped geometry from selected anchors only.

        This helper deliberately accepts no full Gaussian/depth tensor.  For a
        skipped position ``i`` and selected anchor ``q`` it evaluates the
        pre-registered consensus descriptor once:

        ``d_i = sum_q r_iq d_q`` and ``o_i = sum_q r_iq o_q``;
        ``m_i = LiftC2W(x_i, d_i, o_i)``;
        ``C_i = sum_q r_iq [C_q + (m_iq-m_i)(m_iq-m_i)^T]``.

        The caller writes the returned virtual outputs directly.  It must not
        feed them into the retained-anchor update loop, which would introduce
        an undeclared second assignment factor.
        """
        if (
            selected_means.ndim != 2
            or selected_means.shape[1] != 3
            or selected_covariances.shape != (selected_means.shape[0], 3, 3)
            or selected_depths.numel() != selected_means.shape[0]
            or len(selected_positions) != selected_means.shape[0]
            or assignment_matrix.shape
            != (len(target_positions), selected_means.shape[0])
            or not torch.is_floating_point(selected_means)
        ):
            return None

        anchor_count = selected_means.shape[0]
        target_count = len(target_positions)
        if anchor_count < 1:
            return None
        if target_count == 0:
            return (
                selected_means.new_empty((0, 3)),
                selected_covariances.new_empty((0, 3, 3)),
            )
        if (
            self.context_extrinsics is None
            or self.context_intrinsics is None
            or self._camera_origins is None
            or self.ray_depth_mode != "euclidean"
            or not 0 <= view_index < self.view_count
        ):
            return None

        device = selected_means.device
        dtype = selected_means.dtype
        selected_depths = selected_depths.reshape(-1).to(device=device, dtype=dtype)
        selected_covariances = selected_covariances.to(device=device, dtype=dtype)
        assignment_matrix = assignment_matrix.to(device=device, dtype=dtype)
        if (
            not bool(torch.isfinite(selected_means).all())
            or not bool(torch.isfinite(selected_covariances).all())
            or not bool(torch.isfinite(selected_depths).all())
            or not bool(torch.isfinite(assignment_matrix).all())
        ):
            return None
        tiny = torch.as_tensor(torch.finfo(dtype).eps, device=device, dtype=dtype)
        if bool((selected_depths <= tiny).any()):
            return None

        # ``paper_assignment_weights`` emits a simplex row.  Do not repair an
        # invalid caller-provided row by renormalising it: fail closed instead.
        simplex_tolerance = torch.as_tensor(1e-5, device=device, dtype=dtype)
        if bool((assignment_matrix < -simplex_tolerance).any()) or bool(
            (assignment_matrix.sum(dim=1) - 1.0).abs().max() > simplex_tolerance
        ):
            return None

        selected_covariances = (selected_covariances + selected_covariances.mT) * 0.5
        try:
            selected_eigenvalues = torch.linalg.eigvalsh(selected_covariances)
        except RuntimeError:
            return None
        if not bool(torch.isfinite(selected_eigenvalues).all()) or bool(
            (selected_eigenvalues < -1e-7).any()
        ):
            return None

        offsets = []
        for anchor_index, source_position in enumerate(selected_positions):
            offset = self._recover_adapter_offset(
                selected_means[anchor_index],
                selected_depths[anchor_index],
                source_position,
                view_index=view_index,
            )
            if offset is None:
                return None
            offsets.append(offset)
        selected_offsets = torch.stack(offsets, dim=0)

        consensus_depths = assignment_matrix @ selected_depths
        consensus_offsets = assignment_matrix @ selected_offsets
        consensus_means = self._lift_adapter_offsets(
            target_positions,
            consensus_depths,
            consensus_offsets,
            view_index=view_index,
        )
        if consensus_means is None:
            return None

        conditional_means = []
        for anchor_index in range(anchor_count):
            lifted = self._lift_adapter_offsets(
                target_positions,
                selected_depths[anchor_index].expand(target_count),
                selected_offsets[anchor_index].unsqueeze(0).expand(target_count, -1),
                view_index=view_index,
            )
            if lifted is None:
                return None
            conditional_means.append(lifted)
        conditional_means = torch.stack(conditional_means, dim=1)
        displacements = conditional_means - consensus_means.unsqueeze(1)
        second_moments = selected_covariances.unsqueeze(0) + torch.einsum(
            "mki,mkj->mkij", displacements, displacements
        )
        consensus_covariances = torch.einsum(
            "mk,mkij->mij", assignment_matrix, second_moments
        )
        consensus_covariances = (consensus_covariances + consensus_covariances.mT) * 0.5
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(consensus_covariances)
        except RuntimeError:
            return None
        if not bool(torch.isfinite(eigenvalues).all()):
            return None
        consensus_covariances = (
            eigenvectors
            @ torch.diag_embed(eigenvalues.clamp_min(1e-8))
            @ eigenvectors.mT
        )
        consensus_covariances = (consensus_covariances + consensus_covariances.mT) * 0.5
        try:
            output_eigenvalues = torch.linalg.eigvalsh(consensus_covariances)
        except RuntimeError:
            return None
        if (
            not bool(torch.isfinite(consensus_means).all())
            or not bool(torch.isfinite(consensus_covariances).all())
            or not bool(torch.isfinite(output_eigenvalues).all())
            or bool((output_eigenvalues < -1e-7).any())
        ):
            return None
        return consensus_means, consensus_covariances

    def _context_projected_footprint_scales(
        self,
        means: torch.Tensor,
        covariances: torch.Tensor,
        *,
        view_index: int,
    ) -> torch.Tensor | None:
        """Return context-camera 2D Gaussian footprint scales.

        Alpha compositing consumes projected 2D Gaussian support, not 3D
        covariance volume. The producer's own context C2W/intrinsics provide
        the local projection Jacobian, so no target-view geometry is needed.
        ``None`` means the tile must remain Full rather than inventing a
        footprint without a valid camera projection.
        """
        if (
            self.context_extrinsics is None
            or self.context_intrinsics is None
            or means.ndim != 2
            or means.shape[1] != 3
            or covariances.shape != (means.shape[0], 3, 3)
            or not 0 <= view_index < self.view_count
        ):
            return None
        dtype = means.dtype
        device = means.device
        extrinsic = self.context_extrinsics[0, view_index].to(
            device=device, dtype=dtype
        )
        intrinsic = self.context_intrinsics[0, view_index].to(
            device=device, dtype=dtype
        )
        rotation_c2w = extrinsic[:3, :3]
        camera_points = torch.einsum(
            "ij,nj->ni", rotation_c2w.mT, means - extrinsic[:3, 3]
        )
        depth = camera_points[:, 2]
        tiny = torch.as_tensor(torch.finfo(dtype).tiny, device=device, dtype=dtype)
        if not bool(torch.isfinite(camera_points).all()) or bool((depth <= tiny).any()):
            return None
        normalized_jacobian = torch.zeros(
            (means.shape[0], 2, 3), device=device, dtype=dtype
        )
        normalized_jacobian[:, 0, 0] = depth.reciprocal()
        normalized_jacobian[:, 1, 1] = depth.reciprocal()
        normalized_jacobian[:, 0, 2] = -camera_points[:, 0] / depth.square()
        normalized_jacobian[:, 1, 2] = -camera_points[:, 1] / depth.square()
        image_jacobian = torch.einsum(
            "ij,njk->nik", intrinsic[:2, :2], normalized_jacobian
        )
        camera_covariances = torch.einsum(
            "ij,njk,kl->nil", rotation_c2w.mT, covariances, rotation_c2w
        )
        projected_covariances = torch.einsum(
            "nij,njk,nlk->nil",
            image_jacobian,
            camera_covariances,
            image_jacobian,
        )
        projected_covariances = (projected_covariances + projected_covariances.mT) * 0.5
        eigenvalues, eigenvectors = torch.linalg.eigh(projected_covariances)
        if not bool(torch.isfinite(eigenvalues).all()) or bool(
            (eigenvalues < -1e-7).any()
        ):
            return None
        projected_covariances = (
            eigenvectors
            @ torch.diag_embed(eigenvalues.clamp_min(tiny))
            @ eigenvectors.mT
        )
        determinants = torch.linalg.det(projected_covariances)
        if not bool(torch.isfinite(determinants).all()) or bool(
            (determinants <= 0.0).any()
        ):
            return None
        return torch.sqrt(determinants)

    def _context_projected_moments(
        self,
        means: torch.Tensor,
        covariances: torch.Tensor,
        *,
        view_index: int,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None
    ):
        """Project world-space Gaussian moments into one producer camera.

        The dense oracle uses this only after a full Stage-3 pass.  Keeping the
        projection here makes the oracle's coverage calculation share the
        same C2W/intrinsics convention as the runtime diagnostic rather than
        approximating image footprint with 3D covariance volume.
        """
        if (
            self.context_extrinsics is None
            or self.context_intrinsics is None
            or means.ndim != 2
            or means.shape[1] != 3
            or covariances.shape != (means.shape[0], 3, 3)
            or not 0 <= view_index < self.view_count
        ):
            return None
        dtype = means.dtype
        device = means.device
        extrinsic = self.context_extrinsics[0, view_index].to(
            device=device, dtype=dtype
        )
        intrinsic = self.context_intrinsics[0, view_index].to(
            device=device, dtype=dtype
        )
        rotation_c2w = extrinsic[:3, :3]
        camera_points = torch.einsum(
            "ij,nj->ni", rotation_c2w.mT, means - extrinsic[:3, 3]
        )
        depth = camera_points[:, 2]
        tiny = torch.as_tensor(torch.finfo(dtype).tiny, device=device, dtype=dtype)
        if not bool(torch.isfinite(camera_points).all()) or bool((depth <= tiny).any()):
            return None

        homogeneous = torch.einsum("ij,nj->ni", intrinsic, camera_points)
        homogeneous_depth = homogeneous[:, 2]
        if not bool(torch.isfinite(homogeneous).all()) or bool(
            (homogeneous_depth.abs() <= tiny).any()
        ):
            return None
        centers = homogeneous[:, :2] / homogeneous_depth.unsqueeze(1)

        normalized_jacobian = torch.zeros(
            (means.shape[0], 2, 3), device=device, dtype=dtype
        )
        normalized_jacobian[:, 0, 0] = depth.reciprocal()
        normalized_jacobian[:, 1, 1] = depth.reciprocal()
        normalized_jacobian[:, 0, 2] = -camera_points[:, 0] / depth.square()
        normalized_jacobian[:, 1, 2] = -camera_points[:, 1] / depth.square()
        image_jacobian = torch.einsum(
            "ij,njk->nik", intrinsic[:2, :2], normalized_jacobian
        )
        camera_covariances = torch.einsum(
            "ij,njk,kl->nil", rotation_c2w.mT, covariances, rotation_c2w
        )
        projected_covariances = torch.einsum(
            "nij,njk,nlk->nil",
            image_jacobian,
            camera_covariances,
            image_jacobian,
        )
        projected_covariances = (projected_covariances + projected_covariances.mT) * 0.5
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(projected_covariances)
        except RuntimeError:
            return None
        if (
            not bool(torch.isfinite(centers).all())
            or not bool(torch.isfinite(eigenvalues).all())
            or bool((eigenvalues < -1e-7).any())
        ):
            return None
        projected_covariances = (
            eigenvectors
            @ torch.diag_embed(eigenvalues.clamp_min(tiny))
            @ eigenvectors.mT
        )
        return (
            centers,
            projected_covariances,
            camera_points,
            image_jacobian,
            rotation_c2w,
        )

    @staticmethod
    def _interpolate_intrinsic_covariance(
        source_covariances: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Interpolate only local Gaussian shape before the final moment merge.

        A selected-anchor-derived pseudo Gaussian already has its own C2W
        mean. Adding anchor-to-pseudo mean dispersion here and adding
        pseudo-to-anchor dispersion again in the final second-moment update
        counts the same spatial support twice. Keep the pseudo covariance
        intrinsic; the outer-product term belongs exactly once in the
        receiving anchor's first/second-moment match.
        """
        if (
            source_covariances.ndim != 3
            or source_covariances.shape[1:] != (3, 3)
            or weights.numel() != source_covariances.shape[0]
        ):
            raise ValueError("intrinsic covariance interpolation inputs must align")
        weights = weights.reshape(-1).to(
            device=source_covariances.device, dtype=source_covariances.dtype
        )
        covariance = torch.einsum("n,nij->ij", weights, source_covariances)
        return (covariance + covariance.mT) * 0.5

    def _multicontext_tangent_candidate(
        self,
        *,
        output_mean: torch.Tensor,
        local_covariance: torch.Tensor,
        contributor_means: torch.Tensor,
        contributor_covariances: torch.Tensor,
        contributor_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Return a context-only covariance candidate or the exact local value.

        All contributors are already selected-anchor descriptors or local
        pseudo descriptors derived from them. The helper cannot receive target
        cameras, target RGB, or skipped Stage-3 attributes. A failed fit is a
        covariance-only fallback to the current materialization, not a route
        change or a new Full-policy decision.
        """
        if not self.multicontext_tangent_enabled:
            return local_covariance
        self.stats["multicontext_tangent_attempts"] += 1
        self.stats["multicontext_tangent_contributor_count"] += int(
            contributor_weights.numel()
        )
        if self.context_extrinsics is None or self.context_intrinsics is None:
            reason = "missing-context-geometry"
            self.stats["multicontext_tangent_local_fallbacks"] += 1
            reasons = self.stats["multicontext_tangent_fallback_reasons"]
            reasons[reason] = reasons.get(reason, 0) + 1
            return local_covariance

        from saes.multicontext_tangent import multicontext_tangent_covariance

        context_extrinsics = self.context_extrinsics[0].to(
            device=local_covariance.device, dtype=local_covariance.dtype
        )
        context_intrinsics = self.context_intrinsics[0].to(
            device=local_covariance.device, dtype=local_covariance.dtype
        )
        self.stats["multicontext_tangent_context_camera_reads"] += int(
            context_extrinsics.shape[0]
        )
        self.stats["multicontext_tangent_constraint_count"] += int(
            context_extrinsics.shape[0] * 3
        )
        result = multicontext_tangent_covariance(
            output_mean=output_mean,
            local_covariance=local_covariance,
            contributor_means=contributor_means,
            contributor_covariances=contributor_covariances,
            contributor_weights=contributor_weights,
            context_extrinsics=context_extrinsics,
            context_intrinsics=context_intrinsics,
        )
        if result.used_multicontext_fit:
            self.stats["multicontext_tangent_accepted"] += 1
            self.stats["multicontext_tangent_residual_max"] = max(
                self.stats["multicontext_tangent_residual_max"],
                float(result.residual_max or 0.0),
            )
            return result.covariance

        self.stats["multicontext_tangent_local_fallbacks"] += 1
        reasons = self.stats["multicontext_tangent_fallback_reasons"]
        reasons[result.reason] = reasons.get(result.reason, 0) + 1
        return local_covariance

    def _same_budget_dense_oracle_plan(
        self,
        gaussians_full,
        *,
        probe_indices: List[int],
        non_probe_items: List[Tuple[Tuple[int, int], int]],
        retained_positions: List[Tuple[int, int]],
        assignment_matrix: torch.Tensor,
        tile_y: int,
        tile_x: int,
        view_index: int,
    ) -> Dict | None:
        """Build a post-hoc K/2K projected-moment reduction plan.

        This is deliberately an oracle, not a runtime implementation: it may
        consume full Stage-3 attributes at every tile position after the dense
        encoder has completed.  The returned plan nevertheless writes only the
        preselected L0 K or L1 2K output slots and removes every other slot.
        It contains no target RGB or target-view input.
        """
        count = len(probe_indices)
        self._same_budget_dense_oracle_last_failure = None

        def reject(reason: str) -> None:
            self._same_budget_dense_oracle_last_failure = reason
            return None

        source_indices = list(probe_indices) + [index for _, index in non_probe_items]
        source_positions = list(retained_positions) + [
            position for position, _ in non_probe_items
        ]
        source_count = len(source_indices)
        if (
            count < 1
            or len(retained_positions) != count
            or len(source_positions) != source_count
            or assignment_matrix.shape != (len(non_probe_items), count)
        ):
            return reject("input-contract")

        means = gaussians_full.means[0, source_indices].clone()
        covariances = gaussians_full.covariances[0, source_indices].clone()
        harmonics = gaussians_full.harmonics[0, source_indices].clone()
        opacities = gaussians_full.opacities[0, source_indices].clone()
        if (
            not bool(torch.isfinite(means).all())
            or not bool(torch.isfinite(covariances).all())
            or not bool(torch.isfinite(harmonics).all())
            or not bool(torch.isfinite(opacities).all())
        ):
            return reject("nonfinite-dense-descriptor")
        flat_opacities = opacities.reshape(source_count, -1)
        if flat_opacities.shape[1] != 1 or bool(
            ((flat_opacities < 0.0) | (flat_opacities >= 1.0)).any()
        ):
            return reject("opacity-domain")
        covariances = (covariances + covariances.mT) * 0.5
        try:
            covariance_eigenvalues = torch.linalg.eigvalsh(covariances)
        except RuntimeError:
            return reject("dense-covariance-eigendecomposition")
        if not bool(torch.isfinite(covariance_eigenvalues).all()) or bool(
            (covariance_eigenvalues < -1e-7).any()
        ):
            return reject("dense-covariance-psd")

        projected = self._context_projected_moments(
            means, covariances, view_index=view_index
        )
        if projected is None:
            return reject("producer-projection-geometry")
        centers, projected_covariances, camera_points, _, rotation_c2w = projected
        dtype = means.dtype
        device = means.device
        tiny = torch.as_tensor(torch.finfo(dtype).tiny, device=device, dtype=dtype)
        try:
            projected_determinants = torch.linalg.det(projected_covariances)
        except RuntimeError:
            return reject("projected-footprint-determinant")
        if not bool(torch.isfinite(projected_determinants).all()) or bool(
            (projected_determinants <= tiny).any()
        ):
            return reject("projected-footprint-determinant")
        source_areas = torch.sqrt(projected_determinants)
        source_tau = -torch.log1p(-flat_opacities[:, 0])
        source_mass = source_tau * source_areas
        if not bool(torch.isfinite(source_mass).all()) or bool(
            (source_mass < 0.0).any()
        ):
            return reject("source-optical-mass")
        if assignment_matrix.numel() and (
            not bool(torch.isfinite(assignment_matrix).all())
            or bool((assignment_matrix < 0.0).any())
            or bool((assignment_matrix.sum(dim=1) - 1.0).abs().max() > 1e-5)
        ):
            return reject("assignment-simplex")

        # Each selected output keeps its own dense descriptor with unit
        # ownership.  Every skipped descriptor contributes its optical mass
        # exactly once across the existing bilateral assignment simplex.
        ownership = torch.cat(
            (
                torch.eye(count, device=device, dtype=dtype),
                assignment_matrix.mT,
            ),
            dim=1,
        )
        selected_opacities = flat_opacities[:count, 0]
        selected_harmonics = harmonics[:count].reshape(count, -1)
        alpha_min = selected_opacities.amin()
        alpha_max = selected_opacities.amax()
        tau_max = -torch.log1p(-alpha_max)
        if not bool(torch.isfinite(tau_max)) or bool(tau_max <= tiny):
            return reject("selected-opacity-range")
        intrinsic = self.context_intrinsics[0, view_index].to(
            device=device, dtype=dtype
        )
        extrinsic = self.context_extrinsics[0, view_index].to(
            device=device, dtype=dtype
        )
        output_means = []
        output_covariances = []
        output_harmonics = []
        output_opacities = []
        mass_error_max = 0.0
        projected_moment_error_max = 0.0
        footprint_expansion_max = 1.0

        for anchor in range(count):
            mass_weights = ownership[anchor] * source_mass
            total_mass = mass_weights.sum()
            if not bool(torch.isfinite(total_mass)) or bool(total_mass <= tiny):
                return reject("zero-cluster-optical-mass")
            normalized = mass_weights / total_mass
            center = torch.einsum("n,ni->i", normalized, centers)
            centered = centers - center.unsqueeze(0)
            target_projected_covariance = torch.einsum(
                "n,nij->ij",
                normalized,
                projected_covariances + torch.einsum("ni,nj->nij", centered, centered),
            )
            target_projected_covariance = (
                target_projected_covariance + target_projected_covariance.mT
            ) * 0.5
            try:
                target_eigenvalues, target_eigenvectors = torch.linalg.eigh(
                    target_projected_covariance
                )
            except RuntimeError:
                return reject("projected-second-moment-eigendecomposition")
            if not bool(torch.isfinite(target_eigenvalues).all()) or bool(
                (target_eigenvalues < -1e-7).any()
            ):
                return reject("projected-second-moment-psd")
            target_projected_covariance = (
                target_eigenvectors
                @ torch.diag(target_eigenvalues.clamp_min(tiny))
                @ target_eigenvectors.mT
            )
            target_area = torch.sqrt(torch.linalg.det(target_projected_covariance))
            if not bool(torch.isfinite(target_area)) or bool(target_area <= tiny):
                return reject("projected-second-moment-area")
            raw_tau = total_mass / target_area
            raw_alpha = -torch.expm1(-raw_tau)
            if not bool(torch.isfinite(raw_alpha)) or bool(raw_alpha < 0.0):
                return reject("raw-opacity")

            # The range constraint remains anchored to actual selected outputs.
            # If the cluster needs more footprint to avoid exceeding that
            # opacity, expand tangential covariance rather than inventing alpha.
            footprint_expansion = torch.ones((), device=device, dtype=dtype)
            if bool(raw_alpha > alpha_max):
                required_area = total_mass / tau_max
                footprint_expansion = required_area / target_area
                if not bool(torch.isfinite(footprint_expansion)) or bool(
                    footprint_expansion < 1.0
                ):
                    return reject("required-footprint-expansion")
                target_projected_covariance = (
                    target_projected_covariance * footprint_expansion
                )
                target_area = target_area * footprint_expansion
            output_tau = total_mass / target_area
            output_alpha = -torch.expm1(-output_tau)
            if (
                not bool(torch.isfinite(output_alpha))
                or bool(output_alpha < alpha_min - 1e-5)
                or bool(output_alpha > alpha_max + 1e-5)
            ):
                return reject("selected-opacity-range")

            # Reconstruct a 3D covariance whose local image-plane projection
            # matches the weighted 2D moment.  The null-space component keeps
            # the full-source depth variance without perturbing that projection.
            depth = torch.einsum("n,n->", normalized, camera_points[:, 2])
            homogeneous_center = torch.cat(
                (center, torch.ones(1, device=device, dtype=dtype))
            )
            try:
                camera_ray = torch.linalg.solve(intrinsic, homogeneous_center)
            except RuntimeError:
                return reject("output-camera-ray")
            if (
                not bool(torch.isfinite(camera_ray).all())
                or bool(camera_ray[2].abs() <= tiny)
                or not bool(torch.isfinite(depth))
                or bool(depth <= tiny)
            ):
                return reject("output-camera-geometry")
            camera_mean = camera_ray / camera_ray[2] * depth
            normalized_jacobian = torch.zeros((2, 3), device=device, dtype=dtype)
            normalized_jacobian[0, 0] = depth.reciprocal()
            normalized_jacobian[1, 1] = depth.reciprocal()
            normalized_jacobian[0, 2] = -camera_mean[0] / depth.square()
            normalized_jacobian[1, 2] = -camera_mean[1] / depth.square()
            output_jacobian = intrinsic[:2, :2] @ normalized_jacobian
            try:
                singular_values = torch.linalg.svdvals(output_jacobian)
            except RuntimeError:
                return reject("output-projection-svd")
            if (
                singular_values.numel() != 2
                or not bool(torch.isfinite(singular_values).all())
                or bool(singular_values[-1] <= tiny)
            ):
                return reject("output-projection-condition")
            output_pseudoinverse = torch.linalg.pinv(output_jacobian)
            ray_direction = camera_mean / camera_mean.norm().clamp_min(tiny)
            depth_variance = torch.einsum(
                "n,n->", normalized, (camera_points[:, 2] - depth).square()
            )
            camera_covariance = (
                output_pseudoinverse
                @ target_projected_covariance
                @ output_pseudoinverse.mT
                + depth_variance * torch.outer(ray_direction, ray_direction)
            )
            camera_covariance = (camera_covariance + camera_covariance.mT) * 0.5
            try:
                output_eigenvalues, output_eigenvectors = torch.linalg.eigh(
                    camera_covariance
                )
            except RuntimeError:
                return reject("output-covariance-eigendecomposition")
            if not bool(torch.isfinite(output_eigenvalues).all()) or bool(
                (output_eigenvalues < -1e-7).any()
            ):
                return reject("output-covariance-psd")
            camera_covariance = (
                output_eigenvectors
                @ torch.diag(output_eigenvalues.clamp_min(tiny))
                @ output_eigenvectors.mT
            )
            observed_projected_covariance = (
                output_jacobian @ camera_covariance @ output_jacobian.mT
            )
            observed_projected_covariance = (
                observed_projected_covariance + observed_projected_covariance.mT
            ) * 0.5
            projected_error = float(
                (observed_projected_covariance - target_projected_covariance)
                .abs()
                .max()
                .item()
            )
            observed_area = torch.sqrt(torch.linalg.det(observed_projected_covariance))
            observed_mass = output_tau * observed_area
            mass_error = float((observed_mass - total_mass).abs().item())
            mass_tolerance = float((1e-6 + 1e-5 * total_mass.abs()).item())
            if (
                not bool(torch.isfinite(observed_area))
                or bool(observed_area <= tiny)
                or mass_error > mass_tolerance
                or projected_error > 1e-3
            ):
                return reject("construction-consistency")

            flat_harmonic = torch.einsum(
                "n,nk->k", normalized, harmonics.reshape(source_count, -1)
            )
            flat_harmonic = torch.maximum(
                torch.minimum(flat_harmonic, selected_harmonics.amax(dim=0)),
                selected_harmonics.amin(dim=0),
            )
            world_mean = rotation_c2w @ camera_mean + extrinsic[:3, 3]
            world_covariance = rotation_c2w @ camera_covariance @ rotation_c2w.mT
            world_covariance = (world_covariance + world_covariance.mT) * 0.5
            output_means.append(world_mean)
            output_covariances.append(world_covariance)
            output_harmonics.append(flat_harmonic.reshape_as(harmonics[0]))
            output_opacities.append(output_alpha.reshape_as(opacities[0]))
            mass_error_max = max(mass_error_max, mass_error)
            projected_moment_error_max = max(
                projected_moment_error_max, projected_error
            )
            footprint_expansion_max = max(
                footprint_expansion_max, float(footprint_expansion.item())
            )

        output_mass = sum(
            float((ownership[index] * source_mass).sum().item())
            for index in range(count)
        )
        source_mass_total = float(source_mass.sum().item())
        mass_error_max = max(mass_error_max, abs(output_mass - source_mass_total))
        return {
            "output_indices": tuple(probe_indices),
            "zero_indices": tuple(index for _, index in non_probe_items),
            "means": torch.stack(output_means),
            "covariances": torch.stack(output_covariances),
            "harmonics": torch.stack(output_harmonics),
            "opacities": torch.stack(output_opacities),
            "output_gaussians": count,
            "mass_error_max": mass_error_max,
            "projected_moment_error_max": projected_moment_error_max,
            "required_footprint_expansion_max": footprint_expansion_max,
        }

    def _commit_same_budget_dense_oracle_plan(self, gaussians_full, plan: Dict) -> None:
        """Commit a validated oracle plan after every tile slot has succeeded."""
        output_indices = list(plan["output_indices"])
        gaussians_full.means[0, output_indices] = plan["means"]
        gaussians_full.covariances[0, output_indices] = plan["covariances"]
        gaussians_full.harmonics[0, output_indices] = plan["harmonics"]
        gaussians_full.opacities[0, output_indices] = plan["opacities"]
        self.stats["same_budget_dense_oracle_output_gaussians"] += plan[
            "output_gaussians"
        ]
        self.stats["same_budget_dense_oracle_mass_construction_error_max"] = max(
            self.stats["same_budget_dense_oracle_mass_construction_error_max"],
            plan["mass_error_max"],
        )
        self.stats[
            "same_budget_dense_oracle_projected_moment_construction_error_max"
        ] = max(
            self.stats[
                "same_budget_dense_oracle_projected_moment_construction_error_max"
            ],
            plan["projected_moment_error_max"],
        )
        self.stats["same_budget_dense_oracle_required_footprint_expansion_max"] = max(
            self.stats["same_budget_dense_oracle_required_footprint_expansion_max"],
            plan["required_footprint_expansion_max"],
        )

    def _assignment_feature_variance(self, decision_statistic: float) -> float:
        """Convert the decision statistic to the paper kernel's variance unit."""
        if math.isinf(decision_statistic) and decision_statistic > 0.0:
            return decision_statistic
        if not math.isfinite(decision_statistic) or decision_statistic < 0.0:
            raise ValueError(
                "feature decision statistic must be finite and nonnegative"
            )
        if self.decision_semantics in {
            "probe-normalized-std-first-hit",
            PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS,
        }:
            return float(decision_statistic) ** 2
        return float(decision_statistic)

    @staticmethod
    def inverse_depth_candidate_coordinate(
        depths: torch.Tensor,
        *,
        near: torch.Tensor,
        far: torch.Tensor,
    ) -> torch.Tensor:
        """Map metric depths to the S2 inverse-depth candidate coordinate.

        This uses only the per-context-view near/far bounds already consumed
        by S2. It makes the unit of the paper's L1 probe-depth statistic
        explicit without adding a routing signal or changing its threshold.
        """
        if depths.dim() not in (4, 5) or depths.shape[:2] != near.shape:
            raise ValueError(
                "depths and near bounds must share [batch, view] dimensions"
            )
        if far.shape != near.shape:
            raise ValueError("near and far bounds must have identical shapes")
        if (
            not torch.isfinite(depths).all()
            or not torch.isfinite(near).all()
            or not torch.isfinite(far).all()
        ):
            raise ValueError("depth values and bounds must be finite")
        if (depths <= 0).any() or (near <= 0).any() or (far <= near).any():
            raise ValueError("inverse-depth normalization requires 0 < near < far")
        trailing = (1,) * (depths.dim() - 2)
        near_inverse = near.to(depths).reciprocal().reshape(*near.shape, *trailing)
        far_inverse = far.to(depths).reciprocal().reshape(*far.shape, *trailing)
        return (depths.reciprocal() - far_inverse) / (near_inverse - far_inverse)

    # ------------------------------------------------------------------ #
    # Feature / depth classification helpers                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def classify_tiles_by_features(
        features,
        h: int,
        w: int,
        tile_size: int,
        threshold: float = 0.02,
        per_view: bool = False,
        statistic: str = "current-channel-std",
    ) -> Tuple[Dict[Tuple[int, int], float], "torch.Tensor"]:
        """
        Classify tiles by feature variance from S1 feature maps.

        Returns:
            tile_variances: dict (th, tw) -> feature_variance (float)
            feat_norm:      normalised upsampled features [C, H, W]
        """
        if features.dim() == 5:
            feat = features[0]
        elif features.dim() == 4:
            feat = features
        else:
            return {}, None

        if not per_view:
            feat = feat.mean(dim=0, keepdim=True)
        feat_up = F.interpolate(
            feat, size=(h, w), mode="bilinear", align_corners=False
        )  # [V, C, H, W] or [1, C, H, W]
        feat_norm = feat_up / (feat_up.norm(dim=1, keepdim=True) + 1e-8)

        tiles_h = h // tile_size
        tiles_w = w // tile_size
        if statistic not in (
            "current-channel-std",
            "raw-probe-vector-variance",
            "raw-probe-mean-channel-variance",
            "normalized-probe-total-variance",
            "normalized-probe-vector-standard-deviation",
        ):
            raise ValueError(f"unsupported feature statistic: {statistic}")
        statistic_features = (
            feat_up
            if statistic
            in {
                "raw-probe-vector-variance",
                "raw-probe-mean-channel-variance",
            }
            else feat_norm
        )
        tiled = (
            statistic_features[:, :, : tiles_h * tile_size, : tiles_w * tile_size]
            .unfold(2, tile_size, tile_size)
            .unfold(3, tile_size, tile_size)
            .contiguous()
        )
        if statistic in (
            "raw-probe-vector-variance",
            "raw-probe-mean-channel-variance",
            "normalized-probe-total-variance",
            "normalized-probe-vector-standard-deviation",
        ):
            probes = ProgressiveSAES.compute_probe_positions(tile_size)
            probe_vectors = torch.stack(
                [tiled[..., local_y, local_x] for local_y, local_x in probes],
                dim=-1,
            )
            probe_mean = probe_vectors.mean(dim=-1, keepdim=True)
            squared_deviation = (probe_vectors - probe_mean).square()
            if statistic == "raw-probe-mean-channel-variance":
                # Sum-of-channel variance makes tau_f depend on the model's
                # feature width. The paper's scalar variance is the mean
                # component variance across the probe vectors.
                values = squared_deviation.mean(dim=(1, -1)).detach().cpu()
            else:
                values = squared_deviation.sum(dim=1).mean(dim=-1)
                if statistic == "normalized-probe-vector-standard-deviation":
                    values = values.sqrt()
                values = values.detach().cpu()
        else:
            values = (
                tiled.reshape(tiled.shape[0], tiled.shape[1], tiles_h, tiles_w, -1)
                .std(dim=-1)
                .mean(dim=1)
                .detach()
                .cpu()
            )
        if per_view:
            tile_variances = {
                (view, th, tw): float(values[view, th, tw])
                for view in range(values.shape[0])
                for th in range(tiles_h)
                for tw in range(tiles_w)
            }
            return tile_variances, feat_norm
        tile_variances = {
            (th, tw): float(values[0, th, tw])
            for th in range(tiles_h)
            for tw in range(tiles_w)
        }
        return tile_variances, feat_norm[0]

    @staticmethod
    def check_depth_uniformity(
        depths,
        th: int,
        tw: int,
        tile_size: int,
        h: int,
        w: int,
        threshold: float = 0.03,
        probe_positions: List[Tuple[int, int]] = None,
        view_index: int = 0,
        primitive_slot: int = 0,
        relative: bool = True,
    ) -> bool:
        """
        Check if probe pixel depths within a tile are uniform.

        probe_positions: list of (local_row, local_col) inside the tile.
        Falls back to 4-corner positions if not supplied.
        """
        if probe_positions is None:
            probe_positions = [(0, 0), (0, 3), (3, 0), (3, 3)]

        tile_y, tile_x = th * tile_size, tw * tile_size
        probe_depths = []

        for ly, lx in probe_positions:
            gy, gx = tile_y + ly, tile_x + lx
            if gy >= h or gx >= w:
                return False
            if depths is None:
                return False
            if depths.dim() == 5:
                pixel_idx = gy * w + gx
                values = depths[0, view_index, pixel_idx].reshape(-1)
                if primitive_slot >= values.numel():
                    return False
                d = values[primitive_slot].item()
            elif depths.dim() == 4:
                d = depths[0, view_index, gy, gx].item()
            else:
                return False
            probe_depths.append(d)

        if len(probe_depths) < 2:
            return False

        d_mean = sum(probe_depths) / len(probe_depths)
        d_std = (
            sum((d - d_mean) ** 2 for d in probe_depths) / len(probe_depths)
        ) ** 0.5
        if not relative:
            return d_std < threshold
        ordered = sorted(probe_depths)
        middle = len(ordered) // 2
        median = (
            ordered[middle]
            if len(ordered) % 2
            else 0.5 * (ordered[middle - 1] + ordered[middle])
        )
        scale = abs(median)
        if scale < 1e-8:
            return d_std < threshold
        normalized = [depth / scale for depth in probe_depths]
        normalized_mean = sum(normalized) / len(normalized)
        normalized_std = (
            sum((depth - normalized_mean) ** 2 for depth in normalized)
            / len(normalized)
        ) ** 0.5
        return normalized_std < threshold

    @staticmethod
    def compute_tile_similarity(
        gaussians_full,
        probe_indices: List[int],
        include_position: bool = True,
    ) -> float:
        """
        Compute appearance/geometry similarity among probe Gaussians.

        Weighted metric:
          With position (L2):  cov(0.30) + SH(0.30) + opacity(0.15) + pos(0.25)
          Without position:    cov(0.40) + SH(0.40) + opacity(0.20)
        """
        n = len(probe_indices)
        if n < 2:
            return 1.0

        total_sim = 0.0
        count = 0

        means = gaussians_full.means[0]
        covs = gaussians_full.covariances[0]
        harmo = gaussians_full.harmonics[0]
        opacs = gaussians_full.opacities[0]

        if include_position:
            probe_means = means[probe_indices]
            pos_std = probe_means.std(dim=0).mean().item()
            pos_range = probe_means.abs().max().item() + 1e-8
            pos_sim = 1.0 - min(pos_std / pos_range, 1.0)

        for i in range(n):
            for j in range(i + 1, n):
                idx_i, idx_j = probe_indices[i], probe_indices[j]

                cov1 = covs[idx_i].flatten()
                cov2 = covs[idx_j].flatten()
                cov_sim = (
                    F.cosine_similarity(cov1.unsqueeze(0), cov2.unsqueeze(0)).item() + 1
                ) / 2

                sh1 = harmo[idx_i].flatten()
                sh2 = harmo[idx_j].flatten()
                sh_sim = (
                    F.cosine_similarity(sh1.unsqueeze(0), sh2.unsqueeze(0)).item() + 1
                ) / 2

                op1 = (
                    opacs[idx_i].item()
                    if opacs[idx_i].dim() == 0
                    else opacs[idx_i].squeeze().item()
                )
                op2 = (
                    opacs[idx_j].item()
                    if opacs[idx_j].dim() == 0
                    else opacs[idx_j].squeeze().item()
                )
                opacity_sim = 1.0 - min(abs(op1 - op2), 1.0)

                if include_position:
                    pair_sim = (
                        0.3 * cov_sim
                        + 0.3 * sh_sim
                        + 0.15 * opacity_sim
                        + 0.25 * pos_sim
                    )
                else:
                    pair_sim = 0.4 * cov_sim + 0.4 * sh_sim + 0.2 * opacity_sim
                total_sim += pair_sim
                count += 1

        return total_sim / count if count > 0 else 1.0

    def probe_cross_check_error(
        self,
        gaussians_full,
        probe_indices: List[int],
    ) -> float:
        """
        Return the ASIC-implementable leave-one-out probe prediction error.

        For each probe, predict it from the average of the other probes and
        return the maximum covariance or harmonic cosine error.
        """
        K = len(probe_indices)
        if K < 2:
            return float("inf")

        # Keep the selected-only boundary observable: indexing ``[0]`` first
        # would materialize the complete S3 tensor for wrapper tensors, even
        # though this check needs only the declared primary anchors.
        harmo = gaussians_full.harmonics[0, probe_indices]
        covs = gaussians_full.covariances[0, probe_indices]

        max_err = 0.0
        for leave_out in range(K):
            others = [j for j in range(K) if j != leave_out]

            pred_h = sum(harmo[o] for o in others) / len(others)
            actual_h = harmo[leave_out]
            h_flat_pred = pred_h.flatten()
            h_flat_actual = actual_h.flatten()
            if h_flat_actual.norm() > 1e-8 and h_flat_pred.norm() > 1e-8:
                h_sim = F.cosine_similarity(
                    h_flat_pred.unsqueeze(0), h_flat_actual.unsqueeze(0)
                ).item()
                max_err = max(max_err, 1.0 - max(h_sim, 0.0))

            pred_c = sum(covs[o] for o in others) / len(others)
            actual_c = covs[leave_out]
            c_flat_pred = pred_c.flatten()
            c_flat_actual = actual_c.flatten()
            if c_flat_actual.norm() > 1e-8 and c_flat_pred.norm() > 1e-8:
                c_sim = F.cosine_similarity(
                    c_flat_pred.unsqueeze(0), c_flat_actual.unsqueeze(0)
                ).item()
                max_err = max(max_err, 1.0 - max(c_sim, 0.0))

        return max_err

    def probe_cross_check(
        self,
        gaussians_full,
        probe_indices: List[int],
    ) -> bool:
        """Return whether the continuous probe error passes the configured gate."""
        return (
            self.probe_cross_check_error(gaussians_full, probe_indices)
            <= self.cross_check_threshold
        )

    def probe_cross_check_validity(
        self,
        gaussians_full,
        probe_indices: List[int],
        *,
        level: str,
    ) -> Dict:
        """Validate a candidate from its already selected primary anchors.

        This is deliberately separate from S1/S2 routing.  It is evaluated
        only after the ordinary anchor-attribute guard accepts a candidate,
        and it has no access to a skipped S3 descriptor.  An invalid or
        non-finite selected anchor is a fail-closed Full decision.
        """
        if level not in {"L0", "L1"}:
            raise ValueError("probe cross-check level must be L0 or L1")
        if (
            not isinstance(probe_indices, list)
            or len(probe_indices) != len(self.probe_positions)
            or len(probe_indices) < 2
            or len(probe_indices) != len(set(probe_indices))
            or any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in probe_indices
            )
        ):
            raise ValueError("probe cross-check requires the unique primary anchors")
        count = gaussians_full.means.shape[1]
        if any(index < 0 or index >= count for index in probe_indices):
            raise ValueError("probe cross-check anchor index is out of range")

        covariances = gaussians_full.covariances[0, probe_indices]
        harmonics = gaussians_full.harmonics[0, probe_indices]
        finite = bool(torch.isfinite(covariances).all() and torch.isfinite(harmonics).all())
        error = self.probe_cross_check_error(gaussians_full, probe_indices)
        error_is_finite = math.isfinite(error)
        return {
            "level": level,
            "anchor_count": len(probe_indices),
            "error": float(error) if error_is_finite else None,
            "threshold": self.cross_check_threshold,
            "passed": bool(finite and error_is_finite and error <= self.cross_check_threshold),
            "nonprobe_s3_attribute_reads": 0,
        }

    def probe_materialization_validity(
        self,
        gaussians_full,
        anchor_indices: List[int],
        *,
        level: str,
    ) -> Dict:
        """Validate a retained-anchor materialization without reading S3 skips.

        The existing paper-local consistency checks are applied only to the
        native attributes that the selected L0/L1 anchors would execute. The
        fixed bounds are shared with the execution identity. The result is a
        fail-closed Control decision. It neither changes feature/depth routing
        nor accesses a non-anchor Gaussian descriptor.
        """
        if level not in {"L0", "L1"}:
            raise ValueError("materialization guard level must be L0 or L1")
        if (
            not isinstance(anchor_indices, list)
            or len(anchor_indices) < 2
            or any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in anchor_indices
            )
            or len(anchor_indices) != len(set(anchor_indices))
        ):
            raise ValueError("materialization guard requires unique anchor indices")
        count = gaussians_full.means.shape[1]
        if any(index < 0 or index >= count for index in anchor_indices):
            raise ValueError("materialization guard anchor index is out of range")

        # Index every source tensor once with the declared anchor list. No
        # non-probe index is formed here, which makes poisoning skipped S3
        # descriptors observationally irrelevant to this Control decision.
        covariances = gaussians_full.covariances[0, anchor_indices].reshape(
            len(anchor_indices), -1
        )
        harmonics = gaussians_full.harmonics[0, anchor_indices].reshape(
            len(anchor_indices), -1
        )
        opacities = gaussians_full.opacities[0, anchor_indices].reshape(
            len(anchor_indices), -1
        )
        finite = bool(
            torch.isfinite(covariances).all()
            and torch.isfinite(harmonics).all()
            and torch.isfinite(opacities).all()
        )
        pair_first, pair_second = torch.triu_indices(
            len(anchor_indices), len(anchor_indices), offset=1,
            device=covariances.device,
        )
        if pair_first.numel() == 0:
            covariance_minimum = float("-inf")
            harmonic_minimum = float("-inf")
            opacity_maximum = float("inf")
        else:
            covariance_norms = covariances / covariances.norm(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
            harmonic_norms = harmonics / harmonics.norm(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
            covariance_values = (covariance_norms[pair_first] * covariance_norms[pair_second]).sum(dim=1)
            harmonic_values = (harmonic_norms[pair_first] * harmonic_norms[pair_second]).sum(dim=1)
            opacity_values = (
                opacities[pair_first] - opacities[pair_second]
            ).abs().amax(dim=1)
            covariance_minimum = float(covariance_values.amin().item())
            harmonic_minimum = float(harmonic_values.amin().item())
            opacity_maximum = float(opacity_values.amax().item())
        return {
            "level": level,
            "anchor_indices": list(anchor_indices),
            "anchor_count": len(anchor_indices),
            "covariance_cosine_minimum": covariance_minimum,
            "harmonic_cosine_minimum": harmonic_minimum,
            "opacity_distance_maximum": opacity_maximum,
            "covariance_cosine_threshold": MATERIALIZATION_GUARD_MIN_COVARIANCE_COSINE,
            "harmonic_cosine_threshold": MATERIALIZATION_GUARD_MIN_HARMONIC_COSINE,
            "opacity_distance_threshold": MATERIALIZATION_GUARD_MAX_OPACITY_DISTANCE,
            "passed": bool(
                finite
                and covariance_minimum >= MATERIALIZATION_GUARD_MIN_COVARIANCE_COSINE
                and harmonic_minimum >= MATERIALIZATION_GUARD_MIN_HARMONIC_COSINE
                and opacity_maximum <= MATERIALIZATION_GUARD_MAX_OPACITY_DISTANCE
            ),
            "nonprobe_s3_attribute_reads": 0,
        }

    def _primary_probe_depth_values(
        self,
        depths: torch.Tensor | None,
        *,
        tile_row: int,
        tile_column: int,
        view_index: int,
        primitive_slot: int,
    ) -> torch.Tensor | None:
        """Read only the declared primary-probe depths for one tile.

        The context guard deliberately shares the L1 routing probes rather
        than sampling another depth location.  Invalid layouts return
        ``None`` so the caller can retain the Full tile.
        """
        if depths is None or not torch.is_tensor(depths):
            return None
        values = []
        tile_y = tile_row * self.initial_tile_size
        tile_x = tile_column * self.initial_tile_size
        for local_y, local_x in self.probe_positions:
            row = tile_y + local_y
            column = tile_x + local_x
            if not 0 <= row < self.H or not 0 <= column < self.W:
                return None
            if depths.ndim == 5:
                pixel_index = row * self.W + column
                candidates = depths[0, view_index, pixel_index].reshape(-1)
                if primitive_slot >= candidates.numel():
                    return None
                values.append(candidates[primitive_slot])
            elif depths.ndim == 4:
                values.append(depths[0, view_index, row, column])
            else:
                return None
        if len(values) != len(self.probe_positions):
            return None
        return torch.stack(values)

    def context_coverage_occlusion_safety(
        self,
        gaussians_full,
        anchor_indices: List[int],
        probe_depths: torch.Tensor | None,
        *,
        view_index: int,
        level: str,
    ) -> Dict:
        """Check selected-anchor coverage and probe-depth occlusion risk.

        This optional guard has no route-selection input and never reads an
        unselected descriptor. It rejects missing/invalid producer geometry,
        large projected-footprint spread, large primary-probe depth span, and
        selected anchors whose projected centers do not overlap at two-sigma
        support. A rejection is deliberately conservative: the caller leaves
        the tile on the dense Full path.
        """
        if level not in {"L0", "L1"}:
            raise ValueError("context safety guard level must be L0 or L1")
        if (
            not isinstance(anchor_indices, list)
            or not anchor_indices
            or len(anchor_indices) != len(set(anchor_indices))
            or any(isinstance(index, bool) or not isinstance(index, int) for index in anchor_indices)
        ):
            raise ValueError("context safety guard requires unique anchor indices")
        total = gaussians_full.means.shape[1]
        if any(index < 0 or index >= total for index in anchor_indices):
            raise ValueError("context safety guard anchor index is out of range")

        result = {
            "level": level,
            "anchor_count": len(anchor_indices),
            "coverage_footprint_ratio": float("inf"),
            "relative_depth_span": float("inf"),
            "projected_center_mahalanobis_max": float("inf"),
            "coverage_passed": False,
            "occlusion_passed": False,
            "center_overlap_passed": False,
            "passed": False,
            "reason": "missing_geometry",
            "nonprobe_s3_attribute_reads": 0,
        }
        if self.context_extrinsics is None or self.context_intrinsics is None:
            return result
        if probe_depths is None or not torch.is_tensor(probe_depths):
            result["reason"] = "invalid_probe_depth"
            return result
        probe_depths = probe_depths.reshape(-1)
        if probe_depths.numel() != len(self.probe_positions) or not bool(
            torch.isfinite(probe_depths).all()
        ):
            result["reason"] = "invalid_probe_depth"
            return result

        means = gaussians_full.means[0, anchor_indices]
        covariances = gaussians_full.covariances[0, anchor_indices]
        projected = self._context_projected_moments(
            means, covariances, view_index=view_index
        )
        if projected is None:
            result["reason"] = "invalid_footprint"
            return result
        centers, projected_covariances, _, _, _ = projected
        determinants = torch.linalg.det(projected_covariances)
        if not bool(torch.isfinite(determinants).all()) or bool(
            (determinants <= 0.0).any()
        ):
            result["reason"] = "invalid_footprint"
            return result
        footprints = torch.sqrt(determinants)
        minimum = footprints.amin()
        maximum = footprints.amax()
        tiny = torch.as_tensor(
            torch.finfo(footprints.dtype).tiny,
            device=footprints.device,
            dtype=footprints.dtype,
        )
        if not bool(torch.isfinite(minimum)) or not bool(torch.isfinite(maximum)) or bool(
            minimum <= tiny
        ):
            result["reason"] = "invalid_footprint"
            return result
        footprint_ratio = float((maximum / minimum).item())
        depth_scale = probe_depths.abs().median().clamp_min(1.0e-6)
        relative_depth_span = float(
            ((probe_depths.max() - probe_depths.min()) / depth_scale).item()
        )
        maximum_center_mahalanobis = 0.0
        try:
            pair_first, pair_second = torch.triu_indices(
                len(anchor_indices), len(anchor_indices), offset=1,
                device=centers.device,
            )
            deltas = centers[pair_first] - centers[pair_second]
            pair_covariances = (
                projected_covariances[pair_first]
                + projected_covariances[pair_second]
            )
            solved = torch.linalg.solve(pair_covariances, deltas.unsqueeze(-1)).squeeze(-1)
            distances_squared = (deltas * solved).sum(dim=1)
            if not bool(torch.isfinite(distances_squared).all()):
                result["reason"] = "invalid_footprint"
                return result
            maximum_center_mahalanobis = float(
                distances_squared.clamp_min(0.0).sqrt().amax().item()
            )
        except RuntimeError:
            result["reason"] = "invalid_footprint"
            return result
        coverage_passed = footprint_ratio <= CONTEXT_GUARD_MAX_FOOTPRINT_RATIO
        occlusion_passed = relative_depth_span <= CONTEXT_GUARD_MAX_RELATIVE_DEPTH_SPAN
        center_overlap_passed = (
            maximum_center_mahalanobis <= CONTEXT_GUARD_MAX_CENTER_MAHALANOBIS
        )
        result.update(
            {
                "coverage_footprint_ratio": footprint_ratio,
                "relative_depth_span": relative_depth_span,
                "projected_center_mahalanobis_max": maximum_center_mahalanobis,
                "coverage_passed": coverage_passed,
                "occlusion_passed": occlusion_passed,
                "center_overlap_passed": center_overlap_passed,
                "passed": coverage_passed and occlusion_passed and center_overlap_passed,
                "reason": (
                    "accepted"
                    if coverage_passed and occlusion_passed and center_overlap_passed
                    else "coverage_spread"
                    if not coverage_passed
                    else "occlusion_span"
                    if not occlusion_passed
                    else "center_separation"
                ),
            }
        )
        return result

    @staticmethod
    def _prepare_source_opacities_for_deletion_certificate(
        source_opacities: torch.Tensor | None,
        *,
        source_kind: str | None,
        view_count: int,
        height: int,
        width: int,
        primitives_per_pixel: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, str]:
        """Validate the source-only alpha layout before tile routing.

        The certificate deliberately accepts only the direct S2 density mapping
        used by TranSplat/MVSplat.  DepthSplat's alpha is produced by the raw
        Gaussian head, so accepting it here would silently turn a Stage-3
        value into a route input.
        """
        if source_kind != DELETION_CERTIFICATE_SOURCE_KIND:
            return None, "untrusted_source_kind"
        if source_opacities is None:
            return None, "missing_source_opacities"
        if not torch.is_tensor(source_opacities):
            return None, "invalid_source_opacities"
        expected_shape = (
            1,
            view_count,
            height * width,
            1,
            primitives_per_pixel,
        )
        if tuple(source_opacities.shape) != expected_shape:
            return None, "invalid_source_layout"
        if (
            source_opacities.device != device
            or source_opacities.dtype != dtype
            or not torch.is_floating_point(source_opacities)
        ):
            return None, "invalid_source_tensor_type"
        if not bool(torch.isfinite(source_opacities).all()) or bool(
            ((source_opacities < 0.0) | (source_opacities > 1.0)).any()
        ):
            return None, "invalid_source_opacity_values"
        return source_opacities.reshape(1, -1), "ready"

    def exact_source_opacity_zero_deletion_certificate(
        self,
        gaussians_full,
        source_opacities: torch.Tensor | None,
        *,
        source_status: str,
        anchor_indices: List[int],
        nonprobe_indices: List[int],
        level: str,
    ) -> Dict:
        """Certify a no-op deletion without opening any non-probe S3 value.

        A Gaussian whose adapter input alpha is exactly zero has no raster
        contribution.  The selected anchors are also required to match that
        same source mapping exactly, which binds the known S2 opacity layout
        to the active adapter path.  This path skips moment matching entirely:
        changing a retained anchor would need a separate merge certificate.
        """
        result = {
            "policy": DELETION_CERTIFICATE_POLICY,
            "level": level,
            "anchor_count": len(anchor_indices),
            "nonprobe_count": len(nonprobe_indices),
            "source_status": source_status,
            "passed": False,
            "reason": source_status,
            "anchor_s3_opacity_reads": 0,
            "nonprobe_s3_attribute_reads": 0,
        }
        if source_opacities is None:
            return result
        if (
            not anchor_indices
            or len(anchor_indices) != len(set(anchor_indices))
            or len(nonprobe_indices) != len(set(nonprobe_indices))
            or set(anchor_indices) & set(nonprobe_indices)
        ):
            result["reason"] = "invalid_tile_indices"
            return result
        gaussian_count = int(gaussians_full.opacities.shape[1])
        if any(
            not isinstance(index, int) or index < 0 or index >= gaussian_count
            for index in (*anchor_indices, *nonprobe_indices)
        ):
            result["reason"] = "invalid_tile_indices"
            return result

        # This is the only Stage-3 read: selected anchors already executed by
        # the candidate path.  Non-probe alpha comes exclusively from S2.
        selected_source = source_opacities[0, anchor_indices]
        selected_output = gaussians_full.opacities[0, anchor_indices]
        result["anchor_s3_opacity_reads"] = len(anchor_indices)
        if not torch.equal(selected_source, selected_output):
            result["reason"] = "source_anchor_mapping_mismatch"
            return result
        if not bool((source_opacities[0, nonprobe_indices] == 0.0).all()):
            result["reason"] = "nonzero_source_opacity"
            return result
        result.update({"passed": True, "reason": "exact_zero_alpha"})
        return result

    # ------------------------------------------------------------------ #
    # Core interpolation / update methods                                  #
    # ------------------------------------------------------------------ #

    def _joint_calibrator_descriptor(
        self,
        *,
        probe_offset: int,
        source_means: torch.Tensor,
        source_covariances: torch.Tensor,
        source_harmonics: torch.Tensor,
        source_opacities: torch.Tensor,
        merged_mean: torch.Tensor,
        merged_covariance: torch.Tensor,
        merged_harmonics: torch.Tensor,
        merged_opacity: torch.Tensor,
        assignment_matrix: torch.Tensor,
        probe_depths: torch.Tensor,
        depth_reference: torch.Tensor,
        probe_rows: list[int],
        probe_columns: list[int],
        view_index: int,
        feature_variance: float,
        level: str,
    ) -> torch.Tensor:
        """Build the fixed target-free descriptor for one retained anchor.

        Every statistic derives from the selected source anchors, ordinary
        representative output, S1/S2 route state, or the already-resident
        context camera. The helper deliberately has no argument for skipped
        S3 attributes, target cameras, target RGB, model id, or dataset id.
        """
        source_mean = source_means[probe_offset]
        source_covariance = source_covariances[probe_offset]
        source_opacity = source_opacities[probe_offset].reshape(-1)
        if source_opacity.numel() != 1 or merged_opacity.numel() != 1:
            raise ValueError("joint calibrator requires one opacity per Gaussian")
        mean_scale = source_mean.norm().clamp_min(1.0)
        mean_delta = (merged_mean - source_mean) / mean_scale
        upper = torch.stack(
            (
                merged_covariance[0, 0] - source_covariance[0, 0],
                merged_covariance[0, 1] - source_covariance[0, 1],
                merged_covariance[0, 2] - source_covariance[0, 2],
                merged_covariance[1, 1] - source_covariance[1, 1],
                merged_covariance[1, 2] - source_covariance[1, 2],
                merged_covariance[2, 2] - source_covariance[2, 2],
            )
        )
        covariance_scale = source_covariance.abs().mean().clamp_min(1e-6)
        covariance_delta = upper / covariance_scale
        source_alpha = source_opacity.clamp(1e-6, 1.0 - 1e-6)
        merged_alpha = merged_opacity.reshape(-1).clamp(1e-6, 1.0 - 1e-6)
        opacity_values = torch.stack(
            (
                torch.logit(source_alpha)[0],
                torch.logit(merged_alpha)[0],
            )
        )
        harmonic_values = merged_harmonics.reshape(merged_harmonics.shape[0], -1)
        if harmonic_values.shape[0] != 3:
            raise ValueError("joint calibrator requires RGB spherical harmonics")
        harmonic_summary = torch.cat(
            (
                harmonic_values.mean(dim=1),
                harmonic_values.std(dim=1, unbiased=False),
            )
        )
        if assignment_matrix.numel():
            assignment = assignment_matrix[:, probe_offset].clamp_min(0.0)
            assignment_mass = assignment.sum()
            assignment_probability = assignment / assignment_mass.clamp_min(1e-8)
            assignment_entropy = -(
                assignment_probability
                * assignment_probability.clamp_min(1e-8).log()
            ).sum()
            assignment_summary = torch.stack(
                (
                    assignment_mass,
                    assignment.mean(),
                    assignment.std(unbiased=False),
                    assignment.max(),
                    assignment_entropy,
                )
            )
        else:
            assignment_summary = torch.zeros(
                5, device=merged_mean.device, dtype=merged_mean.dtype
            )
        reference_mean = depth_reference.mean()
        reference_scale = reference_mean.abs().clamp_min(1e-6)
        depth_summary = torch.stack(
            (
                torch.as_tensor(
                    feature_variance,
                    device=merged_mean.device,
                    dtype=merged_mean.dtype,
                ),
                (probe_depths[probe_offset] - reference_mean) / reference_scale,
                depth_reference.std(unbiased=False) / reference_scale,
            )
        )
        level_summary = torch.tensor(
            (1.0 if level == "L0" else 0.0, 1.0 if level == "L1" else 0.0),
            device=merged_mean.device,
            dtype=merged_mean.dtype,
        )
        if self._camera_directions is None or self.context_intrinsics is None:
            ray = torch.zeros(3, device=merged_mean.device, dtype=merged_mean.dtype)
            intrinsics = torch.zeros(
                2, device=merged_mean.device, dtype=merged_mean.dtype
            )
        else:
            ray = self._camera_directions[
                view_index, probe_rows[probe_offset], probe_columns[probe_offset]
            ].to(device=merged_mean.device, dtype=merged_mean.dtype)
            camera_intrinsics = self.context_intrinsics[0, view_index].to(
                device=merged_mean.device, dtype=merged_mean.dtype
            )
            intrinsics = torch.stack(
                (
                    camera_intrinsics[0, 0] / float(max(self.W, 1)),
                    camera_intrinsics[1, 1] / float(max(self.H, 1)),
                )
            )
        descriptor = torch.cat(
            (
                mean_delta,
                covariance_delta,
                opacity_values,
                harmonic_summary,
                assignment_summary,
                depth_summary,
                level_summary,
                ray,
                intrinsics,
            )
        )
        if descriptor.numel() != 32 or not bool(torch.isfinite(descriptor).all()):
            raise RuntimeError("joint calibrator descriptor is invalid")
        return descriptor.reshape(1, 32)

    def _apply_joint_calibrator(
        self,
        *,
        merged_mean: torch.Tensor,
        merged_covariance: torch.Tensor,
        merged_harmonics: torch.Tensor,
        merged_opacity: torch.Tensor,
        descriptor: torch.Tensor,
        level: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the already hash-bound global asset to one selected anchor."""
        if self.joint_calibrator is None or self._joint_calibrator_contract is None:
            return merged_mean, merged_covariance, merged_harmonics, merged_opacity
        coefficient_count = int(merged_harmonics.shape[-1])
        square_root = math.isqrt(coefficient_count)
        if square_root * square_root != coefficient_count or square_root < 1:
            raise ValueError("joint calibrator harmonics have an invalid degree")
        sh_degree = square_root - 1
        with torch.no_grad():
            calibrated = self.joint_calibrator(
                descriptor,
                merged_mean.reshape(1, 3),
                merged_covariance.reshape(1, 3, 3),
                merged_harmonics.reshape(1, 3, coefficient_count),
                merged_opacity.reshape(1, -1),
                sh_degree=sh_degree,
            )
        contract = self.stats["joint_calibrator"]
        contract["calls"] += 1
        contract["selected_descriptor_reads"] += 1
        if level == "L0":
            contract["l0_calls"] += 1
        elif level == "L1":
            contract["l1_calls"] += 1
        else:
            raise ValueError("joint calibrator received a non-SAEs level")
        return (
            calibrated.means[0],
            calibrated.covariances[0],
            calibrated.harmonics[0],
            calibrated.opacities[0].reshape_as(merged_opacity),
        )

    def _capture_offline_joint_calibration_sample(
        self,
        *,
        descriptor: torch.Tensor,
        merged_mean: torch.Tensor,
        merged_covariance: torch.Tensor,
        merged_harmonics: torch.Tensor,
        merged_opacity: torch.Tensor,
        level: str,
        anchor_index: int,
        teacher_nonprobe_indices: tuple[int, ...],
        teacher_assignment_weights: torch.Tensor,
    ) -> None:
        """Emit one immutable selected-only sample for author-side training.

        The callback cannot return a replacement value. It has no source for
        skipped S3 descriptors, target data, or model/dataset identifiers and
        is intentionally absent from normal runtime construction. The emitted
        assignment metadata is not a descriptor input and never reaches the
        runtime asset. It lets the author-side teacher compiler apply the
        already-frozen weights to its separately allowed dense-adaptor output.
        """
        capture = self._offline_joint_calibration_capture
        if capture is None:
            return
        if level not in {"L0", "L1"}:
            raise ValueError("offline joint-calibration capture received a non-SAEs level")
        if not isinstance(anchor_index, int) or anchor_index < 0:
            raise ValueError("offline joint-calibration capture received an invalid anchor index")
        if (
            not isinstance(teacher_nonprobe_indices, tuple)
            or not teacher_nonprobe_indices
            or any(not isinstance(index, int) or index < 0 for index in teacher_nonprobe_indices)
            or len(set(teacher_nonprobe_indices)) != len(teacher_nonprobe_indices)
        ):
            raise ValueError("offline joint-calibration capture received invalid assignment indices")
        if (
            not torch.is_tensor(teacher_assignment_weights)
            or teacher_assignment_weights.shape != (len(teacher_nonprobe_indices),)
            or not bool(torch.isfinite(teacher_assignment_weights).all())
            or bool((teacher_assignment_weights < 0.0).any())
        ):
            raise ValueError("offline joint-calibration capture received invalid assignment weights")
        sample = {
            "descriptor": descriptor.detach().clone(),
            "means": merged_mean.detach().clone(),
            "covariances": merged_covariance.detach().clone(),
            "harmonics": merged_harmonics.detach().clone(),
            "opacities": merged_opacity.detach().clone(),
            "level": level,
            "anchor_index": anchor_index,
            "teacher_nonprobe_indices": teacher_nonprobe_indices,
            "teacher_assignment_weights": teacher_assignment_weights.detach().clone(),
        }
        if capture(sample) is not None:
            raise RuntimeError("offline joint-calibration capture must not return a value")
        self.stats["offline_joint_calibration_capture_calls"] += 1

    def _weighted_moment_match(
        self,
        gaussians_full,
        probe_indices: List[int],
        non_probe_map: Dict[Tuple[int, int], int],
        feat_norm,
        depths,
        level: str,
        tile_y: int,
        tile_x: int,
        view_index: int = 0,
        primitive_slot: int = 0,
        feature_variance: float = 0.0,
        retained_positions: List[Tuple[int, int]] = None,
        opacity_aggregation: str = "range-constrained-average",
        output_style: str = "representative",
        merge_semantics: str = "assignment-mixture",
        defer_assignment_consensus_plan: bool = False,
        defer_same_budget_dense_oracle_plan: bool = False,
    ):
        """Absorb a tile using only selected-anchor Stage-3 attributes.

        Non-anchor pixels contribute S1 feature and coordinate evidence. Their
        Gaussian moments are interpolated from the selected L0/L1 outputs
        before the paper's assignment and moment-conservation update.

        ``assignment-weighted-optical-depth`` is intentionally diagnostic
        only. It keeps the same router, selected anchors, and assignment
        matrix, but aggregates inferred selected-anchor optical depth rather
        than alpha averages. It never reads a skipped Stage-3 Gaussian.
        """
        T = self.initial_tile_size
        device = gaussians_full.means.device
        K = len(probe_indices)
        if K < 1:
            return (
                (True, None)
                if (
                    defer_assignment_consensus_plan
                    or defer_same_budget_dense_oracle_plan
                )
                else None
            )
        if opacity_aggregation not in (
            "range-constrained-average",
            "assignment-weighted-optical-depth",
        ):
            raise ValueError(f"unsupported opacity aggregation: {opacity_aggregation}")
        if output_style not in (
            "representative",
            "virtual-reconstruction",
            "assignment-consensus-adapter-pseudo-descriptor",
            "same-budget-dense-oracle",
        ):
            raise ValueError(f"unsupported SAES output style: {output_style}")
        if merge_semantics not in (
            "assignment-mixture",
            "assignment-consensus-adapter-pseudo-descriptor",
            "same-budget-dense-oracle",
            "conditional-anchor-transport",
            "conditional-adapter-offset-transport",
            "conditional-adapter-offset-attribute-transport",
            "conditional-optical-mass",
            "conditional-projected-optical-mass",
        ):
            raise ValueError(f"unsupported SAES merge semantics: {merge_semantics}")
        if (
            output_style == "virtual-reconstruction"
            and merge_semantics != "assignment-mixture"
        ):
            raise ValueError(
                "virtual reconstruction requires unconditional anchor interpolation"
            )
        if (
            output_style == "assignment-consensus-adapter-pseudo-descriptor"
            and merge_semantics != "assignment-consensus-adapter-pseudo-descriptor"
        ):
            raise ValueError(
                "assignment-consensus pseudo descriptors require their isolated "
                "virtual-output merge semantics"
            )
        if (
            merge_semantics == "assignment-consensus-adapter-pseudo-descriptor"
            and output_style != "assignment-consensus-adapter-pseudo-descriptor"
        ):
            raise ValueError(
                "assignment-consensus merge semantics require virtual-only output"
            )
        if (
            output_style == "same-budget-dense-oracle"
            and merge_semantics != "same-budget-dense-oracle"
        ):
            raise ValueError(
                "same-budget dense oracle requires its isolated merge semantics"
            )
        if (
            merge_semantics == "same-budget-dense-oracle"
            and output_style != "same-budget-dense-oracle"
        ):
            raise ValueError(
                "same-budget dense-oracle merge semantics require oracle output"
            )
        is_assignment_consensus = (
            output_style == "assignment-consensus-adapter-pseudo-descriptor"
        )
        is_same_budget_dense_oracle = output_style == "same-budget-dense-oracle"
        selected_only_s3 = (
            is_assignment_consensus
            or merge_semantics == "conditional-adapter-offset-attribute-transport"
            or self.joint_calibrator is not None
            or self._offline_joint_calibration_capture is not None
        )
        if defer_assignment_consensus_plan and not is_assignment_consensus:
            raise ValueError(
                "only assignment-consensus virtual outputs may defer a tile plan"
            )
        if defer_same_budget_dense_oracle_plan and not is_same_budget_dense_oracle:
            raise ValueError(
                "only same-budget dense-oracle outputs may defer a tile plan"
            )

        gaussian_dtype = gaussians_full.means.dtype
        if not selected_only_s3:
            means = gaussians_full.means[0]
            covs = gaussians_full.covariances[0]
            harmonics = gaussians_full.harmonics[0]
            opacities = gaussians_full.opacities[0]

        # Flat Gaussian indices contain the view and primitive slot, so spatial
        # probe coordinates must be recovered from tile-local positions.
        retained_positions = retained_positions or self.probe_positions
        if len(retained_positions) != K:
            raise ValueError("retained position count does not match probe indices")
        probe_gy = [tile_y + local_y for local_y, _ in retained_positions]
        probe_gx = [tile_x + local_x for _, local_x in retained_positions]

        probe_feats = []
        probe_depths = []
        view_features = None
        if feat_norm is not None:
            view_features = feat_norm[view_index] if feat_norm.dim() == 4 else feat_norm
        for k in range(K):
            gy, gx = probe_gy[k], probe_gx[k]
            probe_feats.append(
                view_features[:, gy, gx]
                if (
                    view_features is not None
                    and gy < view_features.shape[1]
                    and gx < view_features.shape[2]
                )
                else None
            )
            if depths is not None:
                if depths.dim() == 5:
                    pixel_index = gy * self.W + gx
                    values = depths[0, view_index, pixel_index].reshape(-1)
                    slot = min(primitive_slot, values.numel() - 1)
                    probe_depths.append(values[slot].item())
                elif depths.dim() == 4:
                    probe_depths.append(depths[0, view_index, gy, gx].item())
                else:
                    probe_depths.append(1.0)
            else:
                probe_depths.append(1.0)

        probe_depth_tensor = torch.tensor(
            probe_depths, device=device, dtype=gaussian_dtype
        )
        depth_reference_tensor = probe_depth_tensor
        if level == "L1":
            primary_count = len(self.probe_positions)
            if (
                retained_positions[:primary_count] != self.probe_positions
                or primary_count > probe_depth_tensor.numel()
            ):
                raise ValueError(
                    "L1 retained anchors must begin with the primary probe set"
                )
            depth_reference_tensor = probe_depth_tensor[:primary_count]

        if selected_only_s3:
            # Select before indexing the batch dimension. These target-free
            # paths must not materialize a view of skipped S3 descriptors even
            # transiently; their output writes below are likewise direct.
            source_means = gaussians_full.means[0, probe_indices].clone()
            source_covs = gaussians_full.covariances[0, probe_indices].clone()
            source_harmonics = gaussians_full.harmonics[0, probe_indices].clone()
            source_opacities = gaussians_full.opacities[0, probe_indices].clone()
        else:
            source_means = means[probe_indices].clone()
            source_covs = covs[probe_indices].clone()
            source_harmonics = harmonics[probe_indices].clone()
            source_opacities = opacities[probe_indices].clone()
        non_probe_items = list(non_probe_map.items())
        assignments = []
        coordinate_scale = max(T - 1, 1)
        for (local_y, local_x), _ in non_probe_items:
            gy = tile_y + local_y
            gx = tile_x + local_x
            feature_i = (
                view_features[:, gy, gx]
                if view_features is not None
                and gy < view_features.shape[1]
                and gx < view_features.shape[2]
                else None
            )
            spatial_distances = [
                ((gy - probe_gy[k]) / coordinate_scale) ** 2
                + ((gx - probe_gx[k]) / coordinate_scale) ** 2
                for k in range(K)
            ]
            feature_distances = []
            for k in range(K):
                feature = 0.0
                probe_feature = probe_feats[k]
                if (
                    feature_i is not None
                    and probe_feature is not None
                    and feature_i.shape == probe_feature.shape
                ):
                    feature = (feature_i - probe_feature).square().sum().item()
                feature_distances.append(feature)
            assignments.append(
                paper_assignment_weights(
                    torch.tensor(
                        spatial_distances, device=device, dtype=gaussian_dtype
                    ),
                    torch.tensor(
                        feature_distances, device=device, dtype=gaussian_dtype
                    ),
                    feature_variance=feature_variance,
                    beta_x=self.beta_x,
                    beta_f=self.beta_f,
                    level=level,
                    probe_depths=probe_depth_tensor if level == "L1" else None,
                    depth_reference_depths=(
                        depth_reference_tensor if level == "L1" else None
                    ),
                    beta_d=self.beta_d if level == "L1" else None,
                )
            )

        assignment_matrix = (
            torch.stack(assignments, dim=0)
            if assignments
            else torch.empty(0, K, device=device, dtype=gaussian_dtype)
        )
        if assignment_matrix.numel():
            error = float((assignment_matrix.sum(dim=1) - 1.0).abs().max().item())
            self.stats["assignment_weight_sum_error_max"] = max(
                self.stats["assignment_weight_sum_error_max"], error
            )

        if output_style == "assignment-consensus-adapter-pseudo-descriptor":
            if not assignments:
                return (False, None) if defer_assignment_consensus_plan else False
            target_positions = [
                (tile_y + local_y, tile_x + local_x)
                for (local_y, local_x), _ in non_probe_items
            ]
            consensus_geometry = self._assignment_consensus_adapter_pseudo_geometry(
                source_means,
                source_covs,
                probe_depth_tensor,
                list(zip(probe_gy, probe_gx)),
                assignment_matrix,
                target_positions,
                view_index=view_index,
            )
            if consensus_geometry is None:
                return (True, None) if defer_assignment_consensus_plan else True
            consensus_means, consensus_covariances = consensus_geometry
            flat_harmonics = source_harmonics.reshape(K, -1)
            flat_opacities = source_opacities.reshape(K, -1)
            if (
                not bool(torch.isfinite(flat_opacities).all())
                or bool((flat_opacities < 0.0).any())
                or bool((flat_opacities > 1.0).any())
            ):
                return (True, None) if defer_assignment_consensus_plan else True
            consensus_harmonics = assignment_matrix @ flat_harmonics
            consensus_opacities = assignment_matrix @ flat_opacities
            if not bool(torch.isfinite(consensus_harmonics).all()) or not bool(
                torch.isfinite(consensus_opacities).all()
            ):
                return (True, None) if defer_assignment_consensus_plan else True

            plan = {
                "output_indices": tuple(
                    output_index for _, output_index in non_probe_items
                ),
                "means": consensus_means,
                "covariances": consensus_covariances,
                "harmonics": consensus_harmonics,
                "opacities": consensus_opacities,
                "harmonic_shape": source_harmonics.shape[1:],
                "opacity_shape": source_opacities.shape[1:],
                "anchor_count": K,
            }
            if defer_assignment_consensus_plan:
                return False, plan
            self._commit_assignment_consensus_plan(gaussians_full, plan)
            return False

        if output_style == "same-budget-dense-oracle":
            plan = self._same_budget_dense_oracle_plan(
                gaussians_full,
                probe_indices=probe_indices,
                non_probe_items=non_probe_items,
                retained_positions=retained_positions,
                assignment_matrix=assignment_matrix,
                tile_y=tile_y,
                tile_x=tile_x,
                view_index=view_index,
            )
            if plan is None:
                return (True, None) if defer_same_budget_dense_oracle_plan else True
            if defer_same_budget_dense_oracle_plan:
                return False, plan
            self._commit_same_budget_dense_oracle_plan(gaussians_full, plan)
            return False

        if merge_semantics in (
            "conditional-optical-mass",
            "conditional-projected-optical-mass",
        ):
            return self._conditional_optical_mass_merge(
                means=means,
                covariances=covs,
                harmonics=harmonics,
                opacities=opacities,
                source_means=source_means,
                source_covariances=source_covs,
                source_harmonics=source_harmonics,
                source_opacities=source_opacities,
                probe_indices=probe_indices,
                probe_depths=probe_depth_tensor,
                probe_positions=list(zip(probe_gy, probe_gx)),
                non_probe_items=non_probe_items,
                assignment_matrix=assignment_matrix,
                view_index=view_index,
                projected_footprint=(
                    merge_semantics == "conditional-projected-optical-mass"
                ),
            )

        flat_harmonics = source_harmonics.reshape(K, -1)
        flat_opacities = source_opacities.reshape(K, -1).clamp(0.0, 1.0 - 1e-6)
        source_optical_depth = -torch.log1p(-flat_opacities)
        pseudo_means = None
        pseudo_harmonics = None
        pseudo_opacities = None
        pseudo_optical_depth = None
        pseudo_covariances = []
        if assignments and merge_semantics in (
            "assignment-mixture",
            "conditional-adapter-offset-attribute-transport",
        ):
            # The paper's range-constrained SH/opacity average is evaluated
            # from selected anchors only.  The attribute-transport diagnostic
            # uses these convex skipped-descriptor estimates while leaving the
            # adapter-offset geometry path below unchanged.
            pseudo_harmonics = assignment_matrix @ flat_harmonics
            pseudo_opacities = assignment_matrix @ flat_opacities
            pseudo_optical_depth = assignment_matrix @ source_optical_depth
        if merge_semantics == "assignment-mixture" and assignments:
            pseudo_means = assignment_matrix @ source_means
            if self._camera_directions is not None:
                pseudo_means = torch.stack(
                    [
                        self._camera_aware_interpolated_mean(
                            source_means,
                            probe_depth_tensor,
                            list(zip(probe_gy, probe_gx)),
                            view_index=view_index,
                            target_row=tile_y + local_y,
                            target_column=tile_x + local_x,
                            weights=assignment,
                        )
                        for assignment, ((local_y, local_x), _) in zip(
                            assignment_matrix, non_probe_items
                        )
                    ],
                    dim=0,
                )
            pseudo_covariances = [
                self._interpolate_intrinsic_covariance(source_covs, row)
                for row in assignment_matrix
            ]

        if output_style == "virtual-reconstruction":
            # Preserve a Gaussian descriptor at every skipped pixel while
            # avoiding its neural S2/S3 evaluation. The descriptor is fully
            # reconstructed from selected-anchor S1/S2/S3 values: C2W-aware
            # means, intrinsic SPD covariance, SH, and alpha. This is a
            # diagnostic engineering path only until its fixed-function work,
            # storage traffic, and quality are all independently audited.
            harmonic_shape = source_harmonics.shape[1:]
            opacity_shape = source_opacities.shape[1:]
            for pseudo_index, ((_, output_index), pseudo_covariance) in enumerate(
                zip(non_probe_items, pseudo_covariances)
            ):
                eigenvalues, eigenvectors = torch.linalg.eigh(pseudo_covariance)
                eigenvalues = eigenvalues.clamp_min(1e-8)
                reconstructed_covariance = (
                    eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.mT
                )
                if bool(
                    (torch.linalg.eigvalsh(reconstructed_covariance) < -1e-7)
                    .any()
                    .item()
                ):
                    self.stats["covariance_psd_violations"] += 1
                means[output_index] = pseudo_means[pseudo_index]
                covs[output_index] = reconstructed_covariance
                harmonics[output_index] = pseudo_harmonics[pseudo_index].reshape(
                    harmonic_shape
                )
                opacities[output_index] = (
                    pseudo_opacities[pseudo_index]
                    .clamp(0.0, 1.0 - 1e-6)
                    .reshape(opacity_shape)
                )
            self.stats["virtual_reconstructed_gaussians"] += len(non_probe_items)
            return

        adapter_offset_conditional_means = None
        if assignments and merge_semantics in (
            "conditional-adapter-offset-transport",
            "conditional-adapter-offset-attribute-transport",
        ):
            target_positions = [
                (tile_y + local_y, tile_x + local_x)
                for (local_y, local_x), _ in non_probe_items
            ]
            # Validate every selected anchor before writing any output.  A
            # failed geometry recovery must leave the whole tile on Full, not
            # partially update anchors that happened to be visited first.
            adapter_offset_conditional_means = []
            for probe_offset in range(K):
                transported = self._adapter_offset_transport_means(
                    source_means[probe_offset],
                    probe_depth_tensor[probe_offset],
                    (probe_gy[probe_offset], probe_gx[probe_offset]),
                    target_positions,
                    view_index=view_index,
                )
                if transported is None:
                    return True
                adapter_offset_conditional_means.append(transported)
            self.stats["adapter_offset_transport_uses"] += K * len(non_probe_items)
            if merge_semantics == "conditional-adapter-offset-attribute-transport":
                self.stats["adapter_offset_attribute_transport_uses"] += int(
                    assignment_matrix.numel()
                )

        for probe_offset, probe_index in enumerate(probe_indices):
            absorbed = (
                assignment_matrix[:, probe_offset]
                if assignments
                else torch.empty(0, device=device, dtype=gaussian_dtype)
            )
            weights = torch.cat(
                (torch.ones(1, device=device, dtype=gaussian_dtype), absorbed)
            )
            normalized = weights / weights.sum().clamp_min(1e-8)
            if assignments and merge_semantics in (
                "conditional-anchor-transport",
                "conditional-adapter-offset-transport",
                "conditional-adapter-offset-attribute-transport",
            ):
                target_positions = [position for position, _ in non_probe_items]
                conditional_means = (
                    adapter_offset_conditional_means[probe_offset]
                    if merge_semantics
                    in (
                        "conditional-adapter-offset-transport",
                        "conditional-adapter-offset-attribute-transport",
                    )
                    else self._anchor_conditioned_transport_means(
                        source_means[probe_offset],
                        probe_depth_tensor[probe_offset],
                        (probe_gy[probe_offset], probe_gx[probe_offset]),
                        [
                            (tile_y + local_y, tile_x + local_x)
                            for local_y, local_x in target_positions
                        ],
                        view_index=view_index,
                    )
                )
                conditional_covariances = (
                    source_covs[probe_offset]
                    .unsqueeze(0)
                    .expand(len(non_probe_items), -1, -1)
                )
                if merge_semantics == "conditional-adapter-offset-attribute-transport":
                    conditional_harmonics = pseudo_harmonics
                    conditional_opacities = pseudo_opacities
                else:
                    conditional_harmonics = (
                        flat_harmonics[probe_offset]
                        .unsqueeze(0)
                        .expand(len(non_probe_items), -1)
                    )
                    conditional_opacities = (
                        flat_opacities[probe_offset]
                        .unsqueeze(0)
                        .expand(len(non_probe_items), -1)
                    )
                contributor_means = torch.cat(
                    (source_means[probe_offset].unsqueeze(0), conditional_means), dim=0
                )
                contributor_covs = torch.cat(
                    (source_covs[probe_offset].unsqueeze(0), conditional_covariances),
                    dim=0,
                )
                contributor_harmonics = torch.cat(
                    (flat_harmonics[probe_offset].unsqueeze(0), conditional_harmonics),
                    dim=0,
                )
                contributor_opacities = torch.cat(
                    (flat_opacities[probe_offset].unsqueeze(0), conditional_opacities),
                    dim=0,
                )
            else:
                contributor_means = (
                    torch.cat(
                        (source_means[probe_offset].unsqueeze(0), pseudo_means), dim=0
                    )
                    if assignments
                    else source_means[probe_offset].unsqueeze(0)
                )
                contributor_covs = (
                    torch.stack([source_covs[probe_offset], *pseudo_covariances], dim=0)
                    if assignments
                    else source_covs[probe_offset].unsqueeze(0)
                )
                contributor_harmonics = (
                    torch.cat(
                        (flat_harmonics[probe_offset].unsqueeze(0), pseudo_harmonics),
                        dim=0,
                    )
                    if assignments
                    else flat_harmonics[probe_offset].unsqueeze(0)
                )
                contributor_opacities = (
                    torch.cat(
                        (flat_opacities[probe_offset].unsqueeze(0), pseudo_opacities),
                        dim=0,
                    )
                    if assignments
                    else flat_opacities[probe_offset].unsqueeze(0)
                )
            merged_mean = torch.einsum("n,ni->i", normalized, contributor_means)
            centered = contributor_means - merged_mean
            second_moment = contributor_covs + torch.einsum(
                "ni,nj->nij", centered, centered
            )
            merged_covariance = torch.einsum("n,nij->ij", normalized, second_moment)
            merged_covariance = (merged_covariance + merged_covariance.mT) * 0.5
            merged_covariance = self._multicontext_tangent_candidate(
                output_mean=merged_mean,
                local_covariance=merged_covariance,
                contributor_means=contributor_means,
                contributor_covariances=contributor_covs,
                contributor_weights=normalized,
            )

            harmonic_shape = source_harmonics.shape[1:]
            merged_harmonics = torch.einsum(
                "n,nk->k", normalized, contributor_harmonics
            ).reshape(harmonic_shape)
            merged_harmonics = torch.maximum(
                torch.minimum(merged_harmonics, source_harmonics.amax(dim=0)),
                source_harmonics.amin(dim=0),
            )

            opacity_shape = source_opacities.shape[1:]
            if opacity_aggregation == "assignment-weighted-optical-depth":
                # Alpha compositing is additive in tau=-log(1-alpha).  Each
                # skipped position contributes an inferred tau, constructed
                # only from selected anchor opacities, to its soft-assigned
                # retained anchor. This restores the assignment-surrogate
                # mass without consulting the skipped Stage-3 outputs.
                if assignments and merge_semantics in (
                    "conditional-anchor-transport",
                    "conditional-adapter-offset-transport",
                ):
                    absorbed_optical_depth = (
                        absorbed.sum() * source_optical_depth[probe_offset]
                    )
                elif assignments:
                    absorbed_optical_depth = torch.einsum(
                        "n,nk->k", absorbed, pseudo_optical_depth
                    )
                else:
                    absorbed_optical_depth = torch.zeros_like(
                        source_optical_depth[probe_offset]
                    )
                merged_optical_depth = (
                    source_optical_depth[probe_offset] + absorbed_optical_depth
                )
                merged_opacity = (
                    (-torch.expm1(-merged_optical_depth))
                    .clamp(0.0, 1.0 - 1e-6)
                    .reshape(opacity_shape)
                )
            else:
                # Section 3 specifies a range-constrained opacity average. The
                # pseudo Gaussian contributes with the same assignment weight
                # used for its first/second moments; do not multiply
                # pseudo-opacity residuals into a new opaque cluster.
                merged_opacity = torch.einsum(
                    "n,nk->k", normalized, contributor_opacities
                )
                if merge_semantics == "conditional-adapter-offset-attribute-transport":
                    opacity_upper = flat_opacities.amax(dim=0)
                    opacity_lower = flat_opacities.amin(dim=0)
                else:
                    opacity_upper = contributor_opacities.amax(dim=0)
                    opacity_lower = contributor_opacities.amin(dim=0)
                merged_opacity = (
                    torch.maximum(
                        torch.minimum(merged_opacity, opacity_upper),
                        opacity_lower,
                    )
                    .clamp(0.0, 1.0 - 1e-6)
                    .reshape(opacity_shape)
                )

            eigenvalues, eigenvectors = torch.linalg.eigh(merged_covariance)
            eigenvalues = eigenvalues.clamp_min(1e-8)
            merged_covariance = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.mT
            if bool((torch.linalg.eigvalsh(merged_covariance) < -1e-7).any().item()):
                self.stats["covariance_psd_violations"] += 1

            if self.joint_calibrator is not None or self._offline_joint_calibration_capture is not None:
                descriptor = self._joint_calibrator_descriptor(
                    probe_offset=probe_offset,
                    source_means=source_means,
                    source_covariances=source_covs,
                    source_harmonics=source_harmonics,
                    source_opacities=source_opacities,
                    merged_mean=merged_mean,
                    merged_covariance=merged_covariance,
                    merged_harmonics=merged_harmonics,
                    merged_opacity=merged_opacity,
                    assignment_matrix=assignment_matrix,
                    probe_depths=probe_depth_tensor,
                    depth_reference=depth_reference_tensor,
                    probe_rows=probe_gy,
                    probe_columns=probe_gx,
                    view_index=view_index,
                    feature_variance=feature_variance,
                    level=level,
                )
                if self.joint_calibrator is not None:
                    (
                        merged_mean,
                        merged_covariance,
                        merged_harmonics,
                        merged_opacity,
                    ) = self._apply_joint_calibrator(
                        merged_mean=merged_mean,
                        merged_covariance=merged_covariance,
                        merged_harmonics=merged_harmonics,
                        merged_opacity=merged_opacity,
                        descriptor=descriptor,
                        level=level,
                    )
                else:
                    teacher_nonprobe_indices = tuple(
                        int(output_index) for _, output_index in non_probe_items
                    )
                    self._capture_offline_joint_calibration_sample(
                        descriptor=descriptor,
                        merged_mean=merged_mean,
                        merged_covariance=merged_covariance,
                        merged_harmonics=merged_harmonics,
                        merged_opacity=merged_opacity,
                        level=level,
                        anchor_index=probe_index,
                        teacher_nonprobe_indices=teacher_nonprobe_indices,
                        teacher_assignment_weights=assignment_matrix[:, probe_offset],
                    )

            if selected_only_s3:
                gaussians_full.means[0, probe_index] = merged_mean
                gaussians_full.covariances[0, probe_index] = merged_covariance
                gaussians_full.harmonics[0, probe_index] = merged_harmonics
                gaussians_full.opacities[0, probe_index] = merged_opacity
            else:
                means[probe_index] = merged_mean
                covs[probe_index] = merged_covariance
                harmonics[probe_index] = merged_harmonics
                opacities[probe_index] = merged_opacity

        output_opacities = (
            gaussians_full.opacities[0, probe_indices].reshape(K, -1)
            if selected_only_s3
            else opacities[probe_indices].reshape(K, -1)
        )
        if opacity_aggregation == "assignment-weighted-optical-depth":
            observed_optical_depth = -torch.log1p(
                -output_opacities.clamp(0.0, 1.0 - 1e-6)
            ).sum()
            expected_optical_depth = source_optical_depth.sum()
            if assignments and merge_semantics in (
                "assignment-mixture",
                "conditional-adapter-offset-attribute-transport",
            ):
                expected_optical_depth = (
                    expected_optical_depth + pseudo_optical_depth.sum()
                )
            aggregation_error = float(
                (observed_optical_depth - expected_optical_depth).abs().item()
            )
            self.stats["optical_depth_assignment_error_max"] = max(
                self.stats["optical_depth_assignment_error_max"], aggregation_error
            )
        residual = 1.0 - output_opacities
        residual_error = float(
            torch.maximum((-residual).clamp_min(0.0), (residual - 1.0).clamp_min(0.0))
            .max()
            .item()
        )
        self.stats["opacity_transmittance_error_max"] = max(
            self.stats["opacity_transmittance_error_max"], residual_error
        )
        return False

    def _conditional_optical_mass_merge(
        self,
        *,
        means: torch.Tensor,
        covariances: torch.Tensor,
        harmonics: torch.Tensor,
        opacities: torch.Tensor,
        source_means: torch.Tensor,
        source_covariances: torch.Tensor,
        source_harmonics: torch.Tensor,
        source_opacities: torch.Tensor,
        probe_indices: List[int],
        probe_depths: torch.Tensor,
        probe_positions: List[Tuple[int, int]],
        non_probe_items: List[Tuple[Tuple[int, int], int]],
        assignment_matrix: torch.Tensor,
        view_index: int,
        projected_footprint: bool = False,
    ) -> bool:
        """Merge each pseudo descriptor into only its assigned anchor.

        A skipped position ``i`` is represented once for each receiving anchor
        ``p`` with its one existing paper assignment ``r_i,p``.  Its descriptor
        is transported from ``p`` alone, rather than first mixing all anchors
        and then multiplying by ``r_i,p`` a second time. Optical-density mass
        is evaluated either in native 3D volume coordinates or, for the
        projected diagnostic, in the producer context camera's raster-space
        footprint. All candidate updates are held locally; a non-finite,
        non-PSD, out-of-range, or non-conserving result returns ``True`` so the
        caller keeps the tile on the Full path.
        """
        if source_opacities[0].numel() != 1:
            # The submitted models use one opacity per primitive.  Do not
            # silently choose a component-wise mass rule for another layout.
            return True
        dtype = source_means.dtype
        device = source_means.device
        count = len(probe_indices)
        if assignment_matrix.shape != (len(non_probe_items), count):
            return True
        eye = torch.eye(3, device=device, dtype=dtype)
        eps = torch.as_tensor(1e-8, device=device, dtype=dtype)
        source_covariances = (source_covariances + source_covariances.mT) * 0.5
        source_eigenvalues = torch.linalg.eigvalsh(source_covariances)
        if not bool(torch.isfinite(source_eigenvalues).all()) or bool(
            (source_eigenvalues < -1e-7).any()
        ):
            return True
        source_determinants = torch.linalg.det(source_covariances + eps * eye)
        if not bool(torch.isfinite(source_determinants).all()) or bool(
            (source_determinants <= 0.0).any()
        ):
            return True
        source_scales = (
            self._context_projected_footprint_scales(
                source_means, source_covariances, view_index=view_index
            )
            if projected_footprint
            else torch.sqrt(source_determinants)
        )
        if source_scales is None:
            return True
        source_alpha = source_opacities.reshape(count, 1)
        if not bool(torch.isfinite(source_alpha).all()) or bool(
            ((source_alpha < 0.0) | (source_alpha >= 1.0)).any()
        ):
            return True
        source_tau = -torch.log1p(-source_alpha)
        source_mass = source_tau[:, 0] * source_scales
        if not bool(torch.isfinite(source_mass).all()) or bool(
            (source_mass < 0.0).any()
        ):
            return True

        target_positions = [
            (local_y, local_x) for (local_y, local_x), _ in non_probe_items
        ]
        candidate_updates = []
        mass_error_max = 0.0
        for probe_offset, probe_index in enumerate(probe_indices):
            assignment = assignment_matrix[:, probe_offset]
            if not bool(torch.isfinite(assignment).all()) or bool(
                (assignment < 0.0).any()
            ):
                return True
            conditional_means = self._anchor_conditioned_transport_means(
                source_means[probe_offset],
                probe_depths[probe_offset],
                probe_positions[probe_offset],
                target_positions,
                view_index=view_index,
            )
            mass_weights = torch.cat(
                (
                    source_mass[probe_offset].reshape(1),
                    assignment * source_mass[probe_offset],
                )
            )
            total_mass = mass_weights.sum()
            if not bool(torch.isfinite(total_mass)) or bool(total_mass <= 0.0):
                return True
            normalized = mass_weights / total_mass
            contributor_means = torch.cat(
                (source_means[probe_offset].unsqueeze(0), conditional_means), dim=0
            )
            contributor_covariances = (
                source_covariances[probe_offset]
                .unsqueeze(0)
                .expand(contributor_means.shape[0], -1, -1)
            )
            merged_mean = torch.einsum("n,ni->i", normalized, contributor_means)
            centered = contributor_means - merged_mean
            merged_covariance = torch.einsum(
                "n,nij->ij",
                normalized,
                contributor_covariances
                + torch.einsum("ni,nj->nij", centered, centered),
            )
            merged_covariance = (merged_covariance + merged_covariance.mT) * 0.5
            eigenvalues, eigenvectors = torch.linalg.eigh(merged_covariance)
            if not bool(torch.isfinite(eigenvalues).all()) or bool(
                (eigenvalues < -1e-7).any()
            ):
                return True
            merged_covariance = (
                eigenvectors @ torch.diag(eigenvalues.clamp_min(eps)) @ eigenvectors.mT
            )
            merged_determinant = torch.linalg.det(merged_covariance + eps * eye)
            if not bool(torch.isfinite(merged_determinant)) or bool(
                merged_determinant <= 0.0
            ):
                return True
            merged_scale = (
                self._context_projected_footprint_scales(
                    merged_mean.unsqueeze(0),
                    merged_covariance.unsqueeze(0),
                    view_index=view_index,
                )
                if projected_footprint
                else torch.sqrt(merged_determinant).reshape(1)
            )
            if merged_scale is None:
                return True
            merged_scale = merged_scale.reshape(())
            merged_tau = total_mass / merged_scale
            merged_alpha = -torch.expm1(-merged_tau)
            if not bool(torch.isfinite(merged_alpha)) or bool(
                (merged_alpha < 0.0) | (merged_alpha >= 1.0)
            ):
                return True
            observed_mass = merged_tau * merged_scale
            mass_error = float((observed_mass - total_mass).abs().item())
            mass_tolerance = float((1e-6 + 1e-5 * total_mass.detach().abs()).item())
            if mass_error > mass_tolerance:
                return True
            mass_error_max = max(mass_error_max, mass_error)
            harmonic = source_harmonics[probe_offset]
            harmonic = torch.maximum(
                torch.minimum(harmonic, source_harmonics.amax(dim=0)),
                source_harmonics.amin(dim=0),
            )
            candidate_updates.append(
                (probe_index, merged_mean, merged_covariance, harmonic, merged_alpha)
            )

        for probe_index, mean, covariance, harmonic, alpha in candidate_updates:
            means[probe_index] = mean
            covariances[probe_index] = covariance
            harmonics[probe_index] = harmonic
            opacities[probe_index] = alpha.reshape_as(opacities[probe_index])
        self.stats["conditional_assignment_uses"] += int(assignment_matrix.numel())
        self.stats["conditional_mass_conservation_error_max"] = max(
            self.stats["conditional_mass_conservation_error_max"], mass_error_max
        )
        return False

    def _spread_probe_covariances(
        self,
        gaussians_full,
        probe_indices: List[int],
        non_probe_map: Dict[Tuple[int, int], int],
        feat_norm,
        depths,
        level: str,
        tile_y: int,
        tile_x: int,
        view_index: int = 0,
        primitive_slot: int = 0,
        retained_positions: List[Tuple[int, int]] = None,
    ) -> None:
        """Diagnostic probe-only covariance expansion from commit adc7092."""
        device = gaussians_full.means.device
        dtype = gaussians_full.means.dtype
        covariances = gaussians_full.covariances[0]
        retained_positions = retained_positions or self.probe_positions
        if len(retained_positions) != len(probe_indices):
            raise ValueError("retained position count does not match probe indices")
        probe_y = [tile_y + local_y for local_y, _ in retained_positions]
        probe_x = [tile_x + local_x for _, local_x in retained_positions]
        view_features = None
        if feat_norm is not None:
            view_features = feat_norm[view_index] if feat_norm.dim() == 4 else feat_norm

        probe_features = []
        probe_depths = []
        for gy, gx in zip(probe_y, probe_x):
            probe_features.append(
                view_features[:, gy, gx]
                if view_features is not None
                and gy < view_features.shape[1]
                and gx < view_features.shape[2]
                else None
            )
            if depths is not None and depths.dim() == 5:
                values = depths[0, view_index, gy * self.W + gx].reshape(-1)
                slot = min(primitive_slot, values.numel() - 1)
                probe_depths.append(float(values[slot].item()))
            elif depths is not None and depths.dim() == 4:
                probe_depths.append(float(depths[0, view_index, gy, gx].item()))
            else:
                probe_depths.append(1.0)

        count = len(probe_indices)
        spatial_bandwidth_sq = (self.initial_tile_size / 2.0) ** 2
        feature_bandwidth_sq = 0.09
        depth_bandwidth = 0.1
        depth_mean = sum(probe_depths) / max(count, 1)
        spread = [torch.zeros(3, 3, device=device, dtype=dtype) for _ in range(count)]
        weights = [0.0 for _ in range(count)]
        image_scale = float(max(self.W, self.H, 1))

        for local_y, local_x in non_probe_map:
            gy = tile_y + local_y
            gx = tile_x + local_x
            feature = (
                view_features[:, gy, gx]
                if view_features is not None
                and gy < view_features.shape[1]
                and gx < view_features.shape[2]
                else None
            )
            raw = []
            for index in range(count):
                spatial = math.exp(
                    -((gy - probe_y[index]) ** 2 + (gx - probe_x[index]) ** 2)
                    / (spatial_bandwidth_sq + 1e-8)
                )
                probe_feature = probe_features[index]
                affinity = 1.0
                if (
                    feature is not None
                    and probe_feature is not None
                    and feature.shape == probe_feature.shape
                ):
                    affinity = math.exp(
                        -float((feature - probe_feature).square().sum().item())
                        / (feature_bandwidth_sq + 1e-8)
                    )
                weight = spatial * affinity
                if level == "L1":
                    weight *= math.exp(
                        -abs(probe_depths[index] - depth_mean)
                        / (depth_bandwidth + 1e-8)
                    )
                raw.append(weight)
            total = sum(raw) + 1e-8
            for index, value in enumerate(raw):
                normalized = value / total
                if normalized < 1e-6:
                    continue
                scale = max(probe_depths[index], 1e-4) / image_scale
                delta = torch.tensor(
                    [
                        (gx - probe_x[index]) * scale,
                        (gy - probe_y[index]) * scale,
                        0.0,
                    ],
                    device=device,
                    dtype=dtype,
                )
                spread[index] += normalized * torch.outer(delta, delta)
                weights[index] += normalized

        for index, probe_index in enumerate(probe_indices):
            if weights[index] <= 1e-8:
                continue
            addition = spread[index] / weights[index]
            addition = (addition + addition.mT) * 0.5
            addition += torch.eye(3, device=device, dtype=dtype) * 1e-6
            covariances[probe_index] += addition

    # ------------------------------------------------------------------ #
    # Deprecated / legacy                                                   #
    # ------------------------------------------------------------------ #

    def replicate_tile(
        self,
        gaussians_full,
        tile_y: int,
        tile_x: int,
        probe_idx: int,
        other_indices: List[int],
    ):
        """Legacy Level 0: replicate single probe to all other pixels (dead code)."""
        covs = gaussians_full.covariances
        harmo = gaussians_full.harmonics
        opacs = gaussians_full.opacities
        p_cov = covs[0, probe_idx]
        p_harm = harmo[0, probe_idx]
        p_opac = opacs[0, probe_idx]
        for idx in other_indices:
            covs[0, idx] = p_cov * 1.02
            harmo[0, idx] = p_harm
            opacs[0, idx] = p_opac

    def _dense_interpolate_non_probes(
        self,
        gaussians_full,
        probe_indices: List[int],
        non_probe_map: Dict[Tuple[int, int], int],
    ) -> None:
        """Historical dense interpolation retained only for failure analysis."""
        if self.initial_tile_size != 4 or len(probe_indices) != 4:
            raise ValueError("dense diagnostic supports the default 4x4 tile only")
        covariances = gaussians_full.covariances[0]
        harmonics = gaussians_full.harmonics[0]
        opacities = gaussians_full.opacities[0]
        probe_covariances = covariances[probe_indices].clone()
        probe_harmonics = harmonics[probe_indices].clone()
        probe_opacities = opacities[probe_indices].clone()
        opacity_min = probe_opacities.amin(dim=0)
        opacity_max = probe_opacities.amax(dim=0)
        harmonic_norm = probe_harmonics.reshape(4, -1).norm(dim=1).mean()

        for (local_y, local_x), index in non_probe_map.items():
            y = local_y / 3.0
            x = local_x / 3.0
            weights = torch.tensor(
                [(1 - y) * (1 - x), (1 - y) * x, y * (1 - x), y * x],
                device=covariances.device,
                dtype=covariances.dtype,
            )
            covariances[index] = torch.einsum("n,nij->ij", weights, probe_covariances)
            merged_harmonics = torch.einsum(
                "n,nk->k", weights, probe_harmonics.reshape(4, -1)
            ).reshape(probe_harmonics.shape[1:])
            merged_norm = merged_harmonics.norm().clamp_min(1e-8)
            merged_harmonics *= torch.clamp(harmonic_norm / merged_norm, 0.9, 1.1)
            harmonics[index] = merged_harmonics
            merged_opacity = torch.einsum(
                "n,nk->k", weights, probe_opacities.reshape(4, -1)
            ).reshape(probe_opacities.shape[1:])
            opacities[index] = torch.maximum(
                torch.minimum(merged_opacity, opacity_max), opacity_min
            )

    # ------------------------------------------------------------------ #
    # Main tile processing loop                                             #
    # ------------------------------------------------------------------ #

    def process_all_tiles(
        self,
        gaussians_full,
        gpp: int = 1,
        tile_variances: Dict = None,
        depths=None,
        routing_depths=None,
        feat_norm=None,
        source_opacities: torch.Tensor | None = None,
        source_opacity_certificate_kind: str | None = None,
        tile_trace: List[Dict] | None = None,
    ) -> Tuple["torch.Tensor", Dict]:
        """
        Multi-level SAES v4 tile processing (Dataflow-aligned, L0+L1 only).

        For each tile:
          L0 — feature variance check → weighted moment matching (space+feat)
          L1 — primary-probe depth check → execute 2K(T) lightweight anchors
          Full — no modification

        Non-anchor pixels in an early-stopped tile have their opacity set to 0
        after moment matching. L0 retains K(T); L1 retains up to 2K(T)
        deterministic native anchors.

        Returns:
            modified_mask: bool tensor, True for pixels modified (L0/L1)
            stats:         per-level statistics including effective_gaussians
        """
        gaussian_count = gaussians_full.means.shape[1]
        device = gaussians_full.means.device
        if tile_trace is not None and not isinstance(tile_trace, list):
            raise ValueError("tile_trace must be a list when provided")
        if (
            self._camera_directions is not None
            and self._camera_directions.device != device
        ):
            raise ValueError(
                "camera geometry and Gaussian tensors must share one device"
            )
        if (
            self._camera_directions is not None
            and self._camera_directions.dtype != gaussians_full.means.dtype
        ):
            raise ValueError(
                "camera geometry and Gaussian tensors must share one dtype"
            )
        position_count = self.view_count * self.H * self.W
        expected_gaussians = position_count * self.primitives_per_pixel
        if gaussian_count != expected_gaussians:
            raise ValueError(
                "flattened Gaussian count does not match view/pixel layout: "
                f"got {gaussian_count}, expected {expected_gaussians}"
            )

        routing_depths = depths if routing_depths is None else routing_depths
        (
            certificate_source_opacities,
            certificate_source_status,
        ) = self._prepare_source_opacities_for_deletion_certificate(
            source_opacities,
            source_kind=source_opacity_certificate_kind,
            view_count=self.view_count,
            height=self.H,
            width=self.W,
            primitives_per_pixel=self.primitives_per_pixel,
            device=device,
            dtype=gaussians_full.opacities.dtype,
        )
        modified_mask = torch.zeros(gaussian_count, dtype=torch.bool, device=device)
        for key in self.stats:
            if key == "joint_calibrator":
                if self._joint_calibrator_contract is None:
                    raise RuntimeError("joint calibrator stats have no cost contract")
                self.stats[key] = {
                    **self._joint_calibrator_contract,
                    "selected_head": dict(
                        self._joint_calibrator_contract["selected_head"]
                    ),
                }
                continue
            if key not in {
                "decision_semantics",
                "cross_check_threshold",
                "feature_statistic",
                "depth_statistic",
                "camera_aware_moment_matching",
                "l1_depth_reference",
                "merge_semantics",
                "materialization_guard_enabled",
                "probe_cross_check_enabled",
                "context_safety_guard_enabled",
                "deletion_certificate_required",
                "deletion_certificate_policy",
                "same_budget_dense_oracle_runtime_eligible",
                "multicontext_tangent_enabled",
                "multicontext_tangent_runtime_eligible",
            }:
                self.stats[key] = 0
        self.stats["same_budget_dense_oracle_required_footprint_expansion_max"] = 1.0
        self.stats["same_budget_dense_oracle_failure_reasons"] = {}
        self.stats["multicontext_tangent_fallback_reasons"] = {}
        self.stats["deletion_certificate_source_kind"] = source_opacity_certificate_kind
        self.stats["deletion_certificate_source_status"] = certificate_source_status
        self.stats["deletion_certificate_rejection_reasons"] = {}

        tiles_h = self.H // self.initial_tile_size
        tiles_w = self.W // self.initial_tile_size
        tile_size = self.initial_tile_size
        total_zeroed = 0

        def record_sparse_route(
            level: str,
            retained_positions: List[Tuple[int, int]],
            *,
            same_budget_dense_oracle: bool = False,
        ) -> None:
            """Account for a committed L0/L1 tile without touching descriptors."""
            pixel_count = tile_size * tile_size - len(retained_positions)
            if level == "L0":
                self.stats["level0_tiles"] += 1
                self.stats["level0_pixels"] += pixel_count
                self.stats["l0_representatives"] += (
                    len(retained_positions) * self.primitives_per_pixel
                )
                if same_budget_dense_oracle:
                    self.stats["same_budget_dense_oracle_l0_tiles"] += 1
            elif level == "L1":
                self.stats["level1_tiles"] += 1
                self.stats["level1_pixels"] += pixel_count
                self.stats["l1_lightweight_anchors"] += (
                    len(retained_positions) * self.primitives_per_pixel
                )
                if same_budget_dense_oracle:
                    self.stats["same_budget_dense_oracle_l1_tiles"] += 1
            else:
                raise ValueError("sparse route accounting requires L0 or L1")
            self.stats["pixels_original"] += len(retained_positions)

        def flat_index(view: int, pixel: int, primitive_slot: int) -> int:
            return (
                view * self.H * self.W + pixel
            ) * self.primitives_per_pixel + primitive_slot

        for view in range(self.view_count):
            for th in range(tiles_h):
                for tw in range(tiles_w):
                    self.stats["total_tiles_processed"] += 1
                    tile_y = th * tile_size
                    tile_x = tw * tile_size

                    def indices_for_positions(
                        slot: int, positions: List[Tuple[int, int]]
                    ) -> List[int]:
                        return [
                            flat_index(
                                view,
                                (tile_y + local_y) * self.W + tile_x + local_x,
                                slot,
                            )
                            for local_y, local_x in positions
                        ]

                    def non_anchor_map(
                        slot: int, positions: List[Tuple[int, int]]
                    ) -> Dict[Tuple[int, int], int]:
                        retained = set(positions)
                        return {
                            (local_y, local_x): flat_index(
                                view,
                                (tile_y + local_y) * self.W + tile_x + local_x,
                                slot,
                            )
                            for local_y in range(tile_size)
                            for local_x in range(tile_size)
                            if (local_y, local_x) not in retained
                        }

                    feature_key = (view, th, tw)
                    feature_variance = (
                        tile_variances.get(
                            feature_key, tile_variances.get((th, tw), float("inf"))
                        )
                        if tile_variances is not None
                        else float("inf")
                    )
                    assignment_feature_variance = self._assignment_feature_variance(
                        feature_variance
                    )

                    depth_candidate = None

                    def depth_route_passes() -> bool:
                        nonlocal depth_candidate
                        if depth_candidate is None:
                            depth_candidate = bool(
                                routing_depths is not None
                                and self.check_depth_uniformity(
                                    routing_depths,
                                    th,
                                    tw,
                                    tile_size,
                                    self.H,
                                    self.W,
                                    self.depth_std_threshold,
                                    probe_positions=self.probe_positions,
                                    view_index=view,
                                    relative=False,
                                )
                            )
                        return depth_candidate

                    def materialization_guard_passes(
                        positions: List[Tuple[int, int]], level: str
                    ) -> tuple[bool, bool]:
                        if not self.materialization_guard:
                            return True, False
                        records = []
                        cross_check_records = []
                        context_records = []
                        for slot in range(self.primitives_per_pixel):
                            anchor_indices = indices_for_positions(slot, positions)
                            record = self.probe_materialization_validity(
                                gaussians_full, anchor_indices, level=level
                            )
                            records.append(record)
                            self.stats["guard_anchor_attribute_reads"] += 3 * len(
                                anchor_indices
                            )
                            self.stats["guard_nonprobe_s3_attribute_reads"] += record[
                                "nonprobe_s3_attribute_reads"
                            ]
                            # The leave-one-out test reuses covariance and SH
                            # values already fetched by the attribute guard.
                            # It is always over the fixed primary prefix, even
                            # for a 12-anchor L1 candidate.
                            cross_check_record = {
                                "checked": False,
                                "passed": True,
                                "error": None,
                                "threshold": self.cross_check_threshold,
                                "anchor_count": len(self.probe_positions),
                                "nonprobe_s3_attribute_reads": 0,
                            }
                            if record["passed"] and self.probe_cross_check_enabled:
                                primary_indices = indices_for_positions(
                                    slot, self.probe_positions
                                )
                                validity = self.probe_cross_check_validity(
                                    gaussians_full,
                                    primary_indices,
                                    level=level,
                                )
                                cross_check_record = {"checked": True, **validity}
                                self.stats[f"probe_cross_check_{level.lower()}_checks"] += 1
                                if not validity["passed"]:
                                    self.stats[
                                        f"probe_cross_check_{level.lower()}_rejections"
                                    ] += 1
                            cross_check_records.append(cross_check_record)
                            if self.context_safety_guard:
                                context_records.append(
                                    self.context_coverage_occlusion_safety(
                                        gaussians_full,
                                        anchor_indices,
                                        self._primary_probe_depth_values(
                                            routing_depths,
                                            tile_row=th,
                                            tile_column=tw,
                                            view_index=view,
                                            primitive_slot=slot,
                                        ),
                                        view_index=view,
                                        level=level,
                                    )
                                )
                        cross_check_passed = all(
                            record["passed"] for record in cross_check_records
                        )
                        cross_check_force_full = any(
                            record["checked"] and not record["passed"]
                            for record in cross_check_records
                        )
                        context_passed = all(
                            record["passed"] for record in context_records
                        )
                        context_force_full = any(
                            record["reason"]
                            in {
                                "missing_geometry",
                                "invalid_probe_depth",
                                "invalid_footprint",
                                "center_separation",
                            }
                            for record in context_records
                        )
                        if self.context_safety_guard:
                            self.stats["context_guard_checks"] += 1
                            if not context_passed:
                                self.stats["context_guard_rejections"] += 1
                                reasons = {
                                    record["reason"] for record in context_records
                                }
                                if "missing_geometry" in reasons:
                                    self.stats["context_guard_missing_geometry"] += 1
                                if "invalid_footprint" in reasons:
                                    self.stats["context_guard_invalid_footprint"] += 1
                                if "coverage_spread" in reasons:
                                    self.stats[
                                        "context_guard_coverage_rejections"
                                    ] += 1
                                if "occlusion_span" in reasons:
                                    self.stats[
                                        "context_guard_occlusion_rejections"
                                    ] += 1
                                if "center_separation" in reasons:
                                    self.stats[
                                        "context_guard_center_separation_rejections"
                                    ] += 1
                        passed = (
                            all(record["passed"] for record in records)
                            and cross_check_passed
                            and context_passed
                        )
                        if tile_trace is not None:
                            checked_cross_checks = [
                                record
                                for record in cross_check_records
                                if record["checked"]
                            ]
                            trace_record = {
                                "level": level,
                                "anchor_count": len(positions)
                                * self.primitives_per_pixel,
                                "passed": passed,
                                "covariance_cosine_minimum": min(
                                    record["covariance_cosine_minimum"]
                                    for record in records
                                ),
                                "harmonic_cosine_minimum": min(
                                    record["harmonic_cosine_minimum"]
                                    for record in records
                                ),
                                "opacity_distance_maximum": max(
                                    record["opacity_distance_maximum"]
                                    for record in records
                                ),
                                "nonprobe_s3_attribute_reads": sum(
                                    record["nonprobe_s3_attribute_reads"]
                                    for record in records
                                ),
                                "probe_cross_check": {
                                    "checked": bool(checked_cross_checks),
                                    "passed": (
                                        all(
                                            record["passed"]
                                            for record in cross_check_records
                                        )
                                        if len(checked_cross_checks)
                                        == len(cross_check_records)
                                        else None
                                    ),
                                    "error_max": (
                                        max(
                                            float(record["error"])
                                            for record in checked_cross_checks
                                            if record["error"] is not None
                                        )
                                        if any(
                                            record["error"] is not None
                                            for record in checked_cross_checks
                                        )
                                        else None
                                    ),
                                    "threshold": self.cross_check_threshold,
                                    "primary_anchor_count": len(
                                        self.probe_positions
                                    )
                                    * self.primitives_per_pixel,
                                    "nonprobe_s3_attribute_reads": 0,
                                },
                            }
                            if self.context_safety_guard:
                                trace_record["context_safety"] = {
                                    "passed": context_passed,
                                    "coverage_footprint_ratio": max(
                                        record["coverage_footprint_ratio"]
                                        for record in context_records
                                    ),
                                    "relative_depth_span": max(
                                        record["relative_depth_span"]
                                        for record in context_records
                                    ),
                                    "projected_center_mahalanobis_max": max(
                                        record["projected_center_mahalanobis_max"]
                                        for record in context_records
                                    ),
                                    "center_overlap_passed": all(
                                        record["center_overlap_passed"]
                                        for record in context_records
                                    ),
                                    "reason": next(
                                        (
                                            record["reason"]
                                            for record in context_records
                                            if not record["passed"]
                                        ),
                                        "accepted",
                                    ),
                                }
                            tile_guard_checks.append(trace_record)
                        return passed, context_force_full or cross_check_force_full

                    selected_level = None
                    tile_guard_checks: List[Dict] = []
                    if feature_variance < self.feature_var_threshold:
                        if not self.materialization_guard:
                            selected_level = "L0"
                        else:
                            self.stats["l0_guard_checks"] += 1
                            l0_passed, l0_force_full = materialization_guard_passes(
                                self.probe_positions, "L0"
                            )
                            if l0_passed:
                                selected_level = "L0"
                            else:
                                self.stats["l0_guard_rejections"] += 1
                                # A rejected L0 materialization still follows the
                                # paper's second probe test. It is not silently
                                # conflated with Full until the L1 guard rejects.
                                if not l0_force_full:
                                    self.stats["l1_guard_attempts_after_l0_rejection"] += 1
                                if not l0_force_full and depth_route_passes():
                                    self.stats["l1_guard_checks"] += 1
                                    l1_passed, _ = materialization_guard_passes(
                                        self.lightweight_positions, "L1"
                                    )
                                    if l1_passed:
                                        selected_level = "L1"
                                    else:
                                        self.stats["l1_guard_rejections"] += 1
                    elif depth_route_passes():
                        if not self.materialization_guard:
                            selected_level = "L1"
                        else:
                            self.stats["l1_guard_checks"] += 1
                            l1_passed, _ = materialization_guard_passes(
                                self.lightweight_positions, "L1"
                            )
                            if l1_passed:
                                selected_level = "L1"
                            else:
                                self.stats["l1_guard_rejections"] += 1
                    if tile_trace is not None:
                        tile_trace.append(
                            {
                                "view_index": view,
                                "tile_row": th,
                                "tile_column": tw,
                                "feature_variance": float(feature_variance),
                                "feature_candidate": bool(
                                    feature_variance < self.feature_var_threshold
                                ),
                                "depth_candidate": depth_candidate,
                                "guard_enabled": self.materialization_guard,
                                "guard_checks": tile_guard_checks,
                                "routing_level_before_materialization": (
                                    selected_level
                                    if selected_level is not None
                                    else "Full"
                                ),
                            }
                        )

                    if selected_level is not None:
                        retained_positions = (
                            self.probe_positions
                            if selected_level == "L0"
                            else self.lightweight_positions
                        )
                        if (
                            self.require_deletion_certificate
                            and self.materialization == "representative"
                        ):
                            certificate_records = []
                            for slot in range(self.primitives_per_pixel):
                                anchors = indices_for_positions(
                                    slot, retained_positions
                                )
                                nonprobes = list(
                                    non_anchor_map(slot, retained_positions).values()
                                )
                                certificate_records.append(
                                    self.exact_source_opacity_zero_deletion_certificate(
                                        gaussians_full,
                                        certificate_source_opacities,
                                        source_status=certificate_source_status,
                                        anchor_indices=anchors,
                                        nonprobe_indices=nonprobes,
                                        level=selected_level,
                                    )
                                )
                            self.stats["deletion_certificate_checks"] += 1
                            self.stats[
                                "deletion_certificate_anchor_s3_opacity_reads"
                            ] += sum(
                                record["anchor_s3_opacity_reads"]
                                for record in certificate_records
                            )
                            self.stats[
                                "deletion_certificate_nonprobe_s3_attribute_reads"
                            ] += sum(
                                record["nonprobe_s3_attribute_reads"]
                                for record in certificate_records
                            )
                            certificate_passed = all(
                                record["passed"] for record in certificate_records
                            )
                            certificate_reason = next(
                                (
                                    record["reason"]
                                    for record in certificate_records
                                    if not record["passed"]
                                ),
                                "exact_zero_alpha",
                            )
                            certificate_trace = {
                                "policy": DELETION_CERTIFICATE_POLICY,
                                "source_status": certificate_source_status,
                                "passed": certificate_passed,
                                "reason": certificate_reason,
                                "anchor_count": sum(
                                    record["anchor_count"]
                                    for record in certificate_records
                                ),
                                "nonprobe_count": sum(
                                    record["nonprobe_count"]
                                    for record in certificate_records
                                ),
                                "anchor_s3_opacity_reads": sum(
                                    record["anchor_s3_opacity_reads"]
                                    for record in certificate_records
                                ),
                                "nonprobe_s3_attribute_reads": sum(
                                    record["nonprobe_s3_attribute_reads"]
                                    for record in certificate_records
                                ),
                            }
                            if tile_trace is not None:
                                tile_trace[-1]["deletion_certificate"] = certificate_trace
                            if certificate_passed:
                                # A source-certified zero-alpha output is
                                # already absent from the rasterizer. Retained
                                # anchors stay bit-identical, so no moment
                                # matching or non-probe Stage-3 read occurs.
                                certified_zeroed = 0
                                for slot in range(self.primitives_per_pixel):
                                    for index in non_anchor_map(
                                        slot, retained_positions
                                    ).values():
                                        modified_mask[index] = True
                                        certified_zeroed += 1
                                total_zeroed += certified_zeroed
                                self.stats[
                                    "deletion_certificate_accepted_tiles"
                                ] += 1
                                self.stats[
                                    "deletion_certificate_zero_opacity_gaussians"
                                ] += certified_zeroed
                                record_sparse_route(
                                    selected_level, retained_positions
                                )
                                if tile_trace is not None:
                                    tile_trace[-1]["final_route"] = selected_level
                                continue

                            # The only sound certificate declined. Do not use
                            # anchor similarity as a substitute for skipped
                            # coverage, occlusion, or optical contribution.
                            self.stats["deletion_certificate_rejected_tiles"] += 1
                            reasons = self.stats[
                                "deletion_certificate_rejection_reasons"
                            ]
                            reasons[certificate_reason] = (
                                reasons.get(certificate_reason, 0) + 1
                            )
                            self.stats["uncertified_deletion_fallback_tiles"] += 1
                            self.stats["full_tiles"] += 1
                            self.stats["full_stage3_gaussians"] += (
                                tile_size * tile_size * self.primitives_per_pixel
                            )
                            self.stats["pixels_original"] += tile_size * tile_size
                            if tile_trace is not None:
                                tile_trace[-1]["final_route"] = "Full"
                            continue
                        if (
                            self.materialization
                            in (
                                "conditional-optical-mass-diagnostic",
                                "conditional-projected-optical-mass-diagnostic",
                            )
                            and self.primitives_per_pixel != 1
                        ):
                            # The mass rule is defined only for the submitted
                            # one-opacity-per-primitive layout.  Preserve the
                            # tile as Full instead of inventing a vector-alpha rule.
                            self.stats["conditional_mass_fallback_tiles"] += 1
                            self.stats["full_tiles"] += 1
                            self.stats["full_stage3_gaussians"] += (
                                tile_size * tile_size * self.primitives_per_pixel
                            )
                            self.stats["pixels_original"] += tile_size * tile_size
                            if tile_trace is not None:
                                tile_trace[-1]["final_route"] = "Full"
                            continue
                        fallback_to_full = False
                        is_assignment_consensus = self.materialization == (
                            "assignment-consensus-adapter-pseudo-descriptor-diagnostic"
                        )
                        is_same_budget_dense_oracle = self.materialization == (
                            "same-budget-dense-oracle-diagnostic"
                        )
                        consensus_plans = []
                        dense_oracle_plans = []
                        joint_calls_before_tile = (
                            self.stats["joint_calibrator"]["calls"]
                            if self.joint_calibrator is not None
                            else 0
                        )
                        if is_same_budget_dense_oracle:
                            # This diagnostic reads each dense Stage-3 slot
                            # after a route candidate is fixed, including a
                            # candidate that subsequently fails closed.
                            self.stats[
                                "same_budget_dense_oracle_full_stage3_reads"
                            ] += tile_size * tile_size * self.primitives_per_pixel
                        for slot in range(self.primitives_per_pixel):
                            if self.materialization == "dense-diagnostic":
                                materialized_positions = self.probe_positions
                            else:
                                materialized_positions = retained_positions
                            probes = indices_for_positions(slot, materialized_positions)
                            non_probes = non_anchor_map(slot, materialized_positions)
                            if self.materialization in (
                                "representative",
                                "transmittance-diagnostic",
                                "virtual-reconstruction-diagnostic",
                                "assignment-consensus-adapter-pseudo-descriptor-diagnostic",
                                "same-budget-dense-oracle-diagnostic",
                                "conditional-anchor-transport-diagnostic",
                                "conditional-adapter-offset-transport-diagnostic",
                                "conditional-adapter-offset-attribute-transport-diagnostic",
                                "conditional-optical-mass-diagnostic",
                                "conditional-projected-optical-mass-diagnostic",
                                "multicontext-tangent-plane-diagnostic",
                            ):
                                # L1 routing uses only primary probes. After it
                                # succeeds, its deterministic 2K positions are
                                # charged native S2/S3 outputs; only the rest
                                # of the tile is skipped.
                                moment_result = self._weighted_moment_match(
                                    gaussians_full,
                                    probes,
                                    non_probes,
                                    feat_norm,
                                    depths,
                                    selected_level,
                                    tile_y,
                                    tile_x,
                                    view_index=view,
                                    primitive_slot=slot,
                                    feature_variance=assignment_feature_variance,
                                    retained_positions=materialized_positions,
                                    opacity_aggregation=(
                                        "assignment-weighted-optical-depth"
                                        if self.materialization
                                        == "transmittance-diagnostic"
                                        else "range-constrained-average"
                                    ),
                                    output_style=(
                                        "virtual-reconstruction"
                                        if self.materialization
                                        == "virtual-reconstruction-diagnostic"
                                        else "assignment-consensus-adapter-pseudo-descriptor"
                                        if self.materialization
                                        == "assignment-consensus-adapter-pseudo-descriptor-diagnostic"
                                        else "same-budget-dense-oracle"
                                        if self.materialization
                                        == "same-budget-dense-oracle-diagnostic"
                                        else "representative"
                                    ),
                                    merge_semantics=self.merge_semantics,
                                    defer_assignment_consensus_plan=(
                                        is_assignment_consensus
                                    ),
                                    defer_same_budget_dense_oracle_plan=(
                                        is_same_budget_dense_oracle
                                    ),
                                )
                                if (
                                    is_assignment_consensus
                                    or is_same_budget_dense_oracle
                                ):
                                    fallback_to_full, deferred_plan = moment_result
                                    if not fallback_to_full:
                                        # A tile whose anchors cover every
                                        # position has no skipped output. That
                                        # is a valid no-op, not a failed plan.
                                        if deferred_plan is None and non_probes:
                                            raise RuntimeError(
                                                "deferred SAES diagnostic plan is missing"
                                            )
                                        if deferred_plan is not None:
                                            if is_assignment_consensus:
                                                consensus_plans.append(deferred_plan)
                                            else:
                                                dense_oracle_plans.append(deferred_plan)
                                else:
                                    fallback_to_full = moment_result
                                if fallback_to_full:
                                    break
                            elif self.materialization == "probe-spread-diagnostic":
                                self._spread_probe_covariances(
                                    gaussians_full,
                                    probes,
                                    non_probes,
                                    feat_norm,
                                    depths,
                                    selected_level,
                                    tile_y,
                                    tile_x,
                                    view_index=view,
                                    primitive_slot=slot,
                                    retained_positions=materialized_positions,
                                )
                            else:
                                self._dense_interpolate_non_probes(
                                    gaussians_full, probes, non_probes
                                )
                            if is_assignment_consensus or is_same_budget_dense_oracle:
                                # Plans are committed only after every slot
                                # validates, so no failed tile needs a rollback
                                # snapshot before it remains Full.
                                continue
                            preserves_virtual_outputs = self.materialization in (
                                "dense-diagnostic",
                                "virtual-reconstruction-diagnostic",
                                "assignment-consensus-adapter-pseudo-descriptor-diagnostic",
                            )
                            for index in non_probes.values():
                                if not preserves_virtual_outputs:
                                    # Deleted descriptors need no prior alpha.
                                    # Assignment avoids reading skipped S3
                                    # opacity before a target-free output commits.
                                    gaussians_full.opacities[0, index] = 0.0
                                modified_mask[index] = True
                            if not preserves_virtual_outputs:
                                total_zeroed += len(non_probes)
                        if is_assignment_consensus and not fallback_to_full:
                            for consensus_plan in consensus_plans:
                                self._commit_assignment_consensus_plan(
                                    gaussians_full, consensus_plan
                                )
                                for output_index in consensus_plan["output_indices"]:
                                    modified_mask[output_index] = True
                        if is_same_budget_dense_oracle and not fallback_to_full:
                            for dense_oracle_plan in dense_oracle_plans:
                                self._commit_same_budget_dense_oracle_plan(
                                    gaussians_full, dense_oracle_plan
                                )
                                for output_index in dense_oracle_plan["zero_indices"]:
                                    gaussians_full.opacities[0, output_index] *= 0.0
                                    modified_mask[output_index] = True
                                    total_zeroed += 1
                            self.stats["same_budget_dense_oracle_tiles"] += 1
                            if tile_trace is not None:
                                tile_trace[-1]["oracle_materialization"] = "compressed"
                        if fallback_to_full:
                            if (
                                self.joint_calibrator is not None
                                and self.stats["joint_calibrator"]["calls"]
                                != joint_calls_before_tile
                            ):
                                raise RuntimeError(
                                    "joint calibrator cannot fall back after retained updates"
                                )
                            if self.merge_semantics in (
                                "conditional-optical-mass",
                                "conditional-projected-optical-mass",
                            ):
                                self.stats["conditional_mass_fallback_tiles"] += 1
                            elif self.merge_semantics in (
                                "conditional-adapter-offset-transport",
                                "conditional-adapter-offset-attribute-transport",
                            ):
                                self.stats[
                                    "adapter_offset_transport_fallback_tiles"
                                ] += 1
                            elif self.merge_semantics == (
                                "assignment-consensus-adapter-pseudo-descriptor"
                            ):
                                self.stats["assignment_consensus_fallback_tiles"] += 1
                                if selected_level == "L0":
                                    self.stats[
                                        "assignment_consensus_l0_fallback_tiles"
                                    ] += 1
                                else:
                                    self.stats[
                                        "assignment_consensus_l1_fallback_tiles"
                                    ] += 1
                            elif self.merge_semantics == "same-budget-dense-oracle":
                                self.stats[
                                    "same_budget_dense_oracle_fallback_tiles"
                                ] += 1
                                reason = (
                                    getattr(
                                        self,
                                        "_same_budget_dense_oracle_last_failure",
                                        None,
                                    )
                                    or "unknown"
                                )
                                reasons = self.stats[
                                    "same_budget_dense_oracle_failure_reasons"
                                ]
                                reasons[reason] = reasons.get(reason, 0) + 1
                                if tile_trace is not None:
                                    tile_trace[-1]["oracle_materialization"] = (
                                        "full-fallback"
                                    )
                                    tile_trace[-1]["oracle_failure_reason"] = reason
                            self.stats["full_tiles"] += 1
                            self.stats["full_stage3_gaussians"] += (
                                tile_size * tile_size * self.primitives_per_pixel
                            )
                            self.stats["pixels_original"] += tile_size * tile_size
                            continue

                        record_sparse_route(
                            selected_level,
                            retained_positions,
                            same_budget_dense_oracle=is_same_budget_dense_oracle,
                        )
                        if tile_trace is not None:
                            tile_trace[-1]["final_route"] = selected_level
                        continue

                    self.stats["full_tiles"] += 1
                    self.stats["full_stage3_gaussians"] += (
                        tile_size * tile_size * self.primitives_per_pixel
                    )
                    self.stats["pixels_original"] += tile_size * tile_size
                    if tile_trace is not None:
                        tile_trace[-1]["final_route"] = "Full"

        total_tiles = max(1, self.stats["total_tiles_processed"])
        self.stats["total_modified_pixels"] = (
            self.stats["level0_pixels"] + self.stats["level1_pixels"]
        )
        self.stats["zeroed_gaussians"] = total_zeroed
        self.stats["effective_gaussians"] = max(0, gaussian_count - total_zeroed)
        self.stats["full_s2_evaluations"] = (
            self.stats["total_tiles_processed"]
            * tile_size
            * tile_size
            * self.primitives_per_pixel
            * self.num_depth_candidates
        )
        self.stats["executed_s2_evaluations"] = (
            self.stats["l0_representatives"]
            + self.stats["l1_lightweight_anchors"]
            + self.stats["full_stage3_gaussians"]
        ) * self.num_depth_candidates
        self.stats["s2_evaluations_available"] = True

        self.stats["level0_ratio"] = self.stats["level0_tiles"] / total_tiles
        self.stats["level1_ratio"] = self.stats["level1_tiles"] / total_tiles
        self.stats["full_ratio"] = self.stats["full_tiles"] / total_tiles
        self.stats["early_stop_ratio"] = 1.0 - self.stats["full_ratio"]
        self.stats["modification_ratio"] = (
            self.stats["total_modified_pixels"] / position_count
            if position_count > 0
            else 0.0
        )

        # Backward-compatible keys
        self.stats["early_stop_phase1"] = (
            self.stats["level0_tiles"] + self.stats["level1_tiles"]
        )
        self.stats["early_stop_phase2"] = 0
        self.stats["full_processed"] = self.stats["full_tiles"]
        self.stats["pixels_interpolated"] = self.stats["total_modified_pixels"]
        self.stats["interpolation_ratio"] = self.stats["modification_ratio"]

        return modified_mask, self.stats


# -------------------------------------------------------------------- #
# Public API                                                              #
# -------------------------------------------------------------------- #


def apply_progressive_saes(
    gaussians_full,
    H: int,
    W: int,
    tile_size: int = 4,
    gpp: int = 1,
    feature_var_threshold: float = None,
    depth_std_threshold: float = None,
    features=None,
    depths=None,
    cross_check_threshold: float = 0.015,
    feat_norm=None,  # pre-computed [C, H, W]; computed internally if None
    collect_continue_pixels: bool = False,
    view_count: int = None,
    materialization: str = "representative",
    decision_semantics: str = "current",
    beta_x: float = 0.50,
    beta_f: float = 0.10,
    beta_d: float = 1.00,
    num_depth_candidates: int = 1,
    context_extrinsics: torch.Tensor | None = None,
    context_intrinsics: torch.Tensor | None = None,
    ray_depth_mode: str = "euclidean",
    depth_routing_semantics: str = "metric-depth-standard-deviation",
    depth_near: torch.Tensor | None = None,
    depth_far: torch.Tensor | None = None,
    materialization_guard: bool = True,
    context_safety_guard: bool = False,
    require_deletion_certificate: bool = False,
    source_opacities: torch.Tensor | None = None,
    source_opacity_certificate_kind: str | None = None,
    joint_calibrator: Any | None = None,
    joint_calibrator_model: str | None = None,
    joint_calibrator_selected_head: Mapping[str, Any] | None = None,
    offline_joint_calibration_capture: Any | None = None,
    tile_trace: List[Dict] | None = None,
) -> Tuple["torch.Tensor", Dict, List]:
    """
    Apply progressive SAES v4 (Dataflow-aligned, L0+L1) to Gaussians.

    Two-level progressive early-stopping:
      - L0: feature variance check → weighted moment matching (space+feat)
      - L1: primary-probe depth check → execute 2K(T) anchor moment matching
      - Full: no modification

    Non-anchor opacities are zeroed in early-stopped tiles. L0 retains K(T)
    representatives and L1 retains up to 2K(T) deterministic native anchors.

    Args:
        features:  S1 features [B, V, C, H, W] for L0 classification
        depths:    S2 depths   [B, V, H, W] for L1 check
        feat_norm: pre-normalised features [C, H, W] (optional)

    Returns:
        modified_mask:   bool tensor (True = primitive was changed or removed)
        stats:           per-level statistics (incl. effective_gaussians)
        continue_pixels: list of (y, x, pixel_idx) for unmodified pixels
    """
    if depth_routing_semantics not in (
        "metric-depth-standard-deviation",
        "inverse-depth-candidate-coordinate-standard-deviation",
    ):
        raise ValueError(
            f"unsupported depth routing semantics: {depth_routing_semantics}"
        )
    if (depth_near is None) != (depth_far is None):
        raise ValueError("depth_near and depth_far must be provided together")
    if (
        depth_routing_semantics
        == "inverse-depth-candidate-coordinate-standard-deviation"
        and (depth_near is None or depth_far is None)
    ):
        raise ValueError(
            "inverse-depth candidate routing requires depth_near and depth_far"
        )

    gaussian_count = gaussians_full.means.shape[1]
    if view_count is None:
        if features is not None and hasattr(features, "dim") and features.dim() == 5:
            view_count = int(features.shape[1])
        elif depths is not None and hasattr(depths, "dim") and depths.dim() >= 4:
            view_count = int(depths.shape[1])
        else:
            view_count = 1
    position_count = view_count * H * W
    if gaussian_count % position_count != 0:
        raise ValueError(
            "Gaussian count is not divisible by view_count * H * W: "
            f"{gaussian_count} vs {position_count}"
        )
    primitives_per_pixel = gaussian_count // position_count

    tile_variances = None
    _feat_norm = feat_norm  # use caller-supplied if available

    if features is not None and hasattr(features, "shape"):
        _tv, _fn = ProgressiveSAES.classify_tiles_by_features(
            features,
            H,
            W,
            tile_size,
            threshold=(
                feature_var_threshold if feature_var_threshold is not None else 0.012
            ),
            per_view=True,
            statistic=(
                "raw-probe-vector-variance"
                if decision_semantics == "probe-vector-first-hit"
                else "raw-probe-mean-channel-variance"
                if decision_semantics
                in {
                    "probe-channel-variance-first-hit",
                    "paper-probe-feature-variance-first-hit",
                }
                else "normalized-probe-vector-standard-deviation"
                if decision_semantics
                in {
                    "probe-normalized-std-first-hit",
                    PAPER_NORMALIZED_FEATURE_DECISION_SEMANTICS,
                }
                else "normalized-probe-total-variance"
            ),
        )
        tile_variances = _tv
        if _feat_norm is None:
            _feat_norm = _fn

    saes = ProgressiveSAES(
        H,
        W,
        tile_size,
        feature_var_threshold=feature_var_threshold,
        depth_std_threshold=depth_std_threshold,
        cross_check_threshold=cross_check_threshold,
        view_count=view_count,
        primitives_per_pixel=primitives_per_pixel,
        materialization=materialization,
        decision_semantics=decision_semantics,
        beta_x=beta_x,
        beta_f=beta_f,
        beta_d=beta_d,
        num_depth_candidates=num_depth_candidates,
        context_extrinsics=context_extrinsics,
        context_intrinsics=context_intrinsics,
        ray_depth_mode=ray_depth_mode,
        materialization_guard=materialization_guard,
        context_safety_guard=context_safety_guard,
        require_deletion_certificate=require_deletion_certificate,
        joint_calibrator=joint_calibrator,
        joint_calibrator_model=joint_calibrator_model,
        joint_calibrator_selected_head=joint_calibrator_selected_head,
        offline_joint_calibration_capture=offline_joint_calibration_capture,
    )
    routing_depths = (
        ProgressiveSAES.inverse_depth_candidate_coordinate(
            depths, near=depth_near, far=depth_far
        )
        if depths is not None
        and depth_routing_semantics
        == "inverse-depth-candidate-coordinate-standard-deviation"
        else depths
    )
    saes.stats["depth_statistic"] = depth_routing_semantics
    modified_mask, stats = saes.process_all_tiles(
        gaussians_full,
        gpp,
        tile_variances=tile_variances,
        depths=depths,
        routing_depths=routing_depths,
        feat_norm=_feat_norm,
        source_opacities=source_opacities,
        source_opacity_certificate_kind=source_opacity_certificate_kind,
        tile_trace=tile_trace,
    )

    continue_pixels = []
    if collect_continue_pixels:
        indices = torch.nonzero(~modified_mask, as_tuple=False).flatten().cpu().tolist()
        for idx in indices:
            pixel_idx = idx // primitives_per_pixel
            y, x = divmod(pixel_idx, W)
            y %= H
            if x < W:
                continue_pixels.append((y, x, pixel_idx))

    return modified_mask, stats, continue_pixels
