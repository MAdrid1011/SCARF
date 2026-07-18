#!/usr/bin/env python3
"""Optimize fixed K/2K SAES representatives against dense renders only."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.saes_dependency_audit import _context_on_device, remove_target_rgb
from scripts.saes_same_budget_dense_oracle import (
    MATERIALIZATION,
    SEED,
    _oracle_output_accounting,
)
from scripts.saes_selected_output_quality_gate import (
    _image_equivalence,
    _mean_metrics,
    _retain_renderable_gaussians,
    _sha256_mask,
    _target_camera_inputs,
    _view_metrics,
)
from scripts.saes_selected_output_replay_audit import strict_fp32_convolution_execution
from scripts.saes_target_free_materialization_audit import (
    _capture_encoder_execution,
    _clone_gaussians,
)


KIND = "saes_same_budget_render_teacher_oracle"
STEPS = 128
LEARNING_RATE = 0.02
GRADIENT_CLIP_NORM = 1.0
PARAMETER_EPSILON = 1e-5
CHOLESKY_DIAGONAL_MULTIPLIER = 2.0
ATTRIBUTE_NAMES = ("means", "covariances", "harmonics", "opacities")


def _sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().to(device="cpu")
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _sha256_tensors(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name].detach().contiguous().to(device="cpu")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _camera_metadata_sha256(target: dict[str, torch.Tensor]) -> str:
    return _sha256_tensors(
        {key: target[key] for key in ("extrinsics", "intrinsics", "near", "far")}
    )


def _representative_mask(
    modified: torch.Tensor, *, views: int, height: int, width: int
) -> torch.Tensor:
    """Recover exactly the K/2K outputs in tiles that were compressed."""
    from scripts.saes_diagnostics import representative_indices

    indices = representative_indices(
        modified,
        view_count=views,
        height=height,
        width=width,
        tile_size=4,
        primitives_per_pixel=1,
    )
    result = torch.zeros_like(modified)
    result[indices] = True
    return result


def _producer_tile_indices(
    representative_indices: torch.Tensor, *, height: int, width: int
) -> torch.Tensor:
    """Return the dense 4x4 producer tile for every representative slot."""
    if representative_indices.ndim != 1 or representative_indices.numel() < 1:
        raise ValueError("representative indices must be a nonempty vector")
    per_view = height * width
    view = representative_indices // per_view
    pixel = representative_indices % per_view
    row = pixel // width
    column = pixel % width
    origin_row = row // 4 * 4
    origin_column = column // 4 * 4
    if bool((origin_row + 3 >= height).any()) or bool((origin_column + 3 >= width).any()):
        raise ValueError("representative does not belong to a complete 4x4 tile")
    offsets = torch.tensor(
        [(local_row, local_column) for local_row in range(4) for local_column in range(4)],
        dtype=torch.long,
        device=representative_indices.device,
    )
    return (
        view[:, None] * per_view
        + (origin_row[:, None] + offsets[None, :, 0]) * width
        + origin_column[:, None]
        + offsets[None, :, 1]
    )


def _stable_cholesky(covariances: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Project symmetric inputs to PSD before a deterministic Cholesky factor."""
    symmetric = (covariances + covariances.mT) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    if not bool(torch.isfinite(eigenvalues).all()):
        raise RuntimeError("oracle covariance has non-finite eigenvalues")
    floor = torch.as_tensor(epsilon, device=covariances.device, dtype=covariances.dtype)
    positive = eigenvectors @ torch.diag_embed(eigenvalues.clamp_min(floor)) @ eigenvectors.mT
    return torch.linalg.cholesky((positive + positive.mT) * 0.5)


def _inverse_sigmoid(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(value.clamp(PARAMETER_EPSILON, 1.0 - PARAMETER_EPSILON))


class BoundedRepresentativeParameters(torch.nn.Module):
    """A diagnostic-only bounded parameterization with immutable Full slots."""

    def __init__(
        self,
        compact: Any,
        dense: Any,
        representative_global: torch.Tensor,
        representative_local: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> None:
        super().__init__()
        if representative_global.numel() != representative_local.numel() or not representative_global.numel():
            raise ValueError("representative index maps must be nonempty and aligned")
        self.gaussian_type = type(compact)
        self.register_buffer("representative_local", representative_local.to(dtype=torch.long))
        self.register_buffer("representative_global", representative_global.to(dtype=torch.long))
        self.register_buffer("base_means", compact.means.detach().clone())
        self.register_buffer("base_covariances", compact.covariances.detach().clone())
        self.register_buffer("base_harmonics", compact.harmonics.detach().clone())
        self.register_buffer("base_opacities", compact.opacities.detach().clone())

        base_means = self.base_means[0, self.representative_local]
        base_covariances = self.base_covariances[0, self.representative_local]
        base_harmonics = self.base_harmonics[0, self.representative_local]
        base_opacities = self.base_opacities[0, self.representative_local]
        opacity_flat = base_opacities.reshape(base_opacities.shape[0], -1)
        if opacity_flat.shape[1] != 1:
            raise ValueError("render teacher oracle requires scalar opacity")

        tile_indices = _producer_tile_indices(
            self.representative_global, height=height, width=width
        )
        source_means = dense.means[0, tile_indices]
        source_harmonics = dense.harmonics[0, tile_indices]
        source_covariances = dense.covariances[0, tile_indices]
        if not all(
            bool(torch.isfinite(value).all())
            for value in (source_means, source_harmonics, source_covariances)
        ):
            raise RuntimeError("dense producer tile has non-finite oracle bounds")

        self.register_buffer(
            "mean_bound",
            (source_means - base_means[:, None]).abs().amax(dim=1).clamp_min(1e-5),
        )
        self.register_buffer(
            "harmonic_bound",
            (source_harmonics - base_harmonics[:, None]).abs().amax(dim=1).clamp_min(1e-5),
        )
        base_cholesky = _stable_cholesky(base_covariances, PARAMETER_EPSILON)
        source_cholesky = _stable_cholesky(source_covariances, PARAMETER_EPSILON)
        factor_bound = torch.maximum(
            source_cholesky.abs().amax(dim=1), base_cholesky.abs()
        ).clamp_min(PARAMETER_EPSILON)
        diagonal_upper = (
            CHOLESKY_DIAGONAL_MULTIPLIER
            * torch.diagonal(factor_bound, dim1=-2, dim2=-1)
        ).clamp_min(PARAMETER_EPSILON * 2.0)
        diagonal_fraction = torch.diagonal(base_cholesky, dim1=-2, dim2=-1) / diagonal_upper

        self.register_buffer("base_rep_means", base_means)
        self.register_buffer("base_rep_harmonics", base_harmonics)
        self.register_buffer("base_cholesky", base_cholesky)
        self.register_buffer("cholesky_offdiagonal_bound", factor_bound)
        self.register_buffer("cholesky_diagonal_upper", diagonal_upper)
        self.mean_delta = torch.nn.Parameter(torch.zeros_like(base_means))
        self.harmonic_delta = torch.nn.Parameter(torch.zeros_like(base_harmonics))
        self.cholesky_delta = torch.nn.Parameter(torch.zeros_like(base_cholesky))
        self.opacity_logits = torch.nn.Parameter(_inverse_sigmoid(opacity_flat[:, 0]))
        self.cholesky_diagonal_logits = torch.nn.Parameter(
            _inverse_sigmoid(diagonal_fraction)
        )

    def representative_values(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        means = self.base_rep_means + self.mean_bound * torch.tanh(self.mean_delta)
        harmonics = self.base_rep_harmonics + self.harmonic_bound * torch.tanh(
            self.harmonic_delta
        )
        lower = torch.tril(
            self.base_cholesky
            + self.cholesky_offdiagonal_bound * torch.tanh(self.cholesky_delta),
            diagonal=-1,
        )
        diagonal = PARAMETER_EPSILON + (
            self.cholesky_diagonal_upper - PARAMETER_EPSILON
        ) * torch.sigmoid(self.cholesky_diagonal_logits)
        cholesky = lower + torch.diag_embed(diagonal)
        covariances = cholesky @ cholesky.mT
        opacity = PARAMETER_EPSILON + (1.0 - 2.0 * PARAMETER_EPSILON) * torch.sigmoid(
            self.opacity_logits
        )
        return means, covariances, harmonics, opacity

    def compose(self) -> Any:
        means, covariances, harmonics, opacity = self.representative_values()
        opacities = self.base_opacities.index_copy(
            1, self.representative_local, opacity.reshape_as(
                self.base_opacities[0, self.representative_local]
            ).unsqueeze(0)
        )
        return self.gaussian_type(
            means=self.base_means.index_copy(1, self.representative_local, means.unsqueeze(0)),
            covariances=self.base_covariances.index_copy(
                1, self.representative_local, covariances.unsqueeze(0)
            ),
            harmonics=self.base_harmonics.index_copy(
                1, self.representative_local, harmonics.unsqueeze(0)
            ),
            opacities=opacities,
        )

    def bound_hashes(self) -> dict[str, str]:
        return {
            "mean_bound": _sha256_tensor(self.mean_bound),
            "harmonic_bound": _sha256_tensor(self.harmonic_bound),
            "cholesky_offdiagonal_bound": _sha256_tensor(self.cholesky_offdiagonal_bound),
            "cholesky_diagonal_upper": _sha256_tensor(self.cholesky_diagonal_upper),
        }

    def parameter_hashes(self) -> dict[str, str]:
        return {
            name: _sha256_tensor(parameter)
            for name, parameter in self.named_parameters()
        }

    def assert_immutable_slots(self, candidate: Any) -> None:
        frozen = torch.ones(
            self.base_means.shape[1], dtype=torch.bool, device=self.base_means.device
        )
        frozen[self.representative_local] = False
        for name in ATTRIBUTE_NAMES:
            torch.testing.assert_close(
                getattr(candidate, name)[:, frozen], getattr(self, f"base_{name}")[:, frozen]
            )


def _render_one(model: Any, gaussians: Any, target: dict[str, torch.Tensor], view: int, image_shape: tuple[int, int]) -> torch.Tensor:
    """Render one target camera, retaining a differentiable path when needed."""
    output = model.decoder.forward(
        gaussians,
        target["extrinsics"][:, view : view + 1],
        target["intrinsics"][:, view : view + 1],
        target["near"][:, view : view + 1],
        target["far"][:, view : view + 1],
        image_shape,
        depth_mode=None,
    )
    return output.color[0, 0]


def _teacher_fidelity(images: torch.Tensor, teacher: torch.Tensor) -> dict[str, Any]:
    if images.shape != teacher.shape:
        raise RuntimeError("teacher and student renders differ in shape")
    per_view = _view_metrics(images, teacher)
    mse = (images - teacher).square().mean(dim=(1, 2, 3))
    return {
        "teacher_kind": "dense-baseline-render",
        "mean": _mean_metrics(per_view),
        "mse": {"mean": float(mse.mean().item()), "maximum": float(mse.max().item())},
        "views": per_view,
    }


def _parameter_gradient_report(module: BoundedRepresentativeParameters) -> dict[str, dict[str, Any]]:
    report = {}
    for name, parameter in module.named_parameters():
        gradient = parameter.grad
        report[name] = {
            "present": gradient is not None,
            "finite": gradient is not None and bool(torch.isfinite(gradient).all()),
            "nonzero": gradient is not None and bool(torch.count_nonzero(gradient)),
        }
    if not all(item["finite"] and item["nonzero"] for item in report.values()):
        raise RuntimeError("render oracle did not produce finite nonzero gradients for every family")
    return report


def _save_oracle_artifacts(
    output_dir: Path,
    *,
    representative_global: torch.Tensor,
    module: BoundedRepresentativeParameters,
    teacher: torch.Tensor,
) -> dict[str, dict[str, Any]]:
    values = module.representative_values()
    names = ("means", "covariances", "harmonics", "opacities")
    representative_path = output_dir / "optimized_representatives.pt"
    teacher_path = output_dir / "dense_render_teacher.pt"
    torch.save(
        {
            "global_indices": representative_global.detach().cpu(),
            **{name: value.detach().cpu() for name, value in zip(names, values)},
        },
        representative_path,
    )
    torch.save(teacher.detach().cpu(), teacher_path)
    from scripts.result_record import cached_sha256_file

    return {
        "optimized_representatives": {
            "path": representative_path.name,
            "sha256": cached_sha256_file(representative_path),
            "count": int(representative_global.numel()),
        },
        "dense_render_teacher": {
            "path": teacher_path.name,
            "sha256": cached_sha256_file(teacher_path),
            "shape": list(teacher.shape),
        },
    }


def collect_render_teacher_oracle(*, device: torch.device, output_dir: Path) -> dict[str, Any]:
    """Run the fixed bounded post-hoc render-teacher capacity diagnostic."""
    from scripts.ae_config import resolve_claim_selection, resolve_experiment
    from scripts.demo import load_model_and_data
    from scripts.result_record import cached_sha256_file, source_identity
    from scripts.saes_selected_output_quality_gate import _apply_fixed_saes

    experiment = resolve_experiment("transplat", "dl3dv", ROOT)
    selection = resolve_claim_selection("transplat", "dl3dv", ROOT)
    model, batch, _cfg, loaded_device = load_model_and_data(
        "transplat",
        dataset_name="dl3dv",
        checkpoint_path=experiment.checkpoint,
        dataset_root=experiment.dataset_root,
        evaluation_index=selection.index_path,
        experiment_name=experiment.experiment,
        hydra_overrides=experiment.hydra_overrides,
        device=device,
        num_samples=1,
        sample_index=0,
    )
    model.eval()
    native_target_rgb_loaded = remove_target_rgb(batch)
    target = _target_camera_inputs(batch, loaded_device)
    context = _context_on_device(batch, loaded_device)
    _, views, _, height, width = context["image"].shape
    if context["image"].shape[0] != 1:
        raise RuntimeError("render teacher oracle requires batch size one")

    with strict_fp32_convolution_execution() as numerical_execution:
        dense, features, depths = _capture_encoder_execution(model, context)
        full_gaussians = int(dense.means.shape[1])
        if full_gaussians != views * height * width:
            raise RuntimeError("render teacher oracle requires one Gaussian per pixel")
        canonical = _clone_gaussians(dense)
        modified, stats = _apply_fixed_saes(
            canonical,
            features=features,
            depths=depths,
            context=context,
            height=height,
            width=width,
            views=views,
            materialization=MATERIALIZATION,
        )
        accounting = _oracle_output_accounting(
            modified=modified, stats=stats, full_gaussians=full_gaussians
        )
        representative_global = torch.nonzero(
            _representative_mask(modified, views=views, height=height, width=width),
            as_tuple=False,
        ).flatten()
        if representative_global.numel() != int(stats["same_budget_dense_oracle_output_gaussians"]):
            raise RuntimeError("render oracle representative count disagrees with K/2K ledger")
        retained = ~modified
        compact = _retain_renderable_gaussians(canonical, retained)
        global_to_local = torch.full(
            (full_gaussians,), -1, device=retained.device, dtype=torch.long
        )
        global_to_local[torch.nonzero(retained, as_tuple=False).flatten()] = torch.arange(
            int(retained.sum().item()), device=retained.device
        )
        representative_local = global_to_local[representative_global]
        if bool((representative_local < 0).any()):
            raise RuntimeError("a representative was removed from the compact oracle output")
        module = BoundedRepresentativeParameters(
            compact,
            dense,
            representative_global,
            representative_local,
            height=height,
            width=width,
        ).to(loaded_device)
        initial_parameter_hashes = module.parameter_hashes()
        initial_compact = module.compose()
        module.assert_immutable_slots(initial_compact)
        with torch.no_grad():
            teacher = torch.stack(
                [_render_one(model, dense, target, view, (height, width)) for view in range(target["extrinsics"].shape[1])]
            )
            initial_images = torch.stack(
                [_render_one(model, initial_compact, target, view, (height, width)) for view in range(target["extrinsics"].shape[1])]
            )
        initial_reconstruction = _image_equivalence(
            torch.stack(
                [_render_one(model, compact, target, view, (height, width)) for view in range(target["extrinsics"].shape[1])]
            ),
            initial_images,
        )

        optimizer = torch.optim.Adam(module.parameters(), lr=LEARNING_RATE)
        loss_history: list[float] = []
        gradient_report = None
        if loaded_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(loaded_device)
        start = time.monotonic()
        for step in range(STEPS):
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for view in range(teacher.shape[0]):
                candidate = module.compose()
                rendered = _render_one(model, candidate, target, view, (height, width))
                loss = F.mse_loss(rendered, teacher[view]) / teacher.shape[0]
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("render teacher loss became non-finite")
                loss.backward()
                loss_value += float(loss.detach().item())
            gradient_report = _parameter_gradient_report(module)
            gradient_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), GRADIENT_CLIP_NORM)
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("render teacher gradient norm became non-finite")
            optimizer.step()
            loss_history.append(loss_value)
            if step == 0 or (step + 1) % 16 == 0 or step + 1 == STEPS:
                print(json.dumps({"stage": "render_teacher_oracle", "step": step + 1, "loss": loss_value}))
        elapsed_seconds = time.monotonic() - start
        final_compact = _clone_gaussians(module.compose())
        module.assert_immutable_slots(final_compact)
        final_parameter_hashes = module.parameter_hashes()
        with torch.no_grad():
            final_images = torch.stack(
                [_render_one(model, final_compact, target, view, (height, width)) for view in range(target["extrinsics"].shape[1])]
            )
        peak_memory = (
            int(torch.cuda.max_memory_allocated(loaded_device))
            if loaded_device.type == "cuda"
            else None
        )

    artifacts = _save_oracle_artifacts(
        output_dir,
        representative_global=representative_global,
        module=module,
        teacher=teacher,
    )
    return {
        "schema_version": "1.0",
        "kind": KIND,
        "paper_result_eligible": False,
        "quality_retry_authorized": False,
        "model": "transplat",
        "dataset": "dl3dv",
        "sample_index": 0,
        "scene": str(batch["scene"][0]),
        "context_indices": [int(value) for value in batch["context"]["index"][0].tolist()],
        "target_indices": [int(value) for value in batch["target"]["index"][0].tolist()],
        "target_rgb_provenance": {
            "loaded_by_native_dataloader": native_target_rgb_loaded,
            "removed_before_encoder": True,
            "available_to_route_or_teacher": False,
            "used_for_optimization_or_metrics": False,
        },
        "execution_boundary": {
            "dense_s1_s2_s3_executed": True,
            "posthoc_full_s3_read_allowed_for_diagnostic": True,
            "target_camera_metadata_used_for_posthoc_optimization": True,
            "dense_renderer_output_is_teacher": True,
            "target_rgb_used": False,
            "runtime_execution": False,
            "s2_s3_savings_claimed": 0.0,
        },
        "routing": {
            "feature_threshold": 0.2,
            "depth_threshold": 0.1,
            "tile_size": 4,
            "materialization": MATERIALIZATION,
            "committed_modified_mask_sha256": _sha256_mask(modified),
            "representative_global_mask_sha256": _sha256_mask(
                _representative_mask(modified, views=views, height=height, width=width)
            ),
            "stats": stats,
        },
        "same_budget_output_accounting": accounting,
        "optimizer": {
            "kind": "Adam",
            "steps": STEPS,
            "learning_rate": LEARNING_RATE,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "one_target_view_per_backward": True,
            "early_stop": False,
            "best_checkpoint_selection": False,
            "final_iterate_only": True,
            "elapsed_seconds": elapsed_seconds,
            "peak_cuda_memory_allocated_bytes": peak_memory,
            "loss_history": loss_history,
            "gradient_report_last_step": gradient_report,
        },
        "parameterization": {
            "means": "base + producer-tile-range * tanh",
            "harmonics": "base + producer-tile-range * tanh",
            "covariances": "bounded-positive-Cholesky",
            "opacities": "epsilon + (1-2epsilon) * sigmoid",
            "epsilon": PARAMETER_EPSILON,
            "cholesky_diagonal_multiplier": CHOLESKY_DIAGONAL_MULTIPLIER,
            "bound_hashes": module.bound_hashes(),
            "initial_parameter_hashes": initial_parameter_hashes,
            "final_parameter_hashes": final_parameter_hashes,
        },
        "teacher": {
            "camera_metadata_sha256": _camera_metadata_sha256(target),
            "render_sha256": _sha256_tensor(teacher),
            "initial_fidelity": _teacher_fidelity(initial_images, teacher),
            "final_fidelity": _teacher_fidelity(final_images, teacher),
            "initial_compact_reconstruction": initial_reconstruction,
        },
        "artifacts": artifacts,
        "provenance": {
            "seed": SEED,
            "checkpoint": {
                "path": str(experiment.checkpoint),
                "sha256": cached_sha256_file(experiment.checkpoint),
            },
            "evaluation_index": {
                "path": str(selection.index_path),
                "sha256": cached_sha256_file(selection.index_path),
            },
            "source": source_identity(),
            "numerical_execution": numerical_execution,
            "device": str(loaded_device),
        },
        "oracle_conclusion": {
            "paper_result_eligible": False,
            "capacity_diagnostic_only": True,
            "fixed_bounded_optimizer_only": True,
            "does_not_establish_runtime_feasibility": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir must be a new directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    try:
        record = collect_render_teacher_oracle(device=device, output_dir=args.output_dir)
    except Exception as exc:
        (args.output_dir / "FAILED.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
        raise
    destination = args.output_dir / "results.json"
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
