"""
Model Loader

Abstract base class and implementations for loading different 3DGS models.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
from dataclasses import dataclass
import torch


DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "assets/torch/hub"
    / f"facebookresearch_dinov2_{DINOV2_COMMIT}"
)


def load_checkpoint_state(model: torch.nn.Module, checkpoint: Any) -> dict[str, Any]:
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("checkpoint has no non-empty state dictionary")
    model_state = model.state_dict()
    shape_mismatches = [
        key
        for key, value in state_dict.items()
        if key in model_state
        and torch.is_tensor(value)
        and value.shape != model_state[key].shape
    ]
    if shape_mismatches:
        raise ValueError(
            "checkpoint tensor shape mismatch: " + ", ".join(shape_mismatches[:3])
        )
    tensor_state = {
        key: value for key, value in state_dict.items() if torch.is_tensor(value)
    }
    matched = {key: value for key, value in tensor_state.items() if key in model_state}
    if not matched:
        raise ValueError("checkpoint has no tensor matching the configured model")
    incompatible = model.load_state_dict(matched, strict=False)
    checkpoint_numel = sum(value.numel() for value in tensor_state.values())
    matched_numel = sum(value.numel() for value in matched.values())
    report = {
        "checkpoint_tensors": len(tensor_state),
        "matched_tensors": len(matched),
        "missing_model_tensors": len(incompatible.missing_keys),
        "unmatched_checkpoint_tensors": len(tensor_state) - len(matched),
        "checkpoint_numel": checkpoint_numel,
        "matched_numel": matched_numel,
        "matched_checkpoint_numel_fraction": matched_numel / checkpoint_numel,
    }
    model._scarf_checkpoint_load = report
    return report


def get_depthsplat_encoder(get_encoder, encoder_cfg):
    original_hub_load = torch.hub.load

    def pinned_hub_load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir == "facebookresearch/dinov2":
            repo_or_dir = str(DINOV2_SOURCE)
            kwargs["source"] = "local"
            # The hash-pinned DepthSplat checkpoints contain the complete
            # DINO backbone. Avoid redundant weights and all run-time network
            # access while constructing the matching module hierarchy.
            kwargs["pretrained"] = False
        return original_hub_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = pinned_hub_load
    try:
        return get_encoder(encoder_cfg)
    finally:
        torch.hub.load = original_hub_load


@dataclass
class ModelBundle:
    """Container for loaded model components."""
    encoder: Any
    decoder: Any
    model: Any  # Full model wrapper
    config: Any
    device: torch.device


@dataclass  
class DataBundle:
    """Container for loaded data."""
    batch: Dict[str, torch.Tensor]
    data_shim: Any


class EncoderOnlyModel:
    """Checkpoint-compatible encoder proxy for target-free audits."""

    def __init__(self, encoder: Any):
        self.encoder = encoder

    def state_dict(self) -> dict[str, Any]:
        return {
            f"encoder.{key}": value
            for key, value in self.encoder.state_dict().items()
        }

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = False) -> Any:
        from types import SimpleNamespace

        prefix = "encoder."
        encoder_state = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        incompatible = self.encoder.load_state_dict(encoder_state, strict=strict)
        return SimpleNamespace(
            missing_keys=[f"encoder.{key}" for key in incompatible.missing_keys],
            unexpected_keys=[f"encoder.{key}" for key in incompatible.unexpected_keys],
        )

    def to(self, device: torch.device) -> "EncoderOnlyModel":
        self.encoder.to(device)
        return self

    def eval(self) -> "EncoderOnlyModel":
        self.encoder.eval()
        return self


def _decode_calibration_context_images(images: list[Any]) -> torch.Tensor:
    """Decode only declared context images from a target-free sidecar."""
    from PIL import Image
    import torchvision.transforms as transforms

    decoded = []
    for image in images:
        if torch.is_tensor(image):
            if image.dtype != torch.uint8:
                raise ValueError("calibration context image must be uint8 bytes")
            payload = image.detach().cpu().contiguous().numpy().tobytes()
        elif isinstance(image, (bytes, bytearray)):
            payload = bytes(image)
        else:
            raise ValueError("calibration context image has an unsupported type")
        decoded.append(transforms.ToTensor()(Image.open(BytesIO(payload))))
    if not decoded:
        raise ValueError("calibration input has no context images")
    return torch.stack(decoded)


def _calibration_camera_geometry(cameras: Any) -> tuple[torch.Tensor, torch.Tensor]:
    cameras = torch.as_tensor(cameras, dtype=torch.float32).clone()
    if cameras.dim() != 2 or cameras.shape[1] != 18:
        raise ValueError("calibration camera geometry must have shape [views, 18]")
    views = cameras.shape[0]
    intrinsics = torch.eye(3, dtype=torch.float32).repeat(views, 1, 1)
    intrinsics[:, 0, 0] = cameras[:, 0]
    intrinsics[:, 1, 1] = cameras[:, 1]
    intrinsics[:, 0, 2] = cameras[:, 2]
    intrinsics[:, 1, 2] = cameras[:, 3]
    w2c = torch.eye(4, dtype=torch.float32).repeat(views, 1, 1)
    w2c[:, :3] = cameras[:, 6:].reshape(views, 3, 4)
    return torch.linalg.inv(w2c), intrinsics


def load_target_free_calibration_data(
    loader: Any,
    model_bundle: ModelBundle,
    *,
    dataset_name: str,
    dataset_root: Path,
    evaluation_index: Path,
    sample_index: int,
) -> DataBundle:
    """Construct one model-ready batch without loading target RGB pixels."""
    from scripts.calibration_inputs import (
        calibration_scene_order,
        load_target_free_record,
        sha256_file,
        validate_target_free_input_root,
    )
    from scripts.compile_protocol import canonicalize_index

    loader._setup_imports()
    try:
        calibration_identity = validate_target_free_input_root(
            dataset_root, dataset_name
        )
        scene_order = calibration_scene_order(dataset_root)
        index_sha256 = sha256_file(evaluation_index)
        rows, index_summary = canonicalize_index(
            evaluation_index,
            index_sha256,
            execution_scene_order=scene_order,
        )
        if (
            calibration_identity["selection_sha256"]
            != index_summary["sample_selection_sha256"]
        ):
            raise ValueError("calibration input selection does not match its index")
        # The caller may resume a run-pair worker.  Bind the sidecar identity to
        # the exact index bytes it actually opened so a same-scene stale trace
        # cannot be mistaken for evidence from a changed sidecar/protocol.
        calibration_identity = {
            **calibration_identity,
            "index_sha256": index_sha256,
            "index_selection_sha256": index_summary["sample_selection_sha256"],
        }
        if sample_index < 0 or sample_index >= len(rows):
            raise IndexError(f"sample_index={sample_index} is out of range")
        selection = rows[sample_index]
        record = load_target_free_record(dataset_root, selection["scene"])
        if (
            record["context_indices"] != selection["context_indices"]
            or record["target_indices"] != selection["target_indices"]
        ):
            raise ValueError("calibration input record does not match its selection")

        context_images = _decode_calibration_context_images(record["context_images"])
        source_height, source_width = context_images.shape[-2:]
        extrinsics, intrinsics = _calibration_camera_geometry(record["cameras"])
        context_indices = selection["context_indices"]
        target_indices = selection["target_indices"]
        if max([*context_indices, *target_indices]) >= extrinsics.shape[0]:
            raise ValueError("calibration selection exceeds camera geometry")

        dataset_cfg = model_bundle.config.dataset
        context_extrinsics = extrinsics[context_indices]
        scale: torch.Tensor | float = 1.0
        if len(context_indices) == 2 and bool(
            getattr(dataset_cfg, "make_baseline_1", False)
        ):
            scale = (
                context_extrinsics[0, :3, 3] - context_extrinsics[1, :3, 3]
            ).norm()
            if float(scale) < float(getattr(dataset_cfg, "baseline_epsilon", 0.0)):
                raise ValueError("calibration context has insufficient baseline")
            extrinsics[:, :3, 3] /= scale
        near_value = float(getattr(dataset_cfg, "near", -1.0))
        far_value = float(getattr(dataset_cfg, "far", -1.0))
        near_value = 0.1 if near_value == -1.0 else near_value
        far_value = 1000.0 if far_value == -1.0 else far_value
        nf_scale: torch.Tensor | float = (
            scale
            if bool(getattr(dataset_cfg, "baseline_scale_bounds", True))
            else 1.0
        )

        # This all-zero shape carrier is used only by upstream crop shims. It is
        # discarded before any model or calibration computation can inspect it.
        target_placeholder = context_images.new_zeros(
            (len(target_indices), 3, source_height, source_width)
        )
        batch = {
            "context": {
                "extrinsics": extrinsics[context_indices].unsqueeze(0),
                "intrinsics": intrinsics[context_indices].unsqueeze(0),
                "image": context_images.unsqueeze(0),
                "near": torch.full((1, len(context_indices)), near_value) / nf_scale,
                "far": torch.full((1, len(context_indices)), far_value) / nf_scale,
                "index": torch.tensor(context_indices, dtype=torch.long).unsqueeze(0),
            },
            "target": {
                "extrinsics": extrinsics[target_indices].unsqueeze(0),
                "intrinsics": intrinsics[target_indices].unsqueeze(0),
                "image": target_placeholder.unsqueeze(0),
                "near": torch.full((1, len(target_indices)), near_value) / nf_scale,
                "far": torch.full((1, len(target_indices)), far_value) / nf_scale,
                "index": torch.tensor(target_indices, dtype=torch.long).unsqueeze(0),
            },
            "scene": [selection["scene"]],
        }

        from src.dataset.data_module import get_data_shim
        from src.dataset.shims.crop_shim import apply_crop_shim

        batch = apply_crop_shim(batch, tuple(model_bundle.config.dataset.image_shape))
        data_shim = get_data_shim(model_bundle.encoder)
        batch = data_shim(batch)
        batch["target"].pop("image")
        batch["calibration"] = calibration_identity
        return DataBundle(batch=batch, data_shim=data_shim)
    finally:
        loader._restore_cwd()


def load_context_only_audit_data(
    loader: Any,
    model_bundle: ModelBundle,
    *,
    input_root: Path,
) -> DataBundle:
    """Construct one encoder batch from a context-camera-only audit sidecar.

    Unlike the generic calibration path, this entrypoint never constructs a
    target placeholder or target-camera tensor. It mirrors the TranSplat
    context crop and patch shims directly because their batch wrappers require
    a target mapping that this audit contract intentionally forbids.
    """
    from data.context_only_audit_input import load_context_only_audit_record

    record = load_context_only_audit_record(input_root)
    loader._setup_imports()
    try:
        context_images = _decode_calibration_context_images(record["context_images"])
        source_height, source_width = context_images.shape[-2:]
        extrinsics, intrinsics = _calibration_camera_geometry(record["context_cameras"])
        context_indices = record["context_indices"]
        if extrinsics.shape[0] != len(context_indices):
            raise ValueError("context-only audit camera count does not match its context")

        dataset_cfg = model_bundle.config.dataset
        scale: torch.Tensor | float = 1.0
        if len(context_indices) == 2 and bool(
            getattr(dataset_cfg, "make_baseline_1", False)
        ):
            scale = (extrinsics[0, :3, 3] - extrinsics[1, :3, 3]).norm()
            if float(scale) < float(getattr(dataset_cfg, "baseline_epsilon", 0.0)):
                raise ValueError("context-only audit has insufficient baseline")
            extrinsics[:, :3, 3] /= scale
        near_value = float(getattr(dataset_cfg, "near", -1.0))
        far_value = float(getattr(dataset_cfg, "far", -1.0))
        near_value = 0.1 if near_value == -1.0 else near_value
        far_value = 1000.0 if far_value == -1.0 else far_value
        nf_scale: torch.Tensor | float = (
            scale if bool(getattr(dataset_cfg, "baseline_scale_bounds", True)) else 1.0
        )
        context = {
            "extrinsics": extrinsics.unsqueeze(0),
            "intrinsics": intrinsics.unsqueeze(0),
            "image": context_images.unsqueeze(0),
            "near": torch.full((1, len(context_indices)), near_value) / nf_scale,
            "far": torch.full((1, len(context_indices)), far_value) / nf_scale,
            "index": torch.tensor(context_indices, dtype=torch.long).unsqueeze(0),
        }

        from src.dataset.shims.crop_shim import apply_crop_shim_to_views
        from src.dataset.shims.patch_shim import apply_patch_shim_to_views

        context = apply_crop_shim_to_views(
            context, tuple(model_bundle.config.dataset.image_shape)
        )
        encoder_cfg = getattr(model_bundle.encoder, "cfg", None)
        patch_size = int(getattr(encoder_cfg, "shim_patch_size", 1)) * int(
            getattr(encoder_cfg, "downscale_factor", 1)
        )
        if patch_size < 1:
            raise ValueError("context-only audit encoder has an invalid patch size")
        context = apply_patch_shim_to_views(context, patch_size)
        batch = {
            "context": context,
            "scene": [record["key"]],
            "calibration": {
                **record["input_identity"],
                "target_rgb_accessed": False,
                "target_camera_metadata_accessed": False,
                "target_mapping_present": False,
                "source_image_shape": [int(source_height), int(source_width)],
            },
        }
        return DataBundle(batch=batch, data_shim=lambda value: value)
    finally:
        loader._restore_cwd()


def _load_sequential_data(
    loader: Any,
    model_bundle: ModelBundle,
    dataset_name: str,
    num_samples: int,
    sample_index: int,
) -> DataBundle:
    """Reuse one ordered test dataloader for a model/dataset pair."""
    loader._setup_imports()
    try:
        from src.dataset.data_module import DataModule, get_data_shim

        key = (id(model_bundle), dataset_name, num_samples)
        state = getattr(loader, "_data_session", None)
        if state is None or state["key"] != key or sample_index < state["index"]:
            cfg = model_bundle.config
            cfg.dataset.test_len = num_samples
            data_module = DataModule(cfg.dataset, cfg.data_loader)
            data_module.setup("test")
            state = {
                "key": key,
                "iterator": iter(data_module.test_dataloader()),
                "data_module": data_module,
                "data_shim": get_data_shim(model_bundle.encoder),
                "index": -1,
                "batch": None,
            }
            loader._data_session = state

        while state["index"] < sample_index:
            try:
                state["batch"] = next(state["iterator"])
            except StopIteration as exc:
                raise IndexError(f"sample_index={sample_index} is out of range") from exc
            state["index"] += 1
        if state["batch"] is None:
            raise IndexError(f"sample_index={sample_index} is out of range")
        return DataBundle(
            batch=state["data_shim"](state["batch"]),
            data_shim=state["data_shim"],
        )
    finally:
        loader._restore_cwd()


class BaseModelLoader(ABC):
    """
    Abstract base class for model loaders.
    
    Each 3DGS model (TranSplat, MVSplat, DepthSplat) needs
    a specific loader due to different:
    - Config systems (Hydra, YAML, etc.)
    - Model architectures
    - Data formats
    """
    
    @abstractmethod
    def load_model(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        experiment_name: str = 're10k',
        dataset_root: Optional[Path] = None,
        evaluation_index: Optional[Path] = None,
        hydra_overrides: Tuple[str, ...] = (),
    ) -> ModelBundle:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Optional path to config directory
            device: Target device (default: cuda if available)
            experiment_name: Hydra experiment configuration to compose
            dataset_root: Optional absolute dataset root override
            evaluation_index: Optional upstream evaluation sampler index
            hydra_overrides: Checkpoint-variant overrides from the experiment contract
        
        Returns:
            ModelBundle with encoder, decoder, model, config, device
        """
        pass
    
    @abstractmethod
    def load_data(
        self,
        model_bundle: ModelBundle,
        dataset_name: str = 're10k',
        num_samples: int = 1,
        sample_index: int = 0,
    ) -> DataBundle:
        """
        Load test data for the model.
        
        Args:
            model_bundle: Loaded model bundle
            dataset_name: Dataset to load ('re10k', 'acid', 'dtu')
            num_samples: Number of test samples to load
            sample_index: Zero-based sample index to select from the test dataloader
        
        Returns:
            DataBundle with batch and data_shim
        """
        pass
    
    @abstractmethod
    def get_model_type(self) -> str:
        """Return model type identifier."""
        pass


class TransplatLoader(BaseModelLoader):
    """Loader for TranSplat model."""
    
    def __init__(self, transplat_root: Optional[Path] = None):
        """
        Initialize TranSplat loader.
        
        Args:
            transplat_root: Path to transplat repository root.
                           If None, uses SCARF/transplat submodule.
        """
        if transplat_root is None:
            # Default to SCARF/transplat submodule
            self.transplat_root = Path(__file__).parent.parent / 'transplat'
        else:
            self.transplat_root = Path(transplat_root)
        
        self._original_cwd = None
    
    def _setup_imports(self):
        """Setup sys.path for TranSplat imports."""
        import sys
        import os
        
        self._original_cwd = os.getcwd()
        os.chdir(self.transplat_root)
        
        if str(self.transplat_root) not in sys.path:
            sys.path.insert(0, str(self.transplat_root))
    
    def _restore_cwd(self):
        """Restore original working directory."""
        import os
        if self._original_cwd:
            os.chdir(self._original_cwd)
    
    def load_model(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        experiment_name: str = 're10k',
        dataset_root: Optional[Path] = None,
        evaluation_index: Optional[Path] = None,
        hydra_overrides: Tuple[str, ...] = (),
        encoder_only: bool = False,
    ) -> ModelBundle:
        """Load TranSplat, optionally without decoder/loss construction."""
        checkpoint_path = str(Path(checkpoint_path).resolve())
        config_path = str(Path(config_path).resolve()) if config_path is not None else None
        dataset_root = Path(dataset_root).resolve() if dataset_root is not None else None
        evaluation_index = (
            Path(evaluation_index).resolve() if evaluation_index is not None else None
        )
        self._setup_imports()
        
        try:
            from src.config import load_typed_root_config
            from src.model.encoder import get_encoder
            from src.global_cfg import set_cfg
            from hydra import compose, initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
            
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load checkpoint
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            
            # Setup hydra config
            if config_path is None:
                config_path = str(self.transplat_root / 'config')
            
            GlobalHydra.instance().clear()
            
            with initialize_config_dir(config_dir=config_path, version_base=None):
                overrides = [f"+experiment={experiment_name}", *hydra_overrides]
                if evaluation_index is not None:
                    overrides.append("dataset/view_sampler=evaluation")
                cfg_dict = compose(config_name="main", overrides=overrides)

            if dataset_root is not None:
                cfg_dict.dataset.roots = [str(Path(dataset_root).resolve())]
            if evaluation_index is not None:
                index_path = Path(evaluation_index).resolve()
                if not index_path.is_file():
                    raise FileNotFoundError(f"evaluation index not found: {index_path}")
                cfg_dict.dataset.view_sampler.index_path = str(index_path)
            cfg_dict.data_loader.test.num_workers = 0
            cfg_dict.data_loader.test.persistent_workers = False
            cfg_dict.data_loader.test.batch_size = 1
            
            cfg_dict.mode = 'test'
            set_cfg(cfg_dict)
            cfg = load_typed_root_config(cfg_dict)
            
            # Target-free SAES audits only need the encoder's S1/S2/S3 output.
            # Avoid constructing the decoder, loss collection, and evaluator,
            # which can initialize LPIPS despite no target image being present.
            encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
            if encoder_only:
                model = EncoderOnlyModel(encoder)
                decoder = None
            else:
                from src.loss import get_losses
                from src.misc.step_tracker import StepTracker
                from src.model.decoder import get_decoder
                from src.model.model_wrapper import ModelWrapper

                decoder = get_decoder(cfg.model.decoder, cfg.dataset)
                losses = get_losses(cfg.loss)
                step_tracker = StepTracker()
                model = ModelWrapper(
                    cfg.optimizer, cfg.test, cfg.train,
                    encoder, encoder_visualizer, decoder, losses, step_tracker
                )
            
            # Load weights
            load_checkpoint_state(model, ckpt)
            
            model = model.to(device)
            model.eval()
            
            return ModelBundle(
                encoder=model.encoder,
                decoder=decoder,
                model=model,
                config=cfg,
                device=device,
            )
        finally:
            self._restore_cwd()
    
    def load_data(
        self,
        model_bundle: ModelBundle,
        dataset_name: str = 're10k',
        num_samples: int = 1,
        sample_index: int = 0,
    ) -> DataBundle:
        """Load TranSplat test data."""
        return _load_sequential_data(
            self, model_bundle, dataset_name, num_samples, sample_index
        )
    
    def get_model_type(self) -> str:
        return 'transplat'


class MVSplatLoader(BaseModelLoader):
    """Loader for MVSplat model."""
    
    def __init__(self, mvsplat_root: Optional[Path] = None):
        """
        Initialize MVSplat loader.
        
        Args:
            mvsplat_root: Path to mvsplat repository root.
                         If None, uses SCARF/mvsplat submodule.
        """
        if mvsplat_root is None:
            self.mvsplat_root = Path(__file__).parent.parent / 'mvsplat'
        else:
            self.mvsplat_root = Path(mvsplat_root)
        
        self._original_cwd = None
    
    def _setup_imports(self):
        """Setup sys.path for MVSplat imports."""
        import sys
        import os
        
        self._original_cwd = os.getcwd()
        os.chdir(self.mvsplat_root)
        
        if str(self.mvsplat_root) not in sys.path:
            sys.path.insert(0, str(self.mvsplat_root))
    
    def _restore_cwd(self):
        """Restore original working directory."""
        import os
        if self._original_cwd:
            os.chdir(self._original_cwd)
    
    def load_model(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        experiment_name: str = 're10k',
        dataset_root: Optional[Path] = None,
        evaluation_index: Optional[Path] = None,
        hydra_overrides: Tuple[str, ...] = (),
    ) -> ModelBundle:
        """Load MVSplat model."""
        checkpoint_path = str(Path(checkpoint_path).resolve())
        config_path = str(Path(config_path).resolve()) if config_path is not None else None
        dataset_root = Path(dataset_root).resolve() if dataset_root is not None else None
        evaluation_index = (
            Path(evaluation_index).resolve() if evaluation_index is not None else None
        )
        self._setup_imports()
        
        try:
            from src.config import load_typed_root_config
            from src.model.model_wrapper import ModelWrapper
            from src.model.encoder import get_encoder
            from src.model.decoder import get_decoder
            from src.loss import get_losses
            from src.misc.step_tracker import StepTracker
            from src.global_cfg import set_cfg
            from hydra import compose, initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
            
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load checkpoint
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            
            # Setup hydra config
            if config_path is None:
                config_path = str(self.mvsplat_root / 'config')
            
            GlobalHydra.instance().clear()
            
            with initialize_config_dir(config_dir=config_path, version_base=None):
                overrides = [f"+experiment={experiment_name}", *hydra_overrides]
                if evaluation_index is not None:
                    overrides.append("dataset/view_sampler=evaluation")
                cfg_dict = compose(config_name="main", overrides=overrides)

            if dataset_root is not None:
                cfg_dict.dataset.roots = [str(Path(dataset_root).resolve())]
            if evaluation_index is not None:
                index_path = Path(evaluation_index).resolve()
                if not index_path.is_file():
                    raise FileNotFoundError(f"evaluation index not found: {index_path}")
                cfg_dict.dataset.view_sampler.index_path = str(index_path)
            cfg_dict.data_loader.test.num_workers = 0
            cfg_dict.data_loader.test.persistent_workers = False
            cfg_dict.data_loader.test.batch_size = 1
            
            cfg_dict.mode = 'test'
            set_cfg(cfg_dict)
            cfg = load_typed_root_config(cfg_dict)
            
            # Build model
            encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
            decoder = get_decoder(cfg.model.decoder, cfg.dataset)
            losses = get_losses(cfg.loss)
            step_tracker = StepTracker()
            
            model = ModelWrapper(
                cfg.optimizer, cfg.test, cfg.train,
                encoder, encoder_visualizer, decoder, losses, step_tracker
            )
            
            # Load weights
            load_checkpoint_state(model, ckpt)
            
            model = model.to(device)
            model.eval()
            
            return ModelBundle(
                encoder=model.encoder,
                decoder=model.decoder,
                model=model,
                config=cfg,
                device=device,
            )
        finally:
            self._restore_cwd()
    
    def load_data(
        self,
        model_bundle: ModelBundle,
        dataset_name: str = 're10k',
        num_samples: int = 1,
        sample_index: int = 0,
    ) -> DataBundle:
        """Load MVSplat test data."""
        return _load_sequential_data(
            self, model_bundle, dataset_name, num_samples, sample_index
        )
    
    def get_model_type(self) -> str:
        return 'mvsplat'


class DepthSplatLoader(BaseModelLoader):
    """Loader for DepthSplat model."""
    
    def __init__(self, depthsplat_root: Optional[Path] = None):
        """
        Initialize DepthSplat loader.
        
        Args:
            depthsplat_root: Path to depthsplat repository root.
                            If None, uses SCARF/depthsplat submodule.
        """
        if depthsplat_root is None:
            self.depthsplat_root = Path(__file__).parent.parent / 'depthsplat'
        else:
            self.depthsplat_root = Path(depthsplat_root)
        
        self._original_cwd = None
    
    def _setup_imports(self):
        """Setup sys.path for DepthSplat imports."""
        import sys
        import os
        
        self._original_cwd = os.getcwd()
        os.chdir(self.depthsplat_root)
        
        if str(self.depthsplat_root) not in sys.path:
            sys.path.insert(0, str(self.depthsplat_root))
    
    def _restore_cwd(self):
        """Restore original working directory."""
        import os
        if self._original_cwd:
            os.chdir(self._original_cwd)
    
    def load_model(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        experiment_name: str = 're10k',
        dataset_root: Optional[Path] = None,
        evaluation_index: Optional[Path] = None,
        hydra_overrides: Tuple[str, ...] = (),
    ) -> ModelBundle:
        """Load DepthSplat model."""
        checkpoint_path = str(Path(checkpoint_path).resolve())
        config_path = str(Path(config_path).resolve()) if config_path is not None else None
        dataset_root = Path(dataset_root).resolve() if dataset_root is not None else None
        evaluation_index = (
            Path(evaluation_index).resolve() if evaluation_index is not None else None
        )
        self._setup_imports()
        
        try:
            from src.config import load_typed_root_config
            from src.model.model_wrapper import ModelWrapper
            from src.model.encoder import get_encoder
            from src.model.decoder import get_decoder
            from src.loss import get_losses
            from src.misc.step_tracker import StepTracker
            from src.global_cfg import set_cfg
            from hydra import compose, initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
            
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load checkpoint
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            
            # Setup hydra config
            if config_path is None:
                config_path = str(self.depthsplat_root / 'config')
            
            GlobalHydra.instance().clear()
            
            with initialize_config_dir(config_dir=config_path, version_base=None):
                overrides = [f"+experiment={experiment_name}", *hydra_overrides]
                if evaluation_index is not None:
                    overrides.append("dataset/view_sampler=evaluation")
                cfg_dict = compose(config_name="main", overrides=overrides)

            if dataset_root is not None:
                cfg_dict.dataset.roots = [str(Path(dataset_root).resolve())]
            if evaluation_index is not None:
                index_path = Path(evaluation_index).resolve()
                if not index_path.is_file():
                    raise FileNotFoundError(f"evaluation index not found: {index_path}")
                cfg_dict.dataset.view_sampler.index_path = str(index_path)
            cfg_dict.data_loader.test.num_workers = 0
            cfg_dict.data_loader.test.persistent_workers = False
            cfg_dict.data_loader.test.batch_size = 1
            
            cfg_dict.mode = 'test'
            set_cfg(cfg_dict)
            cfg = load_typed_root_config(cfg_dict)
            
            # Build model
            encoder, encoder_visualizer = get_depthsplat_encoder(
                get_encoder, cfg.model.encoder
            )
            decoder = get_decoder(cfg.model.decoder, cfg.dataset)
            losses = get_losses(cfg.loss)
            step_tracker = StepTracker()
            
            model = ModelWrapper(
                cfg.optimizer, cfg.test, cfg.train,
                encoder, encoder_visualizer, decoder, losses, step_tracker
            )
            
            # Load weights
            load_checkpoint_state(model, ckpt)
            
            model = model.to(device)
            model.eval()
            
            return ModelBundle(
                encoder=model.encoder,
                decoder=model.decoder,
                model=model,
                config=cfg,
                device=device,
            )
        finally:
            self._restore_cwd()
    
    def load_data(
        self,
        model_bundle: ModelBundle,
        dataset_name: str = 're10k',
        num_samples: int = 1,
        sample_index: int = 0,
    ) -> DataBundle:
        """Load DepthSplat test data."""
        return _load_sequential_data(
            self, model_bundle, dataset_name, num_samples, sample_index
        )
    
    def get_model_type(self) -> str:
        return 'depthsplat'


def create_model_loader(model_type: str, **kwargs) -> BaseModelLoader:
    """
    Factory function to create appropriate model loader.
    
    Args:
        model_type: 'transplat', 'mvsplat', or 'depthsplat'
        **kwargs: Additional arguments for the loader
    
    Returns:
        Configured model loader instance
    """
    loaders = {
        'transplat': TransplatLoader,
        'mvsplat': MVSplatLoader,
        'depthsplat': DepthSplatLoader,
    }
    
    if model_type not in loaders:
        raise ValueError(
            f"Unknown model: {model_type}. "
            f"Supported: {list(loaders.keys())}"
        )
    
    return loaders[model_type](**kwargs)
