"""
Model Loader

Abstract base class and implementations for loading different 3DGS models.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
from dataclasses import dataclass
import torch


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


class BaseModelLoader(ABC):
    """
    Abstract base class for model loaders.
    
    Each 3DGS model (Transplat, MVSplat, DepthSplat) needs
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
    ) -> ModelBundle:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Optional path to config directory
            device: Target device (default: cuda if available)
        
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
    ) -> DataBundle:
        """
        Load test data for the model.
        
        Args:
            model_bundle: Loaded model bundle
            dataset_name: Dataset to load ('re10k', 'acid', 'dtu')
            num_samples: Number of test samples to load
        
        Returns:
            DataBundle with batch and data_shim
        """
        pass
    
    @abstractmethod
    def get_model_type(self) -> str:
        """Return model type identifier."""
        pass


class TransplatLoader(BaseModelLoader):
    """Loader for Transplat model."""
    
    def __init__(self, transplat_root: Optional[Path] = None):
        """
        Initialize Transplat loader.
        
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
        """Setup sys.path for Transplat imports."""
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
    ) -> ModelBundle:
        """Load Transplat model."""
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
                config_path = str(self.transplat_root / 'config')
            
            GlobalHydra.instance().clear()
            
            with initialize_config_dir(config_dir=config_path, version_base=None):
                cfg_dict = compose(config_name="main", overrides=["+experiment=re10k"])
            
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
            state_dict = ckpt.get('state_dict', ckpt)
            model_state = model.state_dict()
            filtered_state = {k: v for k, v in state_dict.items() 
                              if k in model_state and v.shape == model_state[k].shape}
            model.load_state_dict(filtered_state, strict=False)
            
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
    ) -> DataBundle:
        """Load Transplat test data."""
        self._setup_imports()
        
        try:
            from src.dataset.data_module import DataModule, get_data_shim
            
            cfg = model_bundle.config
            cfg.dataset.test_len = num_samples
            
            data_module = DataModule(cfg.dataset, cfg.data_loader)
            data_module.setup("test")
            test_loader = data_module.test_dataloader()
            
            # Get first batch
            batch = next(iter(test_loader))
            
            # Apply data shim
            data_shim = get_data_shim(model_bundle.encoder)
            batch = data_shim(batch)
            
            return DataBundle(batch=batch, data_shim=data_shim)
        finally:
            self._restore_cwd()
    
    def get_model_type(self) -> str:
        return 'transplat'


class MVSplatLoader(BaseModelLoader):
    """Loader for MVSplat model (placeholder)."""
    
    def load_model(self, checkpoint_path: str, config_path: Optional[str] = None,
                   device: Optional[torch.device] = None) -> ModelBundle:
        raise NotImplementedError("MVSplat loader not yet implemented")
    
    def load_data(self, model_bundle: ModelBundle, dataset_name: str = 're10k',
                  num_samples: int = 1) -> DataBundle:
        raise NotImplementedError("MVSplat loader not yet implemented")
    
    def get_model_type(self) -> str:
        return 'mvsplat'


class DepthSplatLoader(BaseModelLoader):
    """Loader for DepthSplat model (placeholder)."""
    
    def load_model(self, checkpoint_path: str, config_path: Optional[str] = None,
                   device: Optional[torch.device] = None) -> ModelBundle:
        raise NotImplementedError("DepthSplat loader not yet implemented")
    
    def load_data(self, model_bundle: ModelBundle, dataset_name: str = 're10k',
                  num_samples: int = 1) -> DataBundle:
        raise NotImplementedError("DepthSplat loader not yet implemented")
    
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
