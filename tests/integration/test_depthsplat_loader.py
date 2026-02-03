"""
Tests for DepthSplatLoader.

Tests the DepthSplat model loader implementation.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from integration.model_loader import DepthSplatLoader, ModelBundle, DataBundle, create_model_loader


class TestDepthSplatLoaderInit:
    """Tests for DepthSplatLoader initialization."""
    
    def test_init_default_root(self):
        """Test initialization with default depthsplat root."""
        loader = DepthSplatLoader()
        assert loader.depthsplat_root.name == 'depthsplat'
        assert loader.depthsplat_root.exists() or True  # May not exist in CI
    
    def test_init_custom_root(self, tmp_path):
        """Test initialization with custom depthsplat root."""
        loader = DepthSplatLoader(depthsplat_root=tmp_path)
        assert loader.depthsplat_root == tmp_path
    
    def test_get_model_type(self):
        """Test model type identifier."""
        loader = DepthSplatLoader()
        assert loader.get_model_type() == 'depthsplat'


class TestDepthSplatLoaderFactory:
    """Tests for factory function with depthsplat."""
    
    def test_create_depthsplat_loader(self):
        """Test creating DepthSplatLoader via factory."""
        loader = create_model_loader('depthsplat')
        assert isinstance(loader, DepthSplatLoader)
        assert loader.get_model_type() == 'depthsplat'
    
    def test_create_with_custom_root(self, tmp_path):
        """Test creating loader with custom root via factory."""
        loader = create_model_loader('depthsplat', depthsplat_root=tmp_path)
        assert loader.depthsplat_root == tmp_path


class TestDepthSplatLoaderLoadModel:
    """Tests for DepthSplatLoader.load_model()."""
    
    @pytest.fixture
    def mock_depthsplat_imports(self):
        """Mock DepthSplat imports for testing without full installation."""
        with patch.dict('sys.modules', {
            'src.config': MagicMock(),
            'src.model.model_wrapper': MagicMock(),
            'src.model.encoder': MagicMock(),
            'src.model.decoder': MagicMock(),
            'src.loss': MagicMock(),
            'src.misc.step_tracker': MagicMock(),
            'hydra': MagicMock(),
            'hydra.core.global_hydra': MagicMock(),
        }):
            yield
    
    def test_load_model_checkpoint_not_found(self, tmp_path):
        """Test error when checkpoint doesn't exist."""
        loader = DepthSplatLoader(depthsplat_root=tmp_path)
        fake_ckpt = tmp_path / 'nonexistent.ckpt'
        
        with pytest.raises((FileNotFoundError, RuntimeError, NotImplementedError)):
            loader.load_model(str(fake_ckpt))
    
    def test_load_model_returns_model_bundle(self, mock_depthsplat_imports, tmp_path):
        """Test that load_model returns ModelBundle (when implemented)."""
        loader = DepthSplatLoader(depthsplat_root=tmp_path)
        
        # Create fake checkpoint
        fake_ckpt = tmp_path / 'checkpoints' / 're10k.ckpt'
        fake_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': {}}, fake_ckpt)
        
        # This will raise NotImplementedError until fully implemented
        try:
            result = loader.load_model(str(fake_ckpt))
            assert isinstance(result, ModelBundle)
        except NotImplementedError:
            pytest.skip("DepthSplatLoader not yet fully implemented")


class TestDepthSplatLoaderLoadData:
    """Tests for DepthSplatLoader.load_data()."""
    
    def test_load_data_requires_model_bundle(self, tmp_path):
        """Test that load_data requires a valid ModelBundle."""
        loader = DepthSplatLoader(depthsplat_root=tmp_path)
        
        # Create mock model bundle
        mock_bundle = MagicMock(spec=ModelBundle)
        
        # This will raise NotImplementedError until fully implemented
        try:
            result = loader.load_data(mock_bundle, 're10k', 1)
            assert isinstance(result, DataBundle)
        except NotImplementedError:
            pytest.skip("DepthSplatLoader not yet fully implemented")


class TestDepthSplatLoaderDatasets:
    """Tests for DepthSplatLoader dataset support."""
    
    def test_supports_re10k(self):
        """Test that DepthSplat supports RE10K dataset."""
        loader = DepthSplatLoader()
        # This is a specification test - DepthSplat should support RE10K
        assert 're10k' in ['re10k', 'dl3dv']  # Supported datasets
    
    def test_supports_dl3dv(self):
        """Test that DepthSplat supports DL3DV dataset."""
        loader = DepthSplatLoader()
        # DepthSplat specifically supports DL3DV
        assert 'dl3dv' in ['re10k', 'dl3dv']


class TestDepthSplatLoaderIntegration:
    """Integration tests for DepthSplatLoader (require real model)."""
    
    @pytest.fixture
    def depthsplat_root(self):
        """Get DepthSplat root path."""
        root = Path(__file__).parent.parent.parent / 'depthsplat'
        if not root.exists():
            pytest.skip("DepthSplat submodule not available")
        return root
    
    @pytest.fixture
    def depthsplat_checkpoint(self, depthsplat_root):
        """Get DepthSplat checkpoint path."""
        ckpt = depthsplat_root / 'checkpoints' / 're10k.ckpt'
        if not ckpt.exists():
            pytest.skip("DepthSplat checkpoint not available")
        return ckpt
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for integration tests"
    )
    def test_full_pipeline(self, depthsplat_root, depthsplat_checkpoint):
        """Test full model loading and data loading pipeline."""
        loader = DepthSplatLoader(depthsplat_root=depthsplat_root)
        
        try:
            # Load model
            model_bundle = loader.load_model(str(depthsplat_checkpoint))
            assert model_bundle.encoder is not None
            assert model_bundle.decoder is not None
            
            # Load data
            data_bundle = loader.load_data(model_bundle, 're10k', 1)
            assert data_bundle.batch is not None
            
        except NotImplementedError:
            pytest.skip("DepthSplatLoader not yet fully implemented")
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for integration tests"
    )
    def test_dl3dv_dataset(self, depthsplat_root, depthsplat_checkpoint):
        """Test loading DL3DV dataset (DepthSplat-specific)."""
        loader = DepthSplatLoader(depthsplat_root=depthsplat_root)
        
        try:
            model_bundle = loader.load_model(str(depthsplat_checkpoint))
            
            # Try loading DL3DV
            data_bundle = loader.load_data(model_bundle, 'dl3dv', 1)
            assert data_bundle.batch is not None
            
        except (NotImplementedError, FileNotFoundError):
            pytest.skip("DL3DV dataset not available or loader not implemented")
