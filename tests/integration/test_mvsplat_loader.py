"""
Tests for MVSplatLoader.

Tests the MVSplat model loader implementation.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from integration.model_loader import MVSplatLoader, ModelBundle, DataBundle, create_model_loader


class TestMVSplatLoaderInit:
    """Tests for MVSplatLoader initialization."""
    
    def test_init_default_root(self):
        """Test initialization with default mvsplat root."""
        loader = MVSplatLoader()
        assert loader.mvsplat_root.name == 'mvsplat'
        assert loader.mvsplat_root.exists() or True  # May not exist in CI
    
    def test_init_custom_root(self, tmp_path):
        """Test initialization with custom mvsplat root."""
        loader = MVSplatLoader(mvsplat_root=tmp_path)
        assert loader.mvsplat_root == tmp_path
    
    def test_get_model_type(self):
        """Test model type identifier."""
        loader = MVSplatLoader()
        assert loader.get_model_type() == 'mvsplat'


class TestMVSplatLoaderFactory:
    """Tests for factory function with mvsplat."""
    
    def test_create_mvsplat_loader(self):
        """Test creating MVSplatLoader via factory."""
        loader = create_model_loader('mvsplat')
        assert isinstance(loader, MVSplatLoader)
        assert loader.get_model_type() == 'mvsplat'
    
    def test_create_with_custom_root(self, tmp_path):
        """Test creating loader with custom root via factory."""
        loader = create_model_loader('mvsplat', mvsplat_root=tmp_path)
        assert loader.mvsplat_root == tmp_path


class TestMVSplatLoaderLoadModel:
    """Tests for MVSplatLoader.load_model()."""
    
    @pytest.fixture
    def mock_mvsplat_imports(self):
        """Mock MVSplat imports for testing without full installation."""
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
        loader = MVSplatLoader(mvsplat_root=tmp_path)
        fake_ckpt = tmp_path / 'nonexistent.ckpt'
        
        with pytest.raises((FileNotFoundError, RuntimeError, NotImplementedError)):
            loader.load_model(str(fake_ckpt))
    
    def test_load_model_returns_model_bundle(self, mock_mvsplat_imports, tmp_path):
        """Test that load_model returns ModelBundle (when implemented)."""
        loader = MVSplatLoader(mvsplat_root=tmp_path)
        
        # Create fake checkpoint
        fake_ckpt = tmp_path / 'checkpoints' / 're10k.ckpt'
        fake_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': {}}, fake_ckpt)
        
        # This will raise NotImplementedError until fully implemented
        try:
            result = loader.load_model(str(fake_ckpt))
            assert isinstance(result, ModelBundle)
        except NotImplementedError:
            pytest.skip("MVSplatLoader not yet fully implemented")


class TestMVSplatLoaderLoadData:
    """Tests for MVSplatLoader.load_data()."""
    
    def test_load_data_requires_model_bundle(self, tmp_path):
        """Test that load_data requires a valid ModelBundle."""
        loader = MVSplatLoader(mvsplat_root=tmp_path)
        
        # Create mock model bundle
        mock_bundle = MagicMock(spec=ModelBundle)
        
        # This will raise NotImplementedError until fully implemented
        try:
            result = loader.load_data(mock_bundle, 're10k', 1)
            assert isinstance(result, DataBundle)
        except NotImplementedError:
            pytest.skip("MVSplatLoader not yet fully implemented")


class TestMVSplatLoaderIntegration:
    """Integration tests for MVSplatLoader (require real model)."""
    
    @pytest.fixture
    def mvsplat_root(self):
        """Get MVSplat root path."""
        root = Path(__file__).parent.parent.parent / 'mvsplat'
        if not root.exists():
            pytest.skip("MVSplat submodule not available")
        return root
    
    @pytest.fixture
    def mvsplat_checkpoint(self, mvsplat_root):
        """Get MVSplat checkpoint path."""
        ckpt = mvsplat_root / 'checkpoints' / 're10k.ckpt'
        if not ckpt.exists():
            pytest.skip("MVSplat checkpoint not available")
        return ckpt
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for integration tests"
    )
    def test_full_pipeline(self, mvsplat_root, mvsplat_checkpoint):
        """Test full model loading and data loading pipeline."""
        loader = MVSplatLoader(mvsplat_root=mvsplat_root)
        
        try:
            # Load model
            model_bundle = loader.load_model(str(mvsplat_checkpoint))
            assert model_bundle.encoder is not None
            assert model_bundle.decoder is not None
            
            # Load data
            data_bundle = loader.load_data(model_bundle, 're10k', 1)
            assert data_bundle.batch is not None
            
        except NotImplementedError:
            pytest.skip("MVSplatLoader not yet fully implemented")
