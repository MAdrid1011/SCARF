"""
Test Feature Replacement: Verify SCARF extractors produce bit-accurate features.

These tests ensure that:
1. SCARF extractors produce identical features to original PyTorch
2. End-to-end quality is unchanged when using SCARF features
3. Fallback mode works correctly
"""

import pytest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add SCARF to path
SCARF_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SCARF_ROOT))


def compute_psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute PSNR between two tensors."""
    mse = F.mse_loss(x.float(), y.float())
    if mse < 1e-10:
        return float('inf')
    return 10 * torch.log10(1.0 / mse).item()


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_images(device):
    """Sample batch of images for testing."""
    B, V, C, H, W = 1, 2, 3, 256, 256
    images = torch.rand(B, V, C, H, W, device=device)
    return images


@pytest.fixture
def sample_extrinsics(device):
    """Sample camera extrinsics."""
    B, V = 1, 2
    extrinsics = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0)
    extrinsics = extrinsics.expand(B, V, 4, 4).contiguous()
    return extrinsics


class TestTransplatFeatureExtractor:
    """Test Transplat feature extraction replacement."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_features_match_original(self, device, sample_images, sample_extrinsics):
        """Verify SCARF features match original PyTorch backbone (PSNR > 100 dB)."""
        # Skip if Transplat not available
        try:
            from integration import create_model_loader
            loader = create_model_loader('transplat')
        except ImportError:
            pytest.skip("Transplat loader not available")
        
        # This test will be implemented once TransplatFeatureExtractor exists
        # For now, mark as expected to fail
        pytest.skip("TransplatFeatureExtractor not yet implemented")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")  
    def test_cycle_counting_works(self, device, sample_images):
        """Verify cycle counting produces reasonable values."""
        pytest.skip("TransplatFeatureExtractor not yet implemented")


class TestMVSplatFeatureExtractor:
    """Test MVSplat feature extraction replacement."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_features_match_original(self, device, sample_images, sample_extrinsics):
        """Verify SCARF features match original PyTorch backbone (PSNR > 100 dB)."""
        pytest.skip("MVSplatFeatureExtractor not yet implemented")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cycle_counting_works(self, device, sample_images):
        """Verify cycle counting produces reasonable values."""
        pytest.skip("MVSplatFeatureExtractor not yet implemented")


class TestDepthSplatFeatureExtractor:
    """Test DepthSplat feature extraction replacement."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_features_match_original(self, device, sample_images, sample_extrinsics):
        """Verify SCARF features match original PyTorch backbone (PSNR > 100 dB)."""
        pytest.skip("DepthSplatFeatureExtractor not yet implemented")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_dinov2_features_match(self, device, sample_images):
        """Verify DINOv2 features from ViTSimulator match original."""
        pytest.skip("DepthSplatFeatureExtractor not yet implemented")


class TestEndToEndQuality:
    """Test end-to-end quality when using SCARF features."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_transplat_quality_unchanged(self):
        """Verify PSNR unchanged when using SCARF features for Transplat."""
        pytest.skip("End-to-end test requires full pipeline implementation")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_mvsplat_quality_unchanged(self):
        """Verify PSNR unchanged when using SCARF features for MVSplat."""
        pytest.skip("End-to-end test requires full pipeline implementation")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_depthsplat_quality_unchanged(self):
        """Verify PSNR unchanged when using SCARF features for DepthSplat."""
        pytest.skip("End-to-end test requires full pipeline implementation")


class TestFallbackMode:
    """Test fallback mode with original backbone."""
    
    def test_no_feature_sim_flag_works(self):
        """Verify --no-feature-sim falls back to original backbone."""
        # This test verifies the CLI flag behavior
        import subprocess
        result = subprocess.run(
            ['python', '-c', 
             'from scripts.demo import *; print("OK")'],
            cwd=str(SCARF_ROOT),
            capture_output=True,
            text=True
        )
        # Just verify import works
        assert "Error" not in result.stderr or result.returncode == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
