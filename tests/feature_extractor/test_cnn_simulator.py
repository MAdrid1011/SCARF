"""
CNN Simulator Tests

Tests for the CNN backbone hardware simulator.
Key requirement: PSNR > 100 dB (bit-accurate match with PyTorch).
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

SCARF_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SCARF_ROOT))

from feature_extractor.types import CNNConfig, FeatureOutput


def compute_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute PSNR between two tensors."""
    mse = F.mse_loss(a.float(), b.float())
    if mse == 0:
        return 100.0  # Perfect match
    return -10 * torch.log10(mse).item()


class TestCNNSimulatorBasics:
    """Basic CNN simulator tests."""
    
    def test_config_creation(self):
        """Test CNNConfig creation."""
        config = CNNConfig()
        assert config.input_channels == 3
        assert config.output_dim == 128
        assert config.downscale_factor == 8
    
    def test_config_custom(self):
        """Test custom CNNConfig."""
        config = CNNConfig(output_dim=256, num_output_scales=0)
        assert config.output_dim == 256
        assert config.downscale_factor == 4


class TestConv7x7Layer:
    """Tests for Conv7x7 layer simulation."""
    
    def test_conv7x7_matches_pytorch(self, device):
        """Conv7x7 should match PyTorch exactly."""
        from feature_extractor.cnn_simulator import Conv7x7LayerSim
        
        # Create PyTorch reference
        pytorch_conv = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False).to(device)
        pytorch_norm = nn.InstanceNorm2d(64).to(device)
        
        # Create simulator with same weights
        sim = Conv7x7LayerSim(3, 64, device=device)
        sim.load_weights(pytorch_conv.weight, pytorch_norm)
        
        # Test
        x = torch.randn(2, 3, 256, 256, device=device)
        
        with torch.no_grad():
            pytorch_out = F.relu(pytorch_norm(pytorch_conv(x)))
            sim_out, cycles = sim.forward(x)
        
        psnr = compute_psnr(sim_out, pytorch_out)
        assert psnr > 100, f"PSNR {psnr:.2f} dB < 100 dB (not bit-accurate)"
        assert cycles > 0, "Should report non-zero cycles"


class TestResidualBlock:
    """Tests for ResidualBlock simulation."""
    
    def test_residual_block_matches_pytorch(self, device):
        """ResidualBlock should match PyTorch exactly."""
        from feature_extractor.cnn_simulator import ResidualBlockSim
        
        # Import original ResidualBlock
        TRANSPLAT_ROOT = SCARF_ROOT / 'transplat'
        sys.path.insert(0, str(TRANSPLAT_ROOT))
        from src.model.encoder.backbone.unimatch.backbone import ResidualBlock
        
        # Create PyTorch reference
        pytorch_block = ResidualBlock(64, 64, stride=1).to(device)
        
        # Create simulator with same weights
        sim = ResidualBlockSim(64, 64, stride=1, device=device)
        sim.load_from_pytorch(pytorch_block)
        
        # Test
        x = torch.randn(2, 64, 128, 128, device=device)
        
        with torch.no_grad():
            pytorch_out = pytorch_block(x)
            sim_out, cycles = sim.forward(x)
        
        psnr = compute_psnr(sim_out, pytorch_out)
        assert psnr > 100, f"PSNR {psnr:.2f} dB < 100 dB"
        assert cycles > 0


class TestCNNEncoder:
    """Tests for full CNN encoder simulation."""
    
    def test_cnn_encoder_matches_pytorch(self, device):
        """Full CNNEncoder should match PyTorch exactly."""
        from feature_extractor.cnn_simulator import CNNEncoderSimulator
        
        # Import original CNNEncoder
        TRANSPLAT_ROOT = SCARF_ROOT / 'transplat'
        sys.path.insert(0, str(TRANSPLAT_ROOT))
        from src.model.encoder.backbone.unimatch.backbone import CNNEncoder
        
        # Create PyTorch reference
        pytorch_encoder = CNNEncoder(output_dim=128, num_output_scales=1).to(device)
        
        # Create simulator with same weights
        config = CNNConfig(output_dim=128, num_output_scales=1)
        sim = CNNEncoderSimulator(config, device=device)
        sim.load_from_pytorch(pytorch_encoder)
        
        # Test
        x = torch.randn(2, 3, 256, 256, device=device)
        
        with torch.no_grad():
            pytorch_out = pytorch_encoder(x)
            if isinstance(pytorch_out, list):
                pytorch_out = pytorch_out[0]
            sim_out, cycles = sim.forward(x)
        
        psnr = compute_psnr(sim_out, pytorch_out)
        assert psnr > 100, f"CNNEncoder PSNR {psnr:.2f} dB < 100 dB"
        assert cycles > 0, "Should report cycles"
    
    def test_cycle_counting(self, device):
        """Test that cycle counting is reasonable."""
        from feature_extractor.cnn_simulator import CNNEncoderSimulator
        
        config = CNNConfig(output_dim=128)
        sim = CNNEncoderSimulator(config, device=device)
        
        x = torch.randn(1, 3, 256, 256, device=device)
        _, cycles = sim.forward(x)
        
        # Rough estimate: should be in millions of cycles
        assert cycles > 100000, f"Cycles {cycles} seems too low"
        assert cycles < 100000000, f"Cycles {cycles} seems too high"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
