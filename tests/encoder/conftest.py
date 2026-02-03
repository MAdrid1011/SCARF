"""
Pytest fixtures for encoder unit tests.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add SCARF root to path
SCARF_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SCARF_ROOT))

from encoder.types import EncoderConfig, ConvConfig, GEMMConfig, BilinearConfig


@pytest.fixture
def default_config():
    """Default encoder configuration."""
    return EncoderConfig()


@pytest.fixture
def transplat_config():
    """Transplat model configuration."""
    return EncoderConfig.transplat_preset()


@pytest.fixture
def mvsplat_config():
    """MVSplat model configuration."""
    return EncoderConfig.mvsplat_preset()


@pytest.fixture
def depthsplat_config():
    """DepthSplat model configuration."""
    return EncoderConfig.depthsplat_preset()


@pytest.fixture
def small_conv_config():
    """Small convolution config for fast tests."""
    return ConvConfig(pe_array_size=4)


@pytest.fixture
def small_gemm_config():
    """Small GEMM config for fast tests."""
    return GEMMConfig(tile_m=4, tile_n=8, array_m=4, array_n=8)


@pytest.fixture
def device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def random_tensor_2d():
    """Factory for random 2D tensors."""
    def _create(batch=1, channels=64, height=32, width=32, device='cpu'):
        return torch.randn(batch, channels, height, width, device=device)
    return _create


@pytest.fixture
def random_tensor_1d():
    """Factory for random 1D tensors."""
    def _create(batch=1, seq_len=1024, dim=128, device='cpu'):
        return torch.randn(batch, seq_len, dim, device=device)
    return _create


@pytest.fixture
def conv_weight():
    """Factory for conv weights."""
    def _create(out_channels=128, in_channels=64, kernel_size=3):
        return torch.randn(out_channels, in_channels, kernel_size, kernel_size)
    return _create
