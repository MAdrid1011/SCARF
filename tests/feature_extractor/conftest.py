"""
Test fixtures for feature extractor tests.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add SCARF to path
SCARF_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SCARF_ROOT))


@pytest.fixture
def device():
    """Get test device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_image(device):
    """Sample input image [B, V, C, H, W]."""
    return torch.randn(1, 2, 3, 256, 256, device=device)


@pytest.fixture
def sample_features(device):
    """Sample features [B, V, C, H, W]."""
    return torch.randn(1, 2, 128, 32, 32, device=device)


@pytest.fixture
def cnn_weights(device):
    """Sample CNN weights for testing."""
    return {
        'conv1': torch.randn(64, 3, 7, 7, device=device),
        'norm1_weight': torch.ones(64, device=device),
        'norm1_bias': torch.zeros(64, device=device),
    }
